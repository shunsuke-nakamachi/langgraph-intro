"""
事実検証サブグラフ：質問の主張が根拠テキストによって裏付けられているかを、
LLM に「エンテイルメント（含意）」という狭いタスクとして判定させる。
「このソースは信頼できるか」という、LLM に答えられない問いは投げない。
複数ドメインでの裏取り（corroboration）はコードで決定論的に数える。

(Verify subgraph: asks the LLM the narrow, checkable question of whether the
question's claim is entailed by the retrieved evidence text — never the
unanswerable "is this source credible?" question. Cross-domain corroboration
is counted deterministically in plain Python, not by the LLM.)
"""

from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Literal
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

from utils import log_event
from search_subgraph import Evidence

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")


class Verdict(TypedDict):
    """検証結果。(The result of verification.)

    direction と confidence を分けているのは、「主張は真/偽/不明のどれか」（方向）と
    「その判断はどれだけ強い根拠に基づくか」（強さ）を別軸として扱うため。
    以前は単一の score フィールドで両方を表そうとしていたが、LLM が「結論への自信」と
    「真/偽の向き」を混同し、明確に偽と判断した主張に対してもスコアを高く返す、という
    バグがライブテストで実際に発生したため分離した。
    (direction and confidence are kept separate because "is the claim true, false, or
    unclear" (direction) and "how strong is that judgment" (magnitude) are genuinely
    different axes. A single score field previously tried to represent both, and the
    LLM actually conflated "confidence in my conclusion" with "true/false direction" —
    scoring a claim it had clearly judged false as high — a real bug caught in live
    testing. Splitting them structurally prevents that conflation.)
    """
    direction: str                # "true" | "false" | "unclear"
    confidence: float              # 0.0-1.0, direction の判断がどれだけ強い根拠に基づくか (how strong the evidence is for that direction)
    corroborating_domains: int    # 主張を支持する tier<=2 の異なるドメイン数 (distinct tier<=2 supporting domains)
    supporting_urls: List[str]    # 主張を支持すると判定された URL (URLs judged to support the claim)
    refuting_domains: int         # 主張に反証する tier<=2 の異なるドメイン数 (distinct tier<=2 refuting domains)
    refuting_urls: List[str]      # 主張に反証すると判定された URL (URLs judged to contradict the claim)
    decision: str                 # "good" | "refuted" | "needs_fix" | "escalate"（reflector が設定 / set by reflector）
    rationale: str
    search_suggestion: str        # 再検索用の短い補足クエリ（自由記述の rationale をそのまま検索に使わない）
                                   # (short supplementary query for re-search — never feed free-text rationale straight into search)


class EvidenceJudgement(BaseModel):
    """1件の根拠に対する、LLM による狭い判定。「支持」と「反証」は別軸であり、
    両方 False（無関係・言及なし）もあり得る。
    (A single narrow per-evidence judgement from the LLM. "Supports" and
    "contradicts" are independent axes — both can be False when the evidence
    is simply irrelevant or silent on the claim.)"""
    url: str = Field(description="判定対象の根拠の URL (the URL of the evidence being judged)")
    supports_claim: bool = Field(description="この根拠は主張を直接支持しているか (does this evidence directly support the claim)")
    contradicts_claim: bool = Field(description="この根拠は主張に直接反証しているか (does this evidence directly contradict/refute the claim)")


class LLMVerdict(BaseModel):
    """LLM の構造化出力全体。(The LLM's full structured output.)"""
    direction: Literal["true", "false", "unclear"] = Field(description=(
        "根拠全体を踏まえて、主張は真・偽・不明のどれか。これは方向を表す値であり、"
        "確信の強さは別途 confidence で表す。"
        "(Which way the evidence points, overall: is the claim true, false, or unclear. "
        "This is the direction only — how strong that judgment is goes in confidence, "
        "separately.)"
    ))
    confidence: float = Field(description=(
        "direction の判断が根拠によってどれだけ強く裏付けられているかを 0.0〜1.0 で表す。"
        "direction が \"false\" であっても、根拠が明確にそれを示していれば confidence は"
        "高くしてよい（例: 明確に偽だと示す強い根拠がある場合は confidence=0.9 のようになる）。"
        "根拠が薄い、あるいは真偽が拮抗している場合は confidence を低くすること。"
        "(0.0-1.0: how strongly the evidence backs up whatever `direction` you chose — "
        "NOT how strongly it backs up \"true\" specifically. If direction is \"false\" and "
        "the evidence clearly shows that, confidence should be HIGH (e.g. 0.9), not low. "
        "Use a low confidence only when evidence is thin or genuinely conflicting.)"
    ))
    rationale: str = Field(description="判断の理由。不足している場合は何が足りないか明記する (reason for the direction/confidence; if insufficient, state what's missing)")
    search_suggestion: str = Field(description=(
        "まだ見つかっていない可能性のある独立したソースを探すための、検索エンジンに"
        "そのまま投げられる短い補足キーワード（3〜8語程度）を必ず入れること。"
        "文章ではなくキーワードで。空文字列にはしないこと — 再検索が必要かどうかは"
        "スコアだけでなく複数ドメインでの裏取り状況も踏まえて別の仕組みが判断するため、"
        "スコアの高さだけを理由にここを空にしないこと。"
        "(Always provide a short, search-engine-ready supplementary keyword phrase "
        "(roughly 3-8 words) aimed at finding an independent source not yet found. Keywords, "
        "not a sentence — this gets fed directly into a search API. Never leave this empty: "
        "whether a re-search actually happens is decided elsewhere based on direction, "
        "confidence, AND cross-domain corroboration together, not on any one of them alone, "
        "so don't leave this blank just because confidence looks high.)"
    ))
    evidence_judgements: List[EvidenceJudgement]


structured_llm = llm.with_structured_output(LLMVerdict)


class VerifyState(TypedDict):
    """事実検証サブグラフの State。(State for the verify subgraph.)"""
    question: str
    evidence: List[Evidence]
    prompt: str
    llm_verdict: dict  # LLMVerdict.model_dump() の結果 (the result of LLMVerdict.model_dump())
    verdict: Verdict
    messages: Annotated[list, add_messages]


def build_prompt(state: VerifyState) -> dict:
    """
    「この根拠はこの主張を支持しているか」という狭い質問のプロンプトを組み立てる。
    「このソースは信頼できるか」は聞かない。
    (Builds the narrow "does this evidence support this claim" prompt.
    Never asks "is this source trustworthy?".)

    Args:
        state (VerifyState): 検証サブグラフの現在の State (current verify subgraph state)

    Returns:
        dict: 更新された `prompt` (dict with the updated `prompt`)
    """
    evidence_block = "\n\n".join(
        f"URL: {e['url']}\nドメイン (domain): {e['domain']}\n本文抜粋 (excerpt): {e['snippet']}"
        for e in state["evidence"]
    ) or "(根拠なし / no evidence retrieved)"

    prompt = f"""以下の質問に対する主張を検証してください。
質問 (question): {state['question']}

各根拠について、その本文だけを根拠に、次の2つを独立に判定してください。
1. 「主張を直接支持しているか」(supports_claim)
2. 「主張に直接反証しているか」(contradicts_claim) — 主張が誤りであることを示す場合はこちらを true にする
関連しない、あるいは何も言及していない根拠は両方 false としてください。
ソースの評判やドメインの信頼性を推測して判定しないこと（それは別の仕組みで扱います）。

search_suggestion には、まだ見つかっていない可能性のある独立したソースを探すための、
検索エンジン向けの短いキーワード（3〜8語）を必ず入れてください。空文字列にしたり、
元の質問文をそのまま繰り返したり、長い説明文にしたりしないこと。
(Always put a short (3-8 word) search-engine-style keyword phrase in search_suggestion,
aimed at finding an independent source not yet found. Never leave it empty, just repeat
the original question, or write a full sentence.)

根拠 (evidence):
{evidence_block}
"""
    return {"prompt": prompt}


async def call_llm(state: VerifyState) -> dict:
    """
    構造化出力で LLM を呼び出し、direction・confidence と各根拠のエンテイルメント判定を得る。
    (Calls the LLM with structured output to get direction, confidence, and per-evidence
    entailment judgements.)

    Args:
        state (VerifyState): 検証サブグラフの現在の State (current verify subgraph state)

    Returns:
        dict: 更新された `llm_verdict`（State に流す前にプレーンな dict へ変換する。
              素の Pydantic モデルを State に置くと、astream_events が内部で使う
              ログ用トレーサーがシリアライズできず警告を出すため）
              (dict with the updated `llm_verdict`; converted to a plain dict before
              entering State — storing a raw Pydantic model in State makes the log
              tracer underlying astream_events fail to serialize it and emit a warning)
    """
    result = await structured_llm.ainvoke([HumanMessage(content=state["prompt"])])
    log_event("call_llm", f"direction={result.direction} confidence={result.confidence:.2f}")
    return {"llm_verdict": result.model_dump()}


def parse_verdict(state: VerifyState) -> dict:
    """
    LLM の判定から、tier<=2 の異なるドメインで裏取りできた件数をコードで
    決定論的に数える。裏取りの判定を LLM の言葉に委ねない。
    (Deterministically counts, in plain Python, how many distinct tier<=2
    domains corroborate the claim — corroboration is never left to the LLM's
    own words.)

    Args:
        state (VerifyState): 検証サブグラフの現在の State (current verify subgraph state)

    Returns:
        dict: 更新された `verdict`（`decision` は空のまま。reflector が設定する）
              (dict with the updated `verdict`; `decision` is left blank for reflector to set)
    """
    llm_verdict = state["llm_verdict"]
    evidence_by_url = {e["url"]: e for e in state["evidence"]}

    def _distinct_tier12_domains(urls: List[str]) -> set:
        return {
            evidence_by_url[url]["domain"]
            for url in urls
            if url in evidence_by_url and evidence_by_url[url]["tier"] <= 2
        }

    judgements = llm_verdict["evidence_judgements"]
    supporting_urls = [j["url"] for j in judgements if j["supports_claim"]]
    refuting_urls = [j["url"] for j in judgements if j["contradicts_claim"]]
    supporting_domains = _distinct_tier12_domains(supporting_urls)
    refuting_domains = _distinct_tier12_domains(refuting_urls)

    verdict: Verdict = {
        "direction": llm_verdict["direction"],
        "confidence": llm_verdict["confidence"],
        "corroborating_domains": len(supporting_domains),
        "supporting_urls": supporting_urls,
        "refuting_domains": len(refuting_domains),
        "refuting_urls": refuting_urls,
        "decision": "",
        "rationale": llm_verdict["rationale"],
        "search_suggestion": llm_verdict["search_suggestion"],
    }
    log_event(
        "parse_verdict",
        f"direction={llm_verdict['direction']} confidence={llm_verdict['confidence']:.2f} "
        f"corroborating_domains={len(supporting_domains)} ({supporting_domains}) "
        f"refuting_domains={len(refuting_domains)} ({refuting_domains})",
    )
    return {"verdict": verdict}


def output(state: VerifyState) -> dict:
    """
    検証結果の要約をメッセージ履歴に追加し、メイングラフへ返すノード。
    (Node that appends a summary of the verdict to the message history,
    returning control to the main graph.)

    Args:
        state (VerifyState): 検証サブグラフの現在の State (current verify subgraph state)

    Returns:
        dict: 更新された `messages` (dict with the updated `messages`)
    """
    v = state["verdict"]
    summary = (
        f"[verify] direction={v['direction']} confidence={v['confidence']:.2f} "
        f"corroborating_domains={v['corroborating_domains']} refuting_domains={v['refuting_domains']}"
    )
    return {"messages": [AIMessage(content=summary)]}


builder = StateGraph(VerifyState)
builder.add_node("build_prompt", build_prompt)
builder.add_node("call_llm", call_llm)
builder.add_node("parse_verdict", parse_verdict)
builder.add_node("output", output)

builder.add_edge(START, "build_prompt")
builder.add_edge("build_prompt", "call_llm")
builder.add_edge("call_llm", "parse_verdict")
builder.add_edge("parse_verdict", "output")
builder.add_edge("output", END)

verify_graph = builder.compile()
