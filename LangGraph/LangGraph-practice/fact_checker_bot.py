"""
ファクトチェッカー メイングラフ。
検索 → 検証 → 決定論的な reflector 判定 → (承認 / 人間による確認 / 再検索) のループ。

信頼性についての設計方針:
  - LLM に「このソースは信頼できるか」を判断させない（判断できないタスクだから）。
  - 代わりに、静的なドメイン tier 表と複数ドメインでの裏取り件数という、
    コードで検証可能な指標だけで良否を決める（reflector は LLM を呼ばない）。
  - 「支持」と「反証」は別軸で追跡する。反証が十分に裏付けられれば "refuted"
    （主張は誤り）と判定し、"good" に到達できないまま無意味にループし続けない。
  - 規制の厳しい業界を想定し、人間によるレビューを既定側に倒す。
    自動承認は「根拠が明確に強い場合」のみの例外とする。

(Fact-checker main graph. search -> verify -> deterministic reflector decision
-> (approve / human review / re-search) loop.

Design stance on trust:
  - The LLM is never asked "is this source credible?" (it can't answer that).
  - Instead, "good" is decided only from code-checkable signals: the static
    domain-tier table and a cross-domain corroboration count (reflector makes
    no LLM call at all).
  - "Supports" and "contradicts" are tracked as independent axes. A
    well-corroborated contradiction becomes "refuted" (the claim is false),
    rather than spinning uselessly toward a "good" verdict it can never reach.
  - Given a highly regulated industry, human review is the default, not the
    exception. Auto-approval only happens when the evidence is unambiguously
    strong.)
"""

import asyncio
import re
import sys
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Optional
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

from utils import log_event
from search_subgraph import search_graph, Evidence
from verify_subgraph import verify_graph, Verdict

load_dotenv()

MAX_LOOPS = 4
# 自動承認のしきい値: 十分に強い根拠がある場合のみ人間の確認を省略する。
# それ以外（"good" だが根拠が薄い場合、"escalate" の場合）は必ず人間に回す。
# (Auto-approve threshold: skip human review only when evidence is unambiguously
# strong. Every other case — a "good" verdict on thinner evidence, or "escalate" —
# always goes to a human.)
AUTO_APPROVE_CONFIDENCE = 0.9
AUTO_APPROVE_DOMAINS = 3


class MainState(TypedDict):
    """メイングラフの State。(State for the main graph.)"""
    messages: Annotated[list, add_messages]
    question: str
    loop_count: int
    retry_hint: Optional[str]
    evidence: Annotated[List[Evidence], operator.add]
    verdict: Optional[Verdict]
    human_decision: Optional[str]  # "approve" | "redo"（human_review ノードの手前で外部から設定される / set externally before the human_review node runs）


def router(state: MainState) -> dict:
    """
    分岐のためだけに存在するパススルーノード。State は変更しない。
    (Pass-through node that exists only as a branching point. Does not modify state.)
    """
    return {}


def route_from_router(state: MainState) -> str:
    """
    verdict の有無・内容に応じて次のノードを決める条件分岐関数。
    (Conditional-edge function that picks the next node based on the verdict, if any.)

    Args:
        state (MainState): メイングラフの現在の State (current main graph state)

    Returns:
        str: "search" | "human_review" | "finalizer"
    """
    verdict = state.get("verdict")
    if verdict is None:
        return "search"  # 初回実行 (first run, nothing to evaluate yet)

    decision = verdict["decision"]
    if decision == "needs_fix":
        return "search"
    if decision == "escalate":
        return "human_review"
    if decision == "refuted":
        # 反証が非常に強い場合のみ自動確定。それ以外は必ず人間の確認へ。
        # (Auto-finalize only when the refutation itself is unambiguously strong.
        # Otherwise always route to a human.)
        if verdict["refuting_domains"] >= AUTO_APPROVE_DOMAINS:
            return "finalizer"
        return "human_review"
    # decision == "good"
    if verdict["confidence"] >= AUTO_APPROVE_CONFIDENCE and verdict["corroborating_domains"] >= AUTO_APPROVE_DOMAINS:
        return "finalizer"
    return "human_review"


async def search(state: MainState) -> dict:
    """
    search_subgraph を呼び出すラッパー。ループ回数をここでインクリメントする。
    既出の URL は除外してから evidence に追加する
    （`evidence` は operator.add でループを跨いで蓄積されるため、
    重複除去しないと同じ URL が再検索のたびに繰り返し積み重なり、
    最終回答の引用リストが同じソースを何度も表示してしまう。実際にライブテストで発生）。
    (Wrapper that invokes search_subgraph. Increments the loop counter here.
    Filters out URLs already seen before merging into `evidence` — since
    `evidence` accumulates across loops via operator.add, without dedup the
    same URL piles up every time a re-search happens to surface it again,
    and the final citation list ends up repeating the same source many
    times. This actually happened in live testing.)

    Args:
        state (MainState): メイングラフの現在の State (current main graph state)

    Returns:
        dict: 更新された `evidence`（新規 URL のみ）・`loop_count`・`messages`
              (dict with updated `evidence` (new URLs only), `loop_count`, `messages`)
    """
    result = await search_graph.ainvoke({
        "question": state["question"],
        "retry_hint": state.get("retry_hint"),
    })
    existing_urls = {e["url"] for e in state["evidence"]}
    new_evidence = [e for e in result["evidence"] if e["url"] not in existing_urls]
    return {
        "evidence": new_evidence,
        "loop_count": state["loop_count"] + 1,
        "messages": result["messages"],
    }


async def verify(state: MainState) -> dict:
    """
    verify_subgraph を呼び出すラッパー。
    (Wrapper that invokes verify_subgraph.)

    Args:
        state (MainState): メイングラフの現在の State (current main graph state)

    Returns:
        dict: 更新された `verdict`・`messages` (dict with the updated `verdict`, `messages`)
    """
    result = await verify_graph.ainvoke({
        "question": state["question"],
        "evidence": state["evidence"],
    })
    return {"verdict": result["verdict"], "messages": result["messages"]}


def reflector(state: MainState) -> dict:
    """
    決定論的なしきい値判定。LLM は一切呼び出さない。
    direction == "true" かつ confidence >= 0.7 かつ corroborating_domains >= 2 の
    場合のみ "good"。direction == "false" かつ confidence >= 0.7 かつ
    refuting_domains >= 2 の場合は "refuted"（主張は誤り）とし、これは "good" に
    到達できないまま無意味にループし続けるのを防ぐため "needs_fix" より優先される。
    ループ上限に達した "needs_fix" は、黙って確定させず必ず "escalate" にする。
    (Deterministic threshold check — makes no LLM call. "good" only when
    direction == "true" AND confidence >= 0.7 AND corroborating_domains >= 2.
    "refuted" when direction == "false" AND confidence >= 0.7 AND
    refuting_domains >= 2 — checked before "needs_fix" so a false claim doesn't
    just spin toward the loop cap since it can never become "good". A
    "needs_fix" that has hit the loop cap is always turned into "escalate"
    rather than silently finalized.)

    Args:
        state (MainState): メイングラフの現在の State (current main graph state)

    Returns:
        dict: 更新された `verdict`（必要なら `retry_hint` も）
              (dict with the updated `verdict`, and `retry_hint` when retrying)
    """
    verdict = dict(state["verdict"])

    if verdict["direction"] == "true" and verdict["confidence"] >= 0.7 and verdict["corroborating_domains"] >= 2:
        verdict["decision"] = "good"
        updates = {"verdict": verdict}
    elif verdict["direction"] == "false" and verdict["confidence"] >= 0.7 and verdict["refuting_domains"] >= 2:
        verdict["decision"] = "refuted"
        updates = {"verdict": verdict}
    elif state["loop_count"] >= MAX_LOOPS:
        verdict["decision"] = "escalate"
        updates = {"verdict": verdict}
    else:
        verdict["decision"] = "needs_fix"
        # rationale（自由記述の説明文）ではなく search_suggestion（検索用の短いキーワード）を使う。
        # rationale をそのまま検索クエリに使うと、ライブテストで実際に検索が脱線した
        # （"is" という単語の辞書的definitionの検索結果が返ってきた）。
        # (Use search_suggestion — a short search-ready keyword phrase — not the free-text
        # rationale. Feeding the raw rationale into search actually derailed a live test run,
        # returning results about the dictionary definition of the word "is".)
        updates = {"verdict": verdict, "retry_hint": verdict["search_suggestion"]}

    log_event(
        "reflector",
        f"direction={verdict['direction']} confidence={verdict['confidence']:.2f} "
        f"domains={verdict['corroborating_domains']} refuting_domains={verdict['refuting_domains']} "
        f"loop={state['loop_count']} -> {verdict['decision']}",
    )
    return updates


def human_review(state: MainState) -> dict:
    """
    パススルーノード。実際の承認/却下の入力は、このノードの手前の interrupt と
    再開の間に、外部の実行コード（__main__ 部分）が収集して State に書き込む。
    (Pass-through node. The actual approve/reject input is collected by the outer
    driver code between the interrupt before this node and the resume, and written
    into state from outside.)
    """
    return {}


def route_after_human_review(state: MainState) -> str:
    """
    human_decision に応じて finalizer か再検索かを決める条件分岐関数。
    (Conditional-edge function that routes to finalizer or back to search,
    based on human_decision.)

    Args:
        state (MainState): メイングラフの現在の State (current main graph state)

    Returns:
        str: "finalizer" | "search"
    """
    return "finalizer" if state.get("human_decision") == "approve" else "search"


def finalizer(state: MainState) -> dict:
    """
    主張を実際に支持した根拠（"refuted" の場合は反証した根拠）だけを引用として
    添えて、最終回答を組み立てる。
    (Composes the final answer, citing only the evidence that actually
    supported the claim — or, when the decision is "refuted", the evidence
    that actually contradicted it.)

    Args:
        state (MainState): メイングラフの現在の State (current main graph state)

    Returns:
        dict: 更新された `messages`（最終回答を含む） (dict with the updated `messages`, containing the final answer)
    """
    verdict = state["verdict"]
    if verdict["decision"] == "refuted":
        header = "この主張は根拠によって否定されています (this claim is contradicted by the evidence):"
        cited_urls = verdict["refuting_urls"]
        stats = f"[direction={verdict['direction']}, confidence={verdict['confidence']:.2f}, refuting_domains={verdict['refuting_domains']}]"
        no_sources_msg = "（反証する情報源なし / no refuting sources found）"
    else:
        header = "根拠 (sources):"
        cited_urls = verdict["supporting_urls"]
        stats = f"[direction={verdict['direction']}, confidence={verdict['confidence']:.2f}, corroborating_domains={verdict['corroborating_domains']}]"
        no_sources_msg = "（根拠となる情報源なし / no supporting sources found）"

    cited = [e for e in state["evidence"] if e["url"] in cited_urls]
    citations = "\n".join(f"- {e['title']} ({e['url']})" for e in cited) or no_sources_msg

    answer = f"{verdict['rationale']}\n\n{header}\n{citations}\n\n{stats}"
    log_event("finalizer", "最終回答を確定 (finalizing answer)")
    return {"messages": [AIMessage(content=answer)]}


builder = StateGraph(MainState)
builder.add_node("router", router)
builder.add_node("search", search)
builder.add_node("verify", verify)
builder.add_node("reflector", reflector)
builder.add_node("human_review", human_review)
builder.add_node("finalizer", finalizer)

builder.add_edge(START, "router")
builder.add_conditional_edges("router", route_from_router, {
    "search": "search",
    "human_review": "human_review",
    "finalizer": "finalizer",
})
builder.add_edge("search", "verify")
builder.add_edge("verify", "reflector")
builder.add_edge("reflector", "router")
builder.add_conditional_edges("human_review", route_after_human_review, {
    "finalizer": "finalizer",
    "search": "search",
})
builder.add_edge("finalizer", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["human_review"])


def _slugify(text: str) -> str:
    """質問文から thread_id を作る。(Derives a thread_id from the question text.)"""
    slug = re.sub(r"[^\w]+", "-", text.strip().lower()).strip("-")
    return slug[:60] or "question"


STREAM_NODES = {"search", "verify", "reflector", "finalizer"}


async def run(question: str) -> None:
    """
    ファクトチェッカーを実行する。同じ質問なら thread_id が一致し、
    永続化された State から再開する（MemorySaver）。
    (Runs the fact-checker. The same question maps to the same thread_id, so a
    re-run resumes from persisted state via MemorySaver.)

    Args:
        question (str): ユーザーの質問 (the user's question)
    """
    thread_id = _slugify(question)
    config = {"configurable": {"thread_id": thread_id}}
    print(f"--- Fact-Checker 開始 (thread_id={thread_id}) ---")

    existing = await graph.aget_state(config)
    input_state = None
    if not existing.values:
        input_state = {
            "messages": [HumanMessage(content=question)],
            "question": question,
            "loop_count": 0,
            "retry_hint": None,
            "evidence": [],
            "verdict": None,
            "human_decision": None,
        }

    while True:
        async for event in graph.astream_events(input_state, config, version="v1"):
            if event["event"] == "on_chain_end" and event["name"] in STREAM_NODES:
                output = event["data"].get("output")
                if isinstance(output, dict) and output.get("messages"):
                    log_event(event["name"], output["messages"][-1].content)

        snapshot = await graph.aget_state(config)
        if not snapshot.next:
            break  # グラフが END に到達 (graph reached END)

        # interrupt_before=["human_review"] で一時停止している (paused before human_review)
        verdict = snapshot.values["verdict"]
        print("\n--- 人間による確認が必要です (human review required) ---")
        print(
            f"decision={verdict['decision']} direction={verdict['direction']} confidence={verdict['confidence']:.2f} "
            f"corroborating_domains={verdict['corroborating_domains']} refuting_domains={verdict['refuting_domains']}"
        )
        print(f"rationale: {verdict['rationale']}")
        answer = input("承認しますか？ y=承認(approve) / n=却下・再検索(reject & redo): ").strip().lower()
        await graph.aupdate_state(config, {"human_decision": "approve" if answer == "y" else "redo"})
        input_state = None  # None を渡して再開する (resume by passing None)

    final = await graph.aget_state(config)
    print("\n--- 完了 (done) ---")
    print(final.values["messages"][-1].content)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('使い方 (usage): python fact_checker_bot.py "質問文 (your question)"')
        sys.exit(1)
    asyncio.run(run(sys.argv[1]))
