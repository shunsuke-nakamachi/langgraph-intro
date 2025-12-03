import re
from typing import TypedDict, List, Annotated

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

# 環境変数読み込み
load_dotenv()


class VerifyState(TypedDict):
    question: str
    evidence: str
    score: float
    verdict: str
    reason: str
    prompt: str
    messages: Annotated[List[BaseMessage], list.__add__]


def build_prompt_node(state: VerifyState):
    """質問と証拠をまとめたプロンプト文字列を返す"""
    prompt = f"""
    あなたはファクトチェッカーです。以下の証拠が質問に答えるのに十分か厳しく評価してください。
    必ず次のJSON形式で1個だけ出力してください。
    {{
      "Score": 0-100 (整数),
      "Verdict": "good" or "needs_fix",
      "Reason": "一行の根拠"
    }}

    質問: {state['question']}
    証拠: {state['evidence']}
    """
    return {"prompt": prompt}


def call_llm_node(state: VerifyState):
    """LLMに評価させる"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    resp = llm.invoke([HumanMessage(content=state["prompt"])])
    return {"reason": resp.content}


def parse_score_node(state: VerifyState):
    """LLM出力からスコアと verdict を抽出"""
    text = state["reason"]
    score = 0
    verdict = "needs_fix"
    # float/ int 両対応
    score_match = re.search(r"Score\\s*[:：]\\s*(\\d+(?:\\.\\d+)?)", text, re.I)
    if score_match:
        try:
            score = float(score_match.group(1))
        except ValueError:
            score = 0
    verdict_match = re.search(r"Verdict\\s*[:：]\\s*(good|needs_fix)", text, re.I)
    if verdict_match:
        verdict = verdict_match.group(1).lower()
    # スコアを0-100にクリップし整数化
    score = int(max(0, min(100, round(score))))
    # verdict が無ければスコアで決定
    if verdict not in ("good", "needs_fix"):
        verdict = "good" if score >= 70 else "needs_fix"
    # verdict とスコアが矛盾する場合は verdict を優先しスコアを補正
    if verdict == "good" and score < 70:
        score = 70
    if verdict == "needs_fix" and score > 69:
        score = 69
    return {"score": score, "verdict": verdict}


def output_node(state: VerifyState):
    reason_short = state.get("reason", "")[:80]
    msg = AIMessage(content=f"✅ Score: {state['score']} Verdict: {state['verdict']} Reason: {reason_short}")
    return {"messages": [msg]}


def build_verify_app():
    builder = StateGraph(VerifyState)
    builder.add_node("build_prompt", build_prompt_node)
    builder.add_node("call_llm", call_llm_node)
    builder.add_node("parse_score", parse_score_node)
    builder.add_node("output", output_node)

    builder.add_edge(START, "build_prompt")
    builder.add_edge("build_prompt", "call_llm")
    builder.add_edge("call_llm", "parse_score")
    builder.add_edge("parse_score", "output")
    builder.add_edge("output", END)

    return builder.compile()


__all__ = ["build_verify_app"]
