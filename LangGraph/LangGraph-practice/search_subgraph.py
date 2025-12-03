import json
import datetime
from typing import TypedDict, List, Annotated

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END

# 環境変数読み込み（OPENAI_API_KEY など）
load_dotenv()


class SearchState(TypedDict):
    query: str
    goal: str
    constraints: str
    search_queries: List[str]   # multiple query variants (positive/negative/authority)
    results: str
    summary: str
    messages: Annotated[List[BaseMessage], list.__add__]


def prepare_query_node(state: SearchState):
    """初期化"""
    return {
        "query": state["query"],
        "goal": "",
        "constraints": "",
        "search_queries": [state["query"]],
    }


def extract_goal_node(state: SearchState):
    """PGC: 質問から目的と制約を抽出"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    sys = SystemMessage(
        content="あなたはGoal Extractorです。出力はJSONのみ。keys: goal(80字以内), constraints(箇条書き1行可)。"
    )
    hum = HumanMessage(content=f"質問: {state['query']}\n目的と制約を抽出してください。")
    resp = llm.invoke([sys, hum]).content
    try:
        data = json.loads(resp)
    except Exception:
        data = {"goal": state["query"], "constraints": ""}
    return {
        "goal": data.get("goal", state["query"]),
        "constraints": data.get("constraints", ""),
    }


def rewrite_query_llm_node(state: SearchState):
    """PGC: 肯定 / 反証 / 公的データ の3本クエリを生成"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    system = SystemMessage(
        content=(
            "あなたはウェブ検索クエリ生成器です。出力は3行のみ。\n"
            "1行目: Positive（主張を裏付ける情報）\n"
            "2行目: Negative（主張を疑う/デマ検証/批判）\n"
            "3行目: Authority（公的データ・論文・政府機関・大手メディア）\n"
            "装飾・説明・ナンバリングは禁止。日本語で。必要なら日付・地名・人名は保持。"
        )
    )
    human = HumanMessage(
        content=(
            f"Goal: {state['goal']}\nConstraints: {state['constraints']}\n"
            "日本語で3本の検索クエリだけを行ごとに返してください。"
        )
    )
    resp = llm.invoke([system, human])
    lines = [l.strip() for l in resp.content.splitlines() if l.strip()]
    # フォールバック: 足りなければ質問文を補充
    while len(lines) < 3:
        lines.append(state["query"])
    queries = lines[:3]
    msgs = [
        AIMessage(content=f"🔧 Positive: {queries[0]}"),
        AIMessage(content=f"🔧 Negative: {queries[1]}"),
        AIMessage(content=f"🔧 Authority: {queries[2]}"),
    ]
    return {"search_queries": queries, "messages": msgs}


def call_api_node(state: SearchState):
    """DuckDuckGoで検索しテキストを取得"""
    search = DuckDuckGoSearchRun()
    try:
        collected = []
        for q in state.get("search_queries", [state["query"]]):
            r = search.invoke(q)
            collected.append(f"### {q}\n{r}")
        res = "\n\n".join(collected)
    except Exception as e:
        res = f"Search error: {e}"
    return {"results": res}


def extract_snippet_node(state: SearchState):
    """検索結果から要約を作成"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        SystemMessage(content="検索結果を元に、質問への回答に役立つ事実だけを日本語で100-150文字に要約してください。"),
        HumanMessage(content=f"質問: {state['query']}\n\n検索結果:\n{state['results']}"),
    ]
    resp = llm.invoke(messages)
    return {"summary": resp.content}


def output_node(state: SearchState):
    """メインに返すメッセージを組み立て"""
    msgs = []
    if "messages" in state:
        msgs.extend(state["messages"])
    msgs.append(AIMessage(content=f"🔎 要約: {state['summary']}"))
    return {"messages": msgs}


def build_search_app():
    builder = StateGraph(SearchState)
    builder.add_node("prepare_query", prepare_query_node)
    builder.add_node("extract_goal", extract_goal_node)
    builder.add_node("rewrite_query", rewrite_query_llm_node)
    builder.add_node("call_api", call_api_node)
    builder.add_node("extract_snippet", extract_snippet_node)
    builder.add_node("output", output_node)

    builder.add_edge(START, "prepare_query")
    builder.add_edge("prepare_query", "extract_goal")
    builder.add_edge("extract_goal", "rewrite_query")
    builder.add_edge("rewrite_query", "call_api")
    builder.add_edge("call_api", "extract_snippet")
    builder.add_edge("extract_snippet", "output")
    builder.add_edge("output", END)

    return builder.compile()


__all__ = ["build_search_app"]
