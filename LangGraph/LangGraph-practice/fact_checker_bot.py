import sys
import operator
from typing import TypedDict, List, Annotated

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from search_subgraph import build_search_app
from verify_subgraph import build_verify_app

# 環境変数読み込み（OPENAI_API_KEY など）
load_dotenv()

# 定数
MAX_LOOPS = 4
GOOD_THRESHOLD = 70


class MainState(TypedDict):
    question: str
    context: str
    score: float
    verdict: str
    loop_count: int
    messages: Annotated[List[BaseMessage], operator.add]


def router_node(state: MainState):
    """ループ継続か終了かを決める"""
    if state.get("verdict") == "good" or state.get("loop_count", 0) >= MAX_LOOPS:
        return {"route": "finalizer"}
    return {"route": "search"}


def call_search(state: MainState):
    """検索サブグラフを呼び出す"""
    search_app = build_search_app()
    res = search_app.invoke({"query": state["question"], "results": "", "summary": "", "messages": []})
    # サマリを context に保存
    context_text = res.get("summary", "")
    msgs = res.get("messages", [])
    return {"context": context_text, "messages": msgs}


def call_verify(state: MainState):
    """検証サブグラフを呼び出す"""
    verify_app = build_verify_app()
    res = verify_app.invoke({
        "question": state["question"],
        "evidence": state.get("context", ""),
        "score": 0,
        "verdict": "needs_fix",
        "reason": "",
        "messages": []
    })
    return {
        "score": res.get("score", 0),
        "verdict": res.get("verdict", "needs_fix"),
        "messages": res.get("messages", [])
    }


def finalizer_node(state: MainState):
    """最終回答を生成"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    sys_msg = SystemMessage(content="あなたはプロのファクトチェッカーです。収集した事実に基づいて簡潔に答えてください。")
    human_msg = HumanMessage(content=f"質問: {state['question']}\n\n収集した証拠:\n{state.get('context','')}")
    resp = llm.invoke([sys_msg, human_msg])
    return {"messages": [resp]}


def reflection_node(state: MainState):
    """verify結果を見て verdict をセットし、ループ回数を進める"""
    score = state.get("score", 0)
    verdict = "good" if score >= GOOD_THRESHOLD else "needs_fix"
    return {"verdict": verdict, "loop_count": state.get("loop_count", 0) + 1}


def build_main_app():
    builder = StateGraph(MainState)
    builder.add_node("router", router_node)
    builder.add_node("search", call_search)
    builder.add_node("verify", call_verify)
    builder.add_node("reflector", reflection_node)
    builder.add_node("finalizer", finalizer_node)

    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        lambda s: s.get("route", "search"),
        {"search": "search", "finalizer": "finalizer"}
    )
    builder.add_edge("search", "verify")
    builder.add_edge("verify", "reflector")
    builder.add_edge("reflector", "router")
    builder.add_edge("finalizer", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory, interrupt_before=["finalizer"])


def main():
    question = sys.argv[1] if len(sys.argv) > 1 else "地球温暖化は本当に起きているか？"
    print(f"\n--- Fact-Checker 開始 ---\n質問: {question}\n")
    app = build_main_app()
    thread = {"configurable": {"thread_id": "fact_checker_demo"}}

    state = {
        "question": question,
        "context": "",
        "score": 0,
        "verdict": "needs_fix",
        "loop_count": 0,
        "messages": []
    }

    # ストリーミング実行（finalizer手前で停止）
    for event in app.stream(state, config=thread):
        for node, val in event.items():
            if not val:
                continue
            if node == "finalizer":
                continue
            if "messages" in val:
                for m in val["messages"]:
                    print(f"[{node}] {m.content}")
            if "score" in val:
                print(f"[{node}] score={val['score']}")

    snap = app.get_state(thread)
    if snap.next:
        print("\n🛑 最終回答の前で停止しました。生成してよいですか？ (y/n): ", end="", flush=True)
        choice = sys.stdin.readline().strip().lower()
        if choice == "y":
            print("再開します...\n")
            for event in app.stream(None, config=thread):
                if "finalizer" in event:
                    msg = event["finalizer"]["messages"][-1].content
                    print(f"--- 最終回答 ---\n{msg}\n")
        else:
            print("中止しました。")
    else:
        print("完了しました。")


if __name__ == "__main__":
    main()
