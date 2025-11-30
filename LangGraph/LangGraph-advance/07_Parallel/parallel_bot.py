import asyncio
import operator
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END

# -------------------------------------------------
# 1. 環境設定
# -------------------------------------------------
load_dotenv()

# デフォルトの LLM（設定がない場合に使用）
default_llm = ChatOpenAI(model="gpt-4o-mini")

# -------------------------------------------------
# 2. State 定義
# -------------------------------------------------
class State(TypedDict):
    query: str
    results: Annotated[List[str], operator.add]
    answer: str

# -------------------------------------------------
# 3. ノード定義
# -------------------------------------------------

async def search_wikipedia(state: State):
    """Wikipedia を検索する（シミュレーション）"""
    print("  [Wikipedia] 検索中...")
    await asyncio.sleep(1)
    return {"results": [f"Wikipedia Result for '{state['query']}'"]}

async def search_news(state: State):
    """ニュースサイトを検索する（シミュレーション）"""
    print("  [News] 検索中...")
    await asyncio.sleep(1)
    return {"results": [f"News Result for '{state['query']}'"]}

async def search_blogs(state: State):
    """ブログを検索する（シミュレーション）"""
    print("  [Blogs] 検索中...")
    await asyncio.sleep(1)
    return {"results": [f"Blog Result for '{state['query']}'"]}

def aggregator(state: State, config: RunnableConfig):
    """すべての検索結果をまとめて回答を生成する。
    config からモデル名やシステムプロンプトを動的に取得します。
    """
    print("  [Aggregator] 集計中...")
    
    # 1. 設定の取得（デフォルト値を指定）
    configurable = config.get("configurable", {})
    model_name = configurable.get("model_name", "gpt-4o-mini")
    system_prompt = configurable.get("system_message", "あなたは役に立つアシスタントです。")
    
    print(f"  [Config] Model: {model_name}, System: {system_prompt[:20]}...")

    # 2. 設定に基づいて LLM を準備
    # ※ 頻繁に呼び出す場合はキャッシュする等の工夫も考えられますが、今回は都度生成します
    llm = ChatOpenAI(model=model_name)

    results_text = "\n".join(state["results"])
    
    # 3. メッセージの構築（システムプロンプトを追加）
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"以下の情報を元に、質問「{state['query']}」に回答してください。\n\n情報:\n{results_text}")
    ]
    
    response = llm.invoke(messages)
    return {"answer": response.content}

# -------------------------------------------------
# 4. グラフ構築
# -------------------------------------------------
builder = StateGraph(State)

builder.add_node("wikipedia", search_wikipedia)
builder.add_node("news", search_news)
builder.add_node("blogs", search_blogs)
builder.add_node("aggregator", aggregator)

builder.add_edge(START, "wikipedia")
builder.add_edge(START, "news")
builder.add_edge(START, "blogs")

builder.add_edge("wikipedia", "aggregator")
builder.add_edge("news", "aggregator")
builder.add_edge("blogs", "aggregator")

builder.add_edge("aggregator", END)

graph = builder.compile()

# -------------------------------------------------
# 5. 実行
# -------------------------------------------------
async def main():
    print("--- Parallel Execution Bot (with Config) 開始 ---")
    
    initial_state = {"query": "LangGraph の特徴", "results": []}
    
    # ケース1: デフォルト設定（gpt-4o-mini, 標準プロンプト）
    print("\n=== Case 1: Default Config ===")
    async for event in graph.astream_events(initial_state, version="v1"):
        if event["event"] == "on_chain_end" and event["name"] == "aggregator":
            print(f"\n[Final Answer]\n{event['data']['output']['answer']}")

    # ケース2: カスタム設定（関西弁プロンプト）
    # ※ モデルは同じですが、プロンプトが変わります
    print("\n=== Case 2: Kansai Dialect Config ===")
    config_kansai = {
        "configurable": {
            "model_name": "gpt-4o-mini",
            "system_message": "あなたは大阪のおばちゃんです。親しみやすい関西弁で答えてください。"
        }
    }
    # state は使い回さず、新しい state で実行します（results が重複しないように）
    async for event in graph.astream_events(initial_state, version="v1", config=config_kansai):
        if event["event"] == "on_chain_end" and event["name"] == "aggregator":
            print(f"\n[Final Answer]\n{event['data']['output']['answer']}")

    # ケース3: モデル切り替え（例: gpt-3.5-turbo）
    # ※ API キーや権限によっては gpt-4 が使えない場合もあるので、利用可能なモデルを指定してください
    print("\n=== Case 3: Model Switch (gpt-3.5-turbo) ===")
    config_gpt35 = {
        "configurable": {
            "model_name": "gpt-3.5-turbo",
            "system_message": "あなたは厳格な学者です。学術的な口調で答えてください。"
        }
    }
    async for event in graph.astream_events(initial_state, version="v1", config=config_gpt35):
        if event["event"] == "on_chain_end" and event["name"] == "aggregator":
            print(f"\n[Final Answer]\n{event['data']['output']['answer']}")

if __name__ == "__main__":
    asyncio.run(main())
    print(graph.get_graph().print_ascii())
