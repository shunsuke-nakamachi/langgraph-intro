"""
LangSmith Integration（トレーシング・デバッグ）

この実装では、LangSmithを使ってLangGraphの実行過程を可視化・デバッグする方法を学びます。

【学ぶこと】
1. LangSmithの基本設定（環境変数、APIキー）
2. トレーシングの有効化
3. LangSmith UIでの実行ログ確認
4. デバッグの活用方法
5. パフォーマンス分析

【前提条件】
1. LangSmithアカウントの作成（無料）
   - https://smith.langchain.com/ にアクセス
   - アカウントを作成してAPIキーを取得
2. 環境変数の設定
   - .envファイルに以下を追加:
     LANGCHAIN_TRACING_V2=true
     LANGCHAIN_API_KEY=your_api_key_here
     LANGCHAIN_PROJECT=langgraph-practice  # プロジェクト名（任意）
"""
import asyncio
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# -------------------------------------------------
# 1. 環境設定とLangSmithの有効化
# -------------------------------------------------
load_dotenv()

# LangSmithの設定確認
tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
api_key = os.getenv("LANGCHAIN_API_KEY")
project_name = os.getenv("LANGCHAIN_PROJECT", "langgraph-practice")

if tracing_enabled and api_key:
    print(f"✅ LangSmith トレーシングが有効です")
    print(f"   プロジェクト名: {project_name}")
    print(f"   LangSmith UI: https://smith.langchain.com/\n")
else:
    print("⚠️  LangSmith トレーシングが無効です")
    print("   .envファイルに以下を設定してください:")
    print("   LANGCHAIN_TRACING_V2=true")
    print("   LANGCHAIN_API_KEY=your_api_key_here")
    print("   LANGCHAIN_PROJECT=langgraph-practice\n")

llm = ChatOpenAI(model="gpt-4o-mini")

# -------------------------------------------------
# 2. State 定義
# -------------------------------------------------
class State(TypedDict):
    messages: Annotated[List, add_messages]
    query: str
    research_result: str
    summary: str

# -------------------------------------------------
# 3. ノード定義
# -------------------------------------------------

def researcher_node(state: State) -> dict:
    """リサーチノード: 質問について情報を調べる"""
    print("\n[Researcher] リサーチ中...")
    
    query = state.get("query", "")
    
    messages = [
        SystemMessage(content="あなたは情報を調べる専門家です。質問に対して正確で有用な情報を提供してください。"),
        HumanMessage(content=f"「{query}」について、重要な情報を3つ挙げてください。")
    ]
    
    response = llm.invoke(messages)
    research_result = response.content
    
    print(f"  ✅ [Researcher] 完了")
    return {"research_result": research_result}

def analyzer_node(state: State) -> dict:
    """分析ノード: リサーチ結果を分析する"""
    print("\n[Analyzer] 分析中...")
    
    query = state.get("query", "")
    research_result = state.get("research_result", "")
    
    messages = [
        SystemMessage(content="あなたは情報を分析する専門家です。リサーチ結果を分析し、要点をまとめてください。"),
        HumanMessage(content=f"""
質問: {query}

リサーチ結果:
{research_result}

上記のリサーチ結果を分析し、質問に対する回答を200文字程度でまとめてください。
""")
    ]
    
    response = llm.invoke(messages)
    summary = response.content
    
    print(f"  ✅ [Analyzer] 完了")
    return {"summary": summary}

def finalizer_node(state: State) -> dict:
    """最終化ノード: 最終的な回答を整形する"""
    print("\n[Finalizer] 最終化中...")
    
    query = state.get("query", "")
    summary = state.get("summary", "")
    
    messages = [
        SystemMessage(content="あなたは回答を整形する専門家です。分かりやすく読みやすい形式で回答を整形してください。"),
        HumanMessage(content=f"""
質問: {query}

分析結果:
{summary}

上記の分析結果を元に、ユーザーへの最終回答を整形してください。
""")
    ]
    
    response = llm.invoke(messages)
    final_answer = response.content
    
    print(f"  ✅ [Finalizer] 完了")
    return {"messages": [AIMessage(content=final_answer)]}

# -------------------------------------------------
# 4. グラフ構築
# -------------------------------------------------
builder = StateGraph(State)

builder.add_node("researcher", researcher_node)
builder.add_node("analyzer", analyzer_node)
builder.add_node("finalizer", finalizer_node)

builder.add_edge(START, "researcher")
builder.add_edge("researcher", "analyzer")
builder.add_edge("analyzer", "finalizer")
builder.add_edge("finalizer", END)

graph = builder.compile()

# -------------------------------------------------
# 5. 実行
# -------------------------------------------------
async def main():
    print("--- LangSmith Integration Bot 開始 ---\n")
    
    if tracing_enabled:
        print("📊 LangSmith UIで実行ログを確認できます:")
        print(f"   https://smith.langchain.com/o/{os.getenv('LANGCHAIN_ORG_ID', 'default')}/projects/p/{project_name}\n")
    
    test_queries = [
        "LangGraphとは何ですか？",
        "マルチエージェントシステムの利点は？",
        "エラーハンドリングのベストプラクティスは？"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"ケース{i}: {query}")
        print(f"{'='*60}\n")
        
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "research_result": "",
            "summary": ""
        }
        
        # グラフを実行
        # LangSmithが自動的に実行ログを記録します
        final_state = None
        async for event in graph.astream_events(initial_state, version="v1"):
            if event["event"] == "on_chain_end":
                name = event.get("name", "")
                if name == "finalizer":
                    output = event["data"]["output"]
                    if "messages" in output and output["messages"]:
                        last_msg = output["messages"][-1]
                        if isinstance(last_msg, AIMessage):
                            print(f"\n[Final Answer]\n{last_msg.content}")
        
        print("\n" + "-"*60)
        
        # テストケース間で少し待機（APIレート制限対策）
        if i < len(test_queries):
            await asyncio.sleep(1)
    
    if tracing_enabled:
        print(f"\n{'='*60}")
        print("📊 LangSmith UIで詳細を確認:")
        print(f"   https://smith.langchain.com/o/{os.getenv('LANGCHAIN_ORG_ID', 'default')}/projects/p/{project_name}")
        print(f"{'='*60}\n")
        print("LangSmith UIでは以下が確認できます:")
        print("  - 各ノードの実行時間")
        print("  - LLMへの入力と出力")
        print("  - トークン使用量")
        print("  - エラーの詳細")
        print("  - Stateの変化")

if __name__ == "__main__":
    asyncio.run(main())

