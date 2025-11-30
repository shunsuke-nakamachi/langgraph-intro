from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# 環境変数の読み込み
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# ==========================================
# 1. 子グラフ（リサーチチーム）の定義
# ==========================================

# 子グラフ用のState
class ResearchState(TypedDict):
    # 親から受け取るメッセージと、内部での会話履歴
    messages: Annotated[list, add_messages]
    # 成果物（調査結果）
    research_summary: str

def researcher(state: ResearchState):
    """調査を行うノード"""
    print("  [子] リサーチャー: 調査中...")
    query = state["messages"][-1].content
    # 実際は検索ツールなどを使うが、今回はLLMでシミュレーション
    response = llm.invoke([
        HumanMessage(content=f"「{query}」について、重要な事実を3つ箇条書きで挙げてください。")
    ])
    return {"messages": [response], "research_summary": response.content}

def reviewer(state: ResearchState):
    """調査結果をレビューするノード"""
    print("  [子] レビュアー: 確認中...")
    # シンプルにパスするだけ（本来は品質チェックなどを行う）
    return {"messages": [AIMessage(content="調査結果を確認しました。問題ありません。")]}

# 子グラフの構築
research_builder = StateGraph(ResearchState)
research_builder.add_node("researcher", researcher)
research_builder.add_node("reviewer", reviewer)

research_builder.add_edge(START, "researcher")
research_builder.add_edge("researcher", "reviewer")
research_builder.add_edge("reviewer", END)

# コンパイル（これが「部品」になる）
research_graph = research_builder.compile()


# ==========================================
# 2. 親グラフ（編集部）の定義
# ==========================================

# 親グラフ用のState
class MainState(TypedDict):
    messages: Annotated[list, add_messages]
    final_article: str

def topic_receiver(state: MainState):
    """ユーザーからトピックを受け取る（最初のノード）"""
    topic = state["messages"][-1].content
    print(f"[親] 編集長: 「{topic}」についての記事を書きたい。リサーチチーム、頼む！")
    # ここでは特に何もしない（次のノードへメッセージを渡すだけ）
    return {}

def writer(state: MainState):
    """リサーチ結果を元に記事を書くノード"""
    print("[親] ライター: リサーチ結果が届いたな。記事を書こう。")
    
    # 直前のメッセージ（子グラフの最後の出力）を取得
    research_result = state["messages"][-1].content
    
    response = llm.invoke([
        HumanMessage(content=f"以下の調査結果を元に、短いブログ記事を書いてください。\n\n{research_result}")
    ])
    print(f"\n[親] ライター: 完成！\n\n{response.content}")
    return {"messages": [response], "final_article": response.content}

# 親グラフの構築
main_builder = StateGraph(MainState)
main_builder.add_node("topic_receiver", topic_receiver)

# ★ここで子グラフを「1つのノード」として追加！
main_builder.add_node("research_team", research_graph)

main_builder.add_node("writer", writer)

main_builder.add_edge(START, "topic_receiver")
main_builder.add_edge("topic_receiver", "research_team") # 親 -> 子
main_builder.add_edge("research_team", "writer")         # 子 -> 親
main_builder.add_edge("writer", END)

# 親グラフのコンパイル
main_graph = main_builder.compile()

# ==========================================
# 3. 実行
# ==========================================

if __name__ == "__main__":
    print("--- アプリケーション開始 ---")
    
    # ユーザーの入力
    user_input = "コーヒーの健康効果"
    
    # 親グラフを実行
    # 子グラフの実行は内部で自動的に行われる
    main_graph.invoke({"messages": [HumanMessage(content=user_input)]})
    
    print("\n--- アプリケーション終了 ---")
    #print(main_graph.get_graph().print_ascii())
    #print(research_graph.get_graph().print_ascii())