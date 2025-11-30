"""
Multi-Agent System（マルチエージェントシステム）

この実装では、複数のエージェントが協調して動作するマルチエージェントシステムを学びます。

【学ぶこと】
1. 複数の専門エージェントの定義
2. エージェント間の協調と情報共有
3. 並列実行と順次実行の組み合わせ
4. エージェント間の通信と結果の統合
5. コーディネーターによる全体制御
"""
import asyncio
import operator
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# -------------------------------------------------
# 1. 環境設定
# -------------------------------------------------
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# -------------------------------------------------
# 2. State 定義
# -------------------------------------------------
class State(TypedDict):
    messages: Annotated[List, add_messages]
    topic: str  # 記事のトピック
    research_results: Annotated[List[str], operator.add]  # リサーチ結果（複数エージェントから集約）
    draft: Optional[str]  # ライターが作成した下書き
    review_feedback: Optional[str]  # レビュアーからのフィードバック
    final_article: Optional[str]  # 最終的な記事

# -------------------------------------------------
# 3. 専門エージェントの定義
# -------------------------------------------------

def researcher_agent_1(state: State) -> dict:
    """リサーチエージェント1: 技術的な情報を収集"""
    print("\n[Researcher 1] 技術情報をリサーチ中...")
    
    topic = state.get("topic", "")
    
    messages = [
        SystemMessage(content="あなたは技術情報を専門にリサーチするエージェントです。技術的な詳細、仕様、実装方法などを調べてください。"),
        HumanMessage(content=f"「{topic}」について、技術的な観点から重要な情報を3つ挙げてください。")
    ]
    
    response = llm.invoke(messages)
    research_result = f"[技術情報] {response.content}"
    
    print(f"  ✅ [Researcher 1] 完了")
    return {"research_results": [research_result]}

def researcher_agent_2(state: State) -> dict:
    """リサーチエージェント2: 市場動向やトレンドを収集"""
    print("\n[Researcher 2] 市場動向をリサーチ中...")
    
    topic = state.get("topic", "")
    
    messages = [
        SystemMessage(content="あなたは市場動向やトレンドを専門にリサーチするエージェントです。業界の動向、市場規模、将来予測などを調べてください。"),
        HumanMessage(content=f"「{topic}」について、市場動向やトレンドの観点から重要な情報を3つ挙げてください。")
    ]
    
    response = llm.invoke(messages)
    research_result = f"[市場動向] {response.content}"
    
    print(f"  ✅ [Researcher 2] 完了")
    return {"research_results": [research_result]}

def researcher_agent_3(state: State) -> dict:
    """リサーチエージェント3: ユーザー視点や事例を収集"""
    print("\n[Researcher 3] ユーザー視点をリサーチ中...")
    
    topic = state.get("topic", "")
    
    messages = [
        SystemMessage(content="あなたはユーザー視点や実用例を専門にリサーチするエージェントです。実際の使用例、ユーザーの声、成功事例などを調べてください。"),
        HumanMessage(content=f"「{topic}」について、ユーザー視点や実用例の観点から重要な情報を3つ挙げてください。")
    ]
    
    response = llm.invoke(messages)
    research_result = f"[ユーザー視点] {response.content}"
    
    print(f"  ✅ [Researcher 3] 完了")
    return {"research_results": [research_result]}

def writer_agent(state: State) -> dict:
    """ライターエージェント: リサーチ結果を元に記事を執筆"""
    print("\n[Writer] 記事を執筆中...")
    
    topic = state.get("topic", "")
    research_results = state.get("research_results", [])
    
    # リサーチ結果をまとめる
    research_summary = "\n\n".join(research_results)
    
    messages = [
        SystemMessage(content="あなたは技術記事を書く専門ライターです。リサーチ結果を元に、分かりやすく読みやすい記事を書いてください。"),
        HumanMessage(content=f"""
以下のリサーチ結果を元に、「{topic}」についての技術記事を執筆してください。

リサーチ結果:
{research_summary}

記事の要件:
- 800文字程度
- 技術的な正確性を保つ
- 読みやすく分かりやすい構成
- 具体例を含める
""")
    ]
    
    response = llm.invoke(messages)
    draft = response.content
    
    print(f"  ✅ [Writer] 下書き完成（{len(draft)}文字）")
    return {"draft": draft}

def reviewer_agent(state: State) -> dict:
    """レビュアーエージェント: 記事をレビューしてフィードバックを提供"""
    print("\n[Reviewer] 記事をレビュー中...")
    
    draft = state.get("draft", "")
    topic = state.get("topic", "")
    
    if not draft:
        return {"review_feedback": "下書きが存在しません。"}
    
    messages = [
        SystemMessage(content="あなたは技術記事のレビュアーです。記事の品質、正確性、読みやすさを評価し、改善点を指摘してください。"),
        HumanMessage(content=f"""
以下の記事をレビューしてください。

トピック: {topic}

記事:
{draft}

以下の観点から評価してください:
1. 技術的な正確性
2. 読みやすさと構成
3. 具体例の適切性
4. 改善すべき点

改善点があれば具体的に指摘してください。
""")
    ]
    
    response = llm.invoke(messages)
    feedback = response.content
    
    print(f"  ✅ [Reviewer] レビュー完了")
    return {"review_feedback": feedback}

def editor_agent(state: State) -> dict:
    """エディターエージェント: レビューフィードバックを元に記事を最終化"""
    print("\n[Editor] 記事を最終化中...")
    
    draft = state.get("draft", "")
    review_feedback = state.get("review_feedback", "")
    topic = state.get("topic", "")
    
    if not draft:
        return {"final_article": "下書きが存在しません。"}
    
    messages = [
        SystemMessage(content="あなたは技術記事のエディターです。レビューフィードバックを踏まえて、記事を最終化してください。"),
        HumanMessage(content=f"""
以下の下書きとレビューフィードバックを元に、記事を最終化してください。

トピック: {topic}

下書き:
{draft}

レビューフィードバック:
{review_feedback}

レビューフィードバックを踏まえて、記事を改善し、最終版を作成してください。
""")
    ]
    
    response = llm.invoke(messages)
    final_article = response.content
    
    print(f"  ✅ [Editor] 最終版完成（{len(final_article)}文字）")
    return {"final_article": final_article}

# -------------------------------------------------
# 4. グラフ構築
# -------------------------------------------------
builder = StateGraph(State)

# ノードの追加
builder.add_node("researcher_1", researcher_agent_1)
builder.add_node("researcher_2", researcher_agent_2)
builder.add_node("researcher_3", researcher_agent_3)
builder.add_node("writer", writer_agent)
builder.add_node("reviewer", reviewer_agent)
builder.add_node("editor", editor_agent)

# エッジの追加
# 3つのリサーチャーを並列実行（Fan-out）
builder.add_edge(START, "researcher_1")
builder.add_edge(START, "researcher_2")
builder.add_edge(START, "researcher_3")

# すべてのリサーチが完了したらライターへ（Fan-in）
builder.add_edge("researcher_1", "writer")
builder.add_edge("researcher_2", "writer")
builder.add_edge("researcher_3", "writer")

# ライター → レビュアー → エディター（順次実行）
builder.add_edge("writer", "reviewer")
builder.add_edge("reviewer", "editor")
builder.add_edge("editor", END)

graph = builder.compile()

# -------------------------------------------------
# 5. 実行
# -------------------------------------------------
async def main():
    print("--- Multi-Agent System Bot 開始 ---\n")
    
    test_topics = [
        "LangGraphの特徴と使い方",
        "マルチエージェントシステムの設計",
        "AIエージェントの実装パターン"
    ]
    
    for i, topic in enumerate(test_topics, 1):
        print(f"\n{'='*60}")
        print(f"ケース{i}: {topic}")
        print(f"{'='*60}\n")
        
        initial_state = {
            "messages": [HumanMessage(content=f"「{topic}」についての記事を作成してください")],
            "topic": topic,
            "research_results": [],
            "draft": None,
            "review_feedback": None,
            "final_article": None
        }
        
        # グラフを実行
        final_state = None
        async for event in graph.astream_events(initial_state, version="v1"):
            if event["event"] == "on_chain_end":
                name = event.get("name", "")
                if name == "editor":
                    output = event["data"]["output"]
                    if "final_article" in output:
                        print(f"\n{'='*60}")
                        print("【最終記事】")
                        print(f"{'='*60}")
                        print(output["final_article"])
        
        print(f"\n{'='*60}")
        print("【リサーチ結果の要約】")
        print(f"{'='*60}")
        for j, result in enumerate(initial_state.get("research_results", []), 1):
            print(f"\n{j}. {result[:100]}...")  # 最初の100文字だけ表示
        
        print("\n" + "-"*60)
        
        # テストケース間で少し待機（APIレート制限対策）
        if i < len(test_topics):
            await asyncio.sleep(2)
    
    # グラフ構造を表示
    print(f"\n{'='*60}")
    print("【グラフ構造】")
    print(f"{'='*60}")
    graph_ascii = graph.get_graph().print_ascii()
    print(graph_ascii)

if __name__ == "__main__":
    asyncio.run(main())

