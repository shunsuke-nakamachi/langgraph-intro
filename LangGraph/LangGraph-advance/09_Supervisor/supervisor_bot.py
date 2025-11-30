"""
Supervisor Pattern（Structured Output使用）

この実装では、LangChainの Structured Output 機能を使用して、
LLMのレスポンスを確実に構造化された形式で取得します。

【特徴】
1. Pydanticモデルでレスポンスの型を厳密に定義
2. パース処理が不要（LangChainが自動で処理）
3. 型安全性が向上
4. エラーハンドリングが明確
"""
import asyncio
from typing import TypedDict, Annotated, List, Literal
from enum import Enum
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
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
    next_agent: str  # 次に実行すべきエージェント名

# -------------------------------------------------
# 3. Pydanticモデルでレスポンス構造を定義（Structured Output用）
# -------------------------------------------------

class AgentChoice(str, Enum):
    """選択可能なエージェントの列挙型"""
    QUESTION = "question_agent"
    CALCULATION = "calculation_agent"
    SEARCH = "search_agent"

class RoutingDecision(BaseModel):
    """Supervisorのルーティング決定を表す構造化データ"""
    agent_name: AgentChoice = Field(
        description="選択されたエージェント名。question_agent, calculation_agent, search_agent のいずれか"
    )
    reason: str = Field(
        description="なぜこのエージェントを選んだかの簡潔な理由（1文程度）"
    )

# Structured Output を使うようにLLMを設定
# これにより、LLMは必ず RoutingDecision の形式で返すようになります
structured_llm = llm.with_structured_output(RoutingDecision)

# -------------------------------------------------
# 4. 専門エージェントの定義
# -------------------------------------------------

def question_agent(state: State):
    """質問に答える専門エージェント"""
    print("  [Question Agent] 質問に回答中...")
    
    user_query = state["messages"][-1].content
    
    messages = [
        SystemMessage(content="あなたは質問に答える専門家です。簡潔で分かりやすい回答を心がけてください。"),
        HumanMessage(content=user_query)
    ]
    
    response = llm.invoke(messages)
    return {"messages": [response]}

def calculation_agent(state: State):
    """計算を行う専門エージェント"""
    print("  [Calculation Agent] 計算を実行中...")
    
    user_query = state["messages"][-1].content
    
    messages = [
        SystemMessage(content="あなたは計算の専門家です。数式や計算問題を正確に解いてください。計算過程も示してください。"),
        HumanMessage(content=f"以下の計算を実行してください: {user_query}")
    ]
    
    response = llm.invoke(messages)
    return {"messages": [response]}

def search_agent(state: State):
    """情報検索を行う専門エージェント（シミュレーション）"""
    print("  [Search Agent] 情報を検索中...")
    
    user_query = state["messages"][-1].content
    
    messages = [
        SystemMessage(content="あなたは情報検索の専門家です。最新の情報や事実に基づいて回答してください。"),
        HumanMessage(content=f"以下のトピックについて、最新の情報を調べて回答してください: {user_query}")
    ]
    
    response = llm.invoke(messages)
    return {"messages": [response]}

# -------------------------------------------------
# 5. Supervisor（スーパーバイザー）の定義
# -------------------------------------------------

def supervisor(state: State) -> dict:
    """ユーザーの質問を分析し、適切なエージェントにルーティングする（Structured Output使用）"""
    print("\n[Supervisor] 質問を分析中...")
    
    user_message = state["messages"][-1].content
    
    routing_prompt = f"""
ユーザーの質問を読んで、どの専門エージェントに振り分けるべきか判断してください。

利用可能なエージェント:
1. question_agent: 一般的な質問に答える（例：「Pythonとは？」「AIとは？」「説明して」）
2. calculation_agent: 計算や数式を解く（例：「123 * 456は？」「計算してください」「円の面積」）
3. search_agent: 最新情報や事実を調べる（例：「2024年の最新技術」「最新のトレンド」「調べて」）

ユーザーの質問: {user_message}
"""
    
    messages = [
        SystemMessage(content="あなたは質問を分析して適切な専門家に振り分けるスーパーバイザーです。"),
        HumanMessage(content=routing_prompt)
    ]
    
    # Structured Output を使用して、確実に構造化されたデータを取得
    # 注意: LangChainのコールバックがカスタムPydanticモデルのシリアライゼーションで
    # 警告を出すことがありますが、これは動作には影響しません
    try:
        # コールバックを無効にして警告を抑制（オプション）
        # config = {"callbacks": []}  # コールバックを無効にする場合
        decision: RoutingDecision = structured_llm.invoke(messages)
        
        # デバッグ情報を表示
        print(f"  [Supervisor] 選択されたエージェント: {decision.agent_name.value}")
        print(f"  [Supervisor] 理由: {decision.reason}")
        
        agent_name = decision.agent_name.value
        
    except Exception as e:
        # エラーが発生した場合のフォールバック
        print(f"  [Supervisor] 警告: Structured Outputの取得に失敗しました: {e}")
        print(f"  [Supervisor] デフォルトの question_agent を使用します")
        agent_name = "question_agent"
    
    print(f"  [Supervisor] → {agent_name} にルーティングします")
    
    return {"next_agent": agent_name}

# -------------------------------------------------
# 6. ルーティング関数（条件分岐）
# -------------------------------------------------

def route_to_agent(state: State) -> Literal["question_agent", "calculation_agent", "search_agent", "__end__"]:
    """Supervisorが決定したエージェントにルーティングする"""
    next_agent = state.get("next_agent", "question_agent")
    
    if len(state["messages"]) > 1:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage):
            return "__end__"
    
    return next_agent  # type: ignore

# -------------------------------------------------
# 7. グラフ構築
# -------------------------------------------------
builder = StateGraph(State)

builder.add_node("supervisor", supervisor)
builder.add_node("question_agent", question_agent)
builder.add_node("calculation_agent", calculation_agent)
builder.add_node("search_agent", search_agent)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "question_agent": "question_agent",
        "calculation_agent": "calculation_agent",
        "search_agent": "search_agent",
        "__end__": END
    }
)

builder.add_edge("question_agent", END)
builder.add_edge("calculation_agent", END)
builder.add_edge("search_agent", END)

graph = builder.compile()

# -------------------------------------------------
# 8. 実行
# -------------------------------------------------
async def main():
    print("--- Supervisor Pattern Bot 開始 ---\n")
    
    test_cases = [
        {
            "name": "ケース1: 一般的な質問",
            "query": "Pythonとは何ですか？"
        },
        {
            "name": "ケース2: 計算問題",
            "query": "123 * 456 を計算してください"
        },
        {
            "name": "ケース3: 情報検索",
            "query": "2024年のAI技術の最新トレンドについて教えてください"
        },
        {
            "name": "ケース4: 複雑な計算",
            "query": "半径5cmの円の面積を計算してください（円周率は3.14とします）"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"{test_case['name']}")
        print(f"{'='*60}")
        print(f"[User] {test_case['query']}\n")
        
        initial_state = {
            "messages": [HumanMessage(content=test_case["query"])],
            "next_agent": ""
        }
        
        final_state = None
        async for event in graph.astream_events(initial_state, version="v1"):
            if event["event"] == "on_chain_end":
                name = event.get("name", "")
                if name in ["question_agent", "calculation_agent", "search_agent"]:
                    output = event["data"]["output"]
                    if "messages" in output and output["messages"]:
                        last_msg = output["messages"][-1]
                        if isinstance(last_msg, AIMessage):
                            print(f"\n[Final Answer]\n{last_msg.content}")
        
        print("\n" + "-"*60)

if __name__ == "__main__":
    asyncio.run(main())
