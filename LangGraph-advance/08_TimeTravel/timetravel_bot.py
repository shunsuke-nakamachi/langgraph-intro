import asyncio
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
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
    messages: Annotated[List[BaseMessage], add_messages]

# -------------------------------------------------
# 3. ノード定義
# -------------------------------------------------
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# -------------------------------------------------
# 4. グラフ構築
# -------------------------------------------------
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Time Travel には Checkpointer が必須です
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# -------------------------------------------------
# 5. 実行 (Time Travel Demo)
# -------------------------------------------------
async def main():
    print("--- Time Travel Demo 開始 ---")
    
    # thread_id を固定して会話を始めます
    config = {"configurable": {"thread_id": "demo_thread"}}
    
    # Step 1: 最初の会話
    print("\n[Step 1] User: こんにちは")
    async for event in graph.astream_events({"messages": [HumanMessage(content="こんにちは")]}, config, version="v1"):
        if event["event"] == "on_chain_end" and event["name"] == "chatbot":
            print(f"Bot: {event['data']['output']['messages'][-1].content}")

    # Step 2: 2回目の会話
    print("\n[Step 2] User: 私はうどんが好きです")
    async for event in graph.astream_events({"messages": [HumanMessage(content="私はうどんが好きです")]}, config, version="v1"):
        if event["event"] == "on_chain_end" and event["name"] == "chatbot":
            print(f"Bot: {event['data']['output']['messages'][-1].content}")

    # Step 3: 3回目の会話
    print("\n[Step 3] User: 私の好きな食べ物は？")
    async for event in graph.astream_events({"messages": [HumanMessage(content="私の好きな食べ物は？")]}, config, version="v1"):
        if event["event"] == "on_chain_end" and event["name"] == "chatbot":
            print(f"Bot: {event['data']['output']['messages'][-1].content}")

    # -------------------------------------------------
    # ここから Time Travel
    # -------------------------------------------------
    print("\n--- Time Travel 実行 ---")
    print("履歴を確認して、Step 2 (うどんが好きと言った直後) の状態に戻ります。")
    
    # 履歴を取得 (新しい順)
    # history[0] = Step 3 完了後
    # history[1] = Step 3 開始前 (Step 2 完了後) ... となるはずですが、
    # 正確には checkpoint の保存タイミングによります。
    all_states = [s async for s in graph.aget_state_history(config)]
    
    # 状態の中身を見て、戻りたい場所を探します
    # 今回は「うどんが好き」と言う前（＝「こんにちは」のやり取りが終わった直後）の状態を探します
    target_config = None
    for state in all_states:
        msgs = state.values["messages"]
        # メッセージ履歴が [User:こんにちは, Bot:こんにちは...] だけの状態を探す
        # （"うどん" がまだ含まれていない状態）
        if len(msgs) > 0:
            # 全メッセージのコンテンツを結合してチェック
            all_content = "".join([m.content for m in msgs])
            if "こんにちは" in all_content and "うどん" not in all_content:
                target_config = state.config
                print(f"DEBUG: Found target config: {target_config}")
                break
    
    if target_config:
        print("\n[Step 4 (Time Travel)] 過去（うどんと言う前）に戻って別の発言をします: 'やっぱりそばが好きです'")
        # 過去の config を使って新しい入力を投げると、そこから分岐（Fork）します
        # これにより、履歴は [こんにちは, こんにちはBot, そば...] という新しいブランチになります
        async for event in graph.astream_events(
            {"messages": [HumanMessage(content="やっぱりそばが好きです")]}, 
            target_config,
            version="v1"
        ):
            if event["event"] == "on_chain_end" and event["name"] == "chatbot":
                print(f"Bot: {event['data']['output']['messages'][-1].content}")
        
        # 確認: 最新の状態はどうなっているか？
        # config (thread_idのみ) を指定すると、最新のブランチ（今更新した方）が使われます
        
        print("\n[Step 5] User: 私の好きな食べ物は？ (分岐後の世界で確認)")
        async for event in graph.astream_events({"messages": [HumanMessage(content="私の好きな食べ物は？")]}, config, version="v1"):
            if event["event"] == "on_chain_end" and event["name"] == "chatbot":
                print(f"Bot: {event['data']['output']['messages'][-1].content}")

    else:
        print("ターゲットの状態が見つかりませんでした。")

if __name__ == "__main__":
    asyncio.run(main())
