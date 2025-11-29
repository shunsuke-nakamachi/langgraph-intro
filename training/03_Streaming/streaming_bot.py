import asyncio
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

# 環境変数の読み込み
load_dotenv()

# 1. Stateの定義
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. ノードの定義
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True) # streaming=True は明示しなくても動くことが多いですが、念のため

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# 3. グラフの構築
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

# 4. 非同期実行関数
async def main():
    print("--- ストリーミング開始 ---")
    
    # astream_events: グラフの中で起きる「イベント」をリアルタイムで受信します
    # version="v2" は必須です（LangChainの仕様）
    async for event in graph.astream_events(
        {"messages": [HumanMessage(content="桃太郎の物語を、SF風にアレンジして300文字以内で語ってください。")]},
        version="v1"
    ):
        # イベントの種類を確認
        kind = event["event"]
        
        # 'on_chat_model_stream' というイベントが、LLMからトークンが届いた瞬間です
        if kind == "on_chat_model_stream":
            # イベントデータからトークン（文字）を取り出す
            content = event["data"]["chunk"].content
            if content:
                # 改行なしで表示（タイピング風エフェクト）
                print(content, end="", flush=True)
                
    print("\n\n--- 終了 ---")

if __name__ == "__main__":
    asyncio.run(main())
