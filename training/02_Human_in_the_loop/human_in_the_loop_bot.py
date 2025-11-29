from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage

# 環境変数の読み込み
load_dotenv()

# 1. Stateの定義
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. ノードの定義
llm = ChatOpenAI(model="gpt-4o-mini")

def chatbot(state: State):
    """ユーザーの要望に基づいてツイート案を作成する"""
    print("\n--- AI: ツイート案を作成中... ---")
    # AIへの指示を追加
    messages = state["messages"] + [HumanMessage(content="この内容でSNSの投稿文案を1つ作成してください。ハッシュタグもつけてください。")]
    response = llm.invoke(messages)
    return {"messages": [response]}

def publisher(state: State):
    """承認されたツイートを公開（表示）する"""
    last_message = state["messages"][-1]
    print(f"\n🚀 公開しました:\n{last_message.content}")
    return {"messages": [AIMessage(content="投稿を公開しました。")]}

# 3. グラフの構築
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_node("publisher", publisher)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", "publisher")
builder.add_edge("publisher", END)

# 4. チェックポインター（メモリ）の準備
memory = MemorySaver()

# 5. コンパイル（★ここで interrupt_before を指定！）
# "publisher" ノードを実行する「直前」で一時停止します
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["publisher"]
)

# 6. 実行設定
thread_id = "user-123"
config = {"configurable": {"thread_id": thread_id}}

# --- 実行パート ---

print("--- ステップ1: ツイート案の作成 ---")
# ユーザーからの入力
input_text = input("投稿したい話題を入力してください: ")
if not input_text:
    input_text = "LangGraphの勉強中"

# グラフを実行（publisherの手前で止まるはず）
events = graph.stream(
    {"messages": [HumanMessage(content=input_text)]},
    config,
    stream_mode="values"
)

for event in events:
    if "messages" in event:
        print(f"Current State: {event['messages'][-1].content[:20]}...")

# 状態確認
snapshot = graph.get_state(config)
print("\n--- ⏸️ 一時停止中 ---")
print("次のステップ:", snapshot.next)

if snapshot.next:
    # ユーザーに承認を求める
    user_approval = input("\nこの投稿案で公開してもよろしいですか？ (y/n): ")

    if user_approval.lower() == "y":
        print("\n--- ステップ2: 公開（再開） ---")
        # None を渡して再開すると、止まっていたところから動き出します
        for event in graph.stream(None, config, stream_mode="values"):
            pass
    else:
        print("\n--- 公開をキャンセルしました ---")
