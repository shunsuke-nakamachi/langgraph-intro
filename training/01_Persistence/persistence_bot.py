from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

# 環境変数の読み込み
load_dotenv()

# 1. Stateの定義
class State(TypedDict):
    messages: Annotated[list, add_messages]

# 2. ノードの定義
llm = ChatOpenAI(model="gpt-4o-mini")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

# 3. グラフの構築
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# 4. チェックポインター（メモリ）の準備
# これが「記憶」を司るデータベースの役割を果たします（今回はオンメモリ）
memory = MemorySaver()

# 5. コンパイル（checkpointerを指定するのが重要！）
graph = builder.compile(checkpointer=memory)

# 6. 実行設定（スレッドIDを決める）
# thread_id が同じなら、記憶が引き継がれます
config = {"configurable": {"thread_id": "user-1"}}

print("--- 1回目の会話 (user-1) ---")
input_message = {"messages": [("user", "私の名前は田中です。")]}
for event in graph.stream(input_message, config):
    for value in event.values():
        print("Assistant:", value["messages"][-1].content)

print("\n--- 2回目の会話 (user-1) ---")
# 前回の会話（名前は田中）を覚えているか確認
input_message = {"messages": [("user", "私の名前を覚えていますか？")]}
for event in graph.stream(input_message, config):
    for value in event.values():
        print("Assistant:", value["messages"][-1].content)

print("\n--- 別のユーザーの会話 (user-2) ---")
# thread_id を変えると、記憶は共有されない
config_2 = {"configurable": {"thread_id": "user-2"}}
input_message = {"messages": [("user", "私の名前を覚えていますか？")]}
for event in graph.stream(input_message, config_2):
    for value in event.values():
        print("Assistant:", value["messages"][-1].content)
