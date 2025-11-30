import os
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# -------------------------------------------------
# 1. 環境設定
# -------------------------------------------------
load_dotenv()

# -------------------------------------------------
# 2. ツール定義
# -------------------------------------------------
@tool
def multiply(a: int, b: int) -> int:
    """2つの整数を掛け算します。"""
    return a * b

@tool
def get_weather(city: str) -> str:
    """指定された都市の天気を取得します（ダミー）。"""
    # 実際には外部APIを叩くところですが、ここではダミーを返します
    if "東京" in city:
        return "晴れ"
    elif "大阪" in city:
        return "雨"
    else:
        return "曇り"

tools = [multiply, get_weather]

# -------------------------------------------------
# 3. LLM とツールのバインディング
# -------------------------------------------------
# LLM に「これらのツールを使っていいよ」と教える
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# -------------------------------------------------
# 4. State 定義
# -------------------------------------------------
class State(TypedDict):
    messages: Annotated[List, add_messages]

# -------------------------------------------------
# 5. ノード定義
# -------------------------------------------------
def chatbot(state: State):
    """LLM を実行するノード。
    ツールが必要ならツール呼び出し（tool_calls）を含むメッセージを返し、
    不要なら普通のテキストを返します。
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# ToolNode は LangGraph が用意している「ツール実行用ノード」です。
# tool_calls を含むメッセージを受け取ると、自動でツールを実行して ToolMessage を返します。
tool_node = ToolNode(tools)

# -------------------------------------------------
# 6. グラフ構築
# -------------------------------------------------
builder = StateGraph(State)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", tool_node)

builder.add_edge(START, "chatbot")

# tools_condition は LangGraph が用意している条件分岐関数です。
# 直前のメッセージに tool_calls が含まれていれば "tools" へ、
# 含まれていなければ END へ遷移します。
builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

builder.add_edge("tools", "chatbot") # ツール実行後は必ず chatbot に戻る（結果を踏まえて回答させるため）

graph = builder.compile()

# -------------------------------------------------
# 7. 実行
# -------------------------------------------------
if __name__ == "__main__":
    print("--- ToolNode Bot 開始 ---")
    
    # ケース1: 普通の会話
    print("\n[User] こんにちは")
    for event in graph.stream({"messages": [HumanMessage(content="こんにちは")]}):
        for key, value in event.items():
            print(f"[{key}] {value['messages'][-1].content}")

    # ケース2: 計算（ツール利用）
    print("\n[User] 123 * 456 は？")
    # 前回の会話を引き継がないように新しい State で開始
    for event in graph.stream({"messages": [HumanMessage(content="123 * 456 は？")]}):
        for key, value in event.items():
            # ToolNode の出力は ToolMessage なので content はツールの実行結果
            msg = value['messages'][-1]
            if isinstance(msg, ToolMessage):
                print(f"[{key}] Tool Output: {msg.content}")
            else:
                print(f"[{key}] {msg.content}")

    # ケース3: 天気（ツール利用）
    print("\n[User] 大阪の天気は？")
    for event in graph.stream({"messages": [HumanMessage(content="大阪の天気は？")]}):
        for key, value in event.items():
            msg = value['messages'][-1]
            if isinstance(msg, ToolMessage):
                print(f"[{key}] Tool Output: {msg.content}")
            else:
                print(f"[{key}] {msg.content}")

    # ケース4: 複雑な計算を実行
    print("\n [User] 1000の階乗 ($1000!$) の末尾に連続してゼロがいくつ並びますか？また、$1000!$ の最上位の桁はいくつですか？(正確な答えが出ないかな)")
    for event in graph.stream({"messages": [HumanMessage(content="1000の階乗 ($1000!$) の末尾に連続してゼロがいくつ並びますか？また、$1000!$ の最上位の桁はいくつですか？")]}):
        for key, value in event.items():
            msg = value['messages'][-1]
            if isinstance(msg, ToolMessage):
                print(f"[{key}] Tool Output: {msg.content}")
            else:
                print(f"[{key}] {msg.content}")

#print(graph.get_graph().print_ascii())