from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# 環境変数の読み込み
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# 1. Stateの定義
class State(TypedDict):
    messages: Annotated[list, add_messages]
    # ループ回数を管理するためのカウンタ
    loop_count: int

# 2. ノードの定義

def generator(state: State):
    """文章を生成（または修正）するノード"""
    messages = state["messages"]
    loop_count = state.get("loop_count", 0)
    
    if loop_count == 0:
        print("\n[Generator] 初稿を作成中...")
        # 初回はユーザーの指示に従って書く
        response = llm.invoke(messages)
    else:
        print(f"\n[Generator] 修正中... (回数: {loop_count})")
        # 2回目以降は、直前の「批評」を踏まえて修正する
        # 直前のメッセージは Reflector からの批評になっているはず
        instructions = messages + [HumanMessage(content="上記の批評を踏まえて、文章をより良く修正してください。")]
        response = llm.invoke(instructions)
        
    return {"messages": [response], "loop_count": loop_count + 1}

def reflector(state: State):
    """文章を批評するノード"""
    print("\n[Reflector] 批評中...")
    
    # 直前のメッセージ（Generatorの出力）を取得
    target_text = state["messages"][-1].content
    
    # 批評を行うプロンプト
    prompt = f"""
    以下の文章を読んで、改善点を具体的に指摘してください。
    特に「論理性」「具体性」「表現の豊かさ」の観点からアドバイスしてください。
    
    対象の文章:
    {target_text}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # 批評を返す（これが次のGeneratorへの入力になる）
    return {"messages": [response]}

# 3. 条件分岐（ルーター）
def router(state: State):
    """ループを続けるか終わるかを判断する"""
    loop_count = state["loop_count"]
    
    # 2回修正したら終了（初稿 + 修正1 + 修正2）
    if loop_count > 2:
        return "end"
    else:
        return "continue"

# 4. グラフの構築
builder = StateGraph(State)
builder.add_node("generator", generator)
builder.add_node("reflector", reflector)

builder.add_edge(START, "generator")

# Generator -> (条件分岐) -> Reflector or END
builder.add_conditional_edges(
    "generator",
    router,
    {
        "continue": "reflector",
        "end": END
    }
)

# Reflector -> Generator (ループ)
builder.add_edge("reflector", "generator")

graph = builder.compile()

# 5. 実行
if __name__ == "__main__":
    print("--- アプリケーション開始 ---")
    
    # 初期状態（loop_countを0で初期化）
    initial_state = {
        "messages": [HumanMessage(content="「AIと人間の共存」について、300文字程度の短いエッセイを書いてください。")],
        "loop_count": 0
    }
    
    # 実行
    final_state = None
    for event in graph.stream(initial_state):
        for key, value in event.items():
            if "messages" in value:
                last_msg = value["messages"][-1]
                # 長すぎるので先頭だけ表示
                print(f"Output from {key}: {last_msg.content[:50]}...")
            
            # 最終状態を更新し続ける
            final_state = value
    
    print("\n--- 最終成果物 ---")
    if final_state and "messages" in final_state:
        print(final_state["messages"][-1].content)
