# 05. Reflection（自己省察）

このコードは、**「AI が自分の書いた文章を自分で批評し、修正する」** というループ（Reflection）を実現するサンプルです。
一発書きの回答よりも、推敲（すいこう）を重ねた回答の方が品質が高くなるという性質を利用しています。

## 構成

1.  **Generator（作成者）**: 文章を書く、または修正する役割。
2.  **Reflector（批評家）**: 文章を読み、改善点を指摘する役割。
3.  **Router（監督）**: 「もう十分良くなったか？（または回数制限に達したか？）」を判断し、ループを継続するか終了するかを決めます。

## 仕組みのポイント

### 1. 循環ループの構築

```python
# Generator -> Router -> Reflector
builder.add_conditional_edges("generator", router, {"continue": "reflector", "end": END})
# Reflector -> Generator
builder.add_edge("reflector", "generator")
```

*   `generator` と `reflector` をぐるぐると回るループを作っています。
*   これにより、「書く」→「指摘される」→「書き直す」→「また指摘される」... というプロセスが自動化されます。

### 2. 文脈の共有

*   `State["messages"]` には、これまでの全てのやり取り（初稿、批評、修正案...）が蓄積されています。
*   Generator は、直前の Reflector の批評（`messages[-1]`）を読むことで、どこを直せばいいかを理解します。

## コードの詳細解説

このコードは、**「書いて、直して、また書く」** という人間の推敲プロセスを模倣しています。

上から順に、何をしているのか詳しく解説します。

### 1. Stateの定義 (13-16行目)

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]
    loop_count: int # ループ回数を管理するためのカウンタ
```

*   `loop_count`: 無限ループを防ぐために、今何回目の修正か数えておく変数です。

### 2. Generator（作成者）ノード (20-37行目)

```python
def generator(state: State):
    # ...
    if loop_count == 0:
        # 初回: ユーザーの指示で書く
        response = llm.invoke(messages)
    else:
        # 2回目以降: 批評を踏まえて修正する
        instructions = messages + [HumanMessage(content="上記の批評を踏まえて、文章をより良く修正してください。")]
        response = llm.invoke(instructions)
        
    return {"messages": [response], "loop_count": loop_count + 1}
```

*   **初回**: まだ批評がないので、普通に文章を書きます。
*   **2回目以降**: 直前のメッセージ（Reflectorからの批評）を読んだ上で、「修正して」という指示を追加して LLM に投げます。これが「自己修正」の肝です。

### 3. Reflector（批評家）ノード (39-57行目)

```python
def reflector(state: State):
    # ...
    prompt = f"""
    以下の文章を読んで、改善点を具体的に指摘してください。
    特に「論理性」「具体性」「表現の豊かさ」の観点からアドバイスしてください。
    対象の文章: {target_text}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}
```

*   Generator が書いた文章（`target_text`）を読み、**具体的な改善点**を指摘します。
*   ここで厳しい批評をさせるほど、次の修正で品質が上がります。

### 4. Router（監督）による制御 (60-68行目)

```python
def router(state: State):
    loop_count = state["loop_count"]
    # 2回修正したら終了
    if loop_count > 2:
        return "end"
    else:
        return "continue"
```

*   今回はシンプルに「回数制限」で終了させていますが、本来は「Reflector が『完璧です』と言ったら終了」のような条件にすると、より高度になります。

## まとめ

この Reflection パターンは、**「Chain of Thought（思考の連鎖）」** をさらに発展させたものです。
一度の出力で完璧を目指すのではなく、**「とりあえず書いてみて、後から直す」** というアプローチを取ることで、複雑なタスクでも高品質な成果を出せるようになります。
