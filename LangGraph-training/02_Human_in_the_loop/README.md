# 02. Human-in-the-loop（人間による承認・介入）

このコードは、**「AI が生成した内容を人間が確認してから次に進む（公開する）」** というフローを実現するサンプルです。
「誤情報の拡散防止」や「機密情報のチェック」など、実用的なアプリケーションでは必須の機能です。

上から順に、何をしているのか詳しく解説します。

## 1. 準備フェーズ (1-14行目)

```python
# 1. Stateの定義
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

*   **State**: 前回と同様、会話履歴を保持するリストです。

## 2. ノードの定義 (16-30行目)

```python
def chatbot(state: State):
    """ユーザーの要望に基づいてツイート案を作成する"""
    # ... (AIに指示を出して案を作らせる処理)
    return {"messages": [response]}

def publisher(state: State):
    """承認されたツイートを公開（表示）する"""
    last_message = state["messages"][-1]
    print(f"\n🚀 公開しました:\n{last_message.content}")
    return {"messages": [AIMessage(content="投稿を公開しました。")]}
```

*   **chatbot**: ツイートの「下書き」を作る役割です。
*   **publisher**: 実際に世に出す（今回は画面に表示する）役割です。
*   **ポイント**: この2つのノードの間に「人間のチェック」を挟みたいわけです。

## 3. グラフの構築と「一時停止」の設定 (33-49行目) ★ここが一番重要

```python
# 3. グラフの構築
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_node("publisher", publisher)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", "publisher") # chatbot -> publisher へつなぐ
builder.add_edge("publisher", END)

# 4. チェックポインター（メモリ）の準備
memory = MemorySaver()

# 5. コンパイル（★ここで interrupt_before を指定！）
graph = builder.compile(
    checkpointer=memory,
    interrupt_before=["publisher"]
)
```

*   **interrupt_before=["publisher"]**: これが魔法の呪文です。「`publisher` ノードを実行する**直前**で、必ず一時停止しなさい」という命令です。
*   これにより、`chatbot` が終わって `publisher` に行こうとした瞬間に、プログラムは自動的に停止し、制御をユーザー（Pythonプログラム）に戻します。

## 4. 実行と承認フロー (52-89行目)

### ① ステップ1: ツイート案の作成（一時停止まで）

```python
# グラフを実行（publisherの手前で止まるはず）
events = graph.stream(
    {"messages": [HumanMessage(content=input_text)]},
    config,
    stream_mode="values"
)
```

*   ここでグラフを実行すると、`chatbot` が動いてツイート案を作ります。
*   そして `publisher` の手前で止まります。

### ② 状態確認と承認

```python
# 状態確認
snapshot = graph.get_state(config)
print("次のステップ:", snapshot.next) # ('publisher',) と表示されるはず
```

*   **graph.get_state(config)**: 現在のグラフの状態（どこで止まっているか、どんなデータを持っているか）を確認できます。
*   ここでユーザーに `input()` で「承認しますか？」と聞きます。

### ③ ステップ2: 公開（再開）

```python
if user_approval.lower() == "y":
    # None を渡して再開すると、止まっていたところから動き出します
    for event in graph.stream(None, config, stream_mode="values"):
        pass
```

*   **graph.stream(None, config)**: 入力を `None` にして再度実行すると、LangGraph は「あ、さっきの一時停止を解除して続きをやるんだな」と理解します。
*   止まっていた `publisher` ノードが実行され、「🚀 公開しました」と表示されます。

## まとめ

この機能を使えば、**「AI が勝手に暴走するのを防ぐ安全弁」** をシステムに組み込むことができます。
Web アプリにする場合は、「承認ボタン」を押したときに `graph.stream(None, ...)` を呼ぶAPIを作ればOKです。