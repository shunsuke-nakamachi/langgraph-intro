# 01. Persistence（永続化）

このコードは、**「AI が会話の内容を記憶し、ユーザーごとに区別して会話する」** という機能（永続化）を実現するための最小限のサンプルです。

上から順に、何をしているのか詳しく解説します。

## 1. 準備フェーズ (1-16行目)

```python
# 1. Stateの定義
class State(TypedDict):
    messages: Annotated[list, add_messages]
```

*   **State（状態）**: 会話の履歴を入れておく「箱」の形を決めています。
*   `Annotated[list, add_messages]`: ここが重要です。「新しいメッセージが来たら、上書きせずにリストの後ろに追加（append）してね」というルールを設定しています。これがないと、AI は直前の言葉しか覚えられません。

```python
# 2. ノードの定義
llm = ChatOpenAI(model="gpt-4o-mini")

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}
```

*   **chatbot ノード**: AI の頭脳です。これまでの会話履歴（`state["messages"]`）を全部読んで、次の返事を考えます。

## 2. グラフの組み立て (21-25行目)

```python
builder = StateGraph(State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)
```

*   非常にシンプルな「一方通行」のグラフを作っています。
*   **START** → **chatbot**（AIが返事する） → **END**（終了）
*   今回はループや条件分岐はありません。

## 3. 記憶の仕組みを導入 (27-32行目) ★ここが一番重要

```python
# 4. チェックポインター（メモリ）の準備
memory = MemorySaver()

# 5. コンパイル（checkpointerを指定するのが重要！）
graph = builder.compile(checkpointer=memory)
```

*   **MemorySaver**: これが「AI の記憶領域」です。今回はプログラムが動いている間だけ有効なメモリを使っていますが、ここをデータベースに変えれば、PCを再起動しても忘れないようになります。
*   **checkpointer=memory**: グラフを作るときに「この記憶領域を使ってね」と指定することで、グラフが終了（END）しても、その時の状態（State）がメモリに保存されるようになります。

## 4. 実際に会話してみる (34-58行目)

### ① 1回目の会話（田中さんとして）

```python
config = {"configurable": {"thread_id": "user-1"}}
# ... "私の名前は田中です" と話しかける
```

*   **thread_id="user-1"**: これが「会員番号」のようなものです。「会員番号 user-1 番の会話記録」としてメモリに保存されます。
*   AI は「こんにちは、田中さん」と返します。この時点で、メモリには「私は田中です」という情報が記録されます。

### ② 2回目の会話（同じ田中さんとして）

```python
# 同じ config (thread_id="user-1") を使う
# ... "私の名前を覚えていますか？" と聞く
```

*   **同じ thread_id** を使って話しかけると、LangGraph はメモリから「user-1 の前回の会話」を引っ張り出してきます。
*   AI は「前回の会話」＋「今回の質問」をセットで受け取るので、「はい、田中さんですね！」と答えられます。

### ③ 別の人の会話（user-2として）

```python
config_2 = {"configurable": {"thread_id": "user-2"}}
# ... "私の名前を覚えていますか？" と聞く
```

*   **違う thread_id** を使うと、LangGraph は「あ、新規のお客さんだ」と判断して、真っ白な状態から会話を始めます。
*   当然、田中さんのことは知らないので、「知りません」と答えます。

## まとめ

このコードがやっていることは、**「`thread_id` という鍵を使って、会話の記憶を引き出したり、新しく保存したりする」** ということです。これが Web アプリで「ログインユーザーごとに会話を分ける」ための基本技術になります。