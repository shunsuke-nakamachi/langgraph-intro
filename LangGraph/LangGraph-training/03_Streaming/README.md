# 03. Streaming（ストリーミング）

このコードは、**「AI が生成する文章を、1文字ずつリアルタイムに表示する」** という機能（ストリーミング）を実現するサンプルです。
ChatGPT のように、文字がカタカタと表示される体験を作るために必須の技術です。

## 仕組みのポイント

### 1. `astream_events` (非同期ストリーミング)

```python
async for event in graph.astream_events(inputs, version="v1"):
```

*   **astream_events**: グラフの実行中に発生するあらゆる出来事（ノードの開始/終了、ツールの呼び出し、LLMの生成など）を、発生した瞬間にイベントとして受け取ることができます。
*   これを使うには `asyncio`（非同期処理）が必要です。

### 2. `on_chat_model_stream` イベント

```python
if kind == "on_chat_model_stream":
    content = event["data"]["chunk"].content
    print(content, end="", flush=True)
```

*   LangGraph の中ではたくさんのイベントが起きますが、私たちが欲しいのは「LLM が文字を生成した瞬間」だけです。
*   それが **`on_chat_model_stream`** というイベントです。
*   このイベントが来るたびに、中身（`chunk.content`）を取り出して画面に表示することで、流れるような表示を実現します。

## ストリーミングの原理

ストリーミングができる原理は、**「サーバー（LLM）からデータが小分けで送られてくる」** という仕組みにあります。

### 1. 通常の通信（非ストリーミング）
通常、Web や API の通信は「リクエスト」→「処理完了まで待機」→「レスポンス（全データ）」という流れです。

*   **例**: ラーメン屋で注文。
    1.  注文する。
    2.  麺を茹で、スープを作り、盛り付けるまで**厨房で全部完成するのを待つ**。
    3.  完成したラーメンがドーンと出てくる。
    *   **欠点**: 完成するまで何も出てこないので、待ち時間が長く感じる。

### 2. ストリーミング通信
一方、ストリーミングは「できた部分から順次送る」という方式です。LLM は文章を「トークン（単語や文字の断片）」単位で生成しているので、1トークン生成するたびに即座にクライアントに送信します。

*   **例**: わんこそば。
    1.  注文する。
    2.  店員さんがそばを茹でる。
    3.  **一口分茹で上がったら、即座にお椀に入れる**。
    4.  次の一口分ができたら、また入れる。
    *   **利点**: 最初のデータが届くのが非常に速い。

### 3. LangGraph における実装 (`astream_events`)

LangGraph の `astream_events` は、この「わんこそば」の仕組みをプログラムで扱えるようにしたものです。

1.  **イベント監視**: グラフを実行すると、LangGraph は内部で起きていること（ノード開始、LLM生成など）を常に監視します。
2.  **`on_chat_model_stream`**: LLM が「1トークン生成しました！」という信号（イベント）を送ってくると、LangGraph はそれをキャッチします。
3.  **リアルタイム処理**: 私たちのコード（`async for event ...`）は、そのイベントを受け取り、中身の文字（`chunk.content`）を即座に `print` します。

```python
# イメージ
async for event in graph.astream_events(...):
    # LLM: "桃" を生成しました！ -> eventとして届く -> print("桃")
    # LLM: "太" を生成しました！ -> eventとして届く -> print("太")
    # LLM: "郎" を生成しました！ -> eventとして届く -> print("郎")
    # ...
```

このように、**「生成された瞬間に受け取って表示する」** を高速に繰り返すことで、あたかもリアルタイムに文字が打たれているように見えるのです。

## コードの詳細解説

### 1. `event["data"]["chunk"].content` とは？

これは、**「LLM から送られてきた、ほんのひとかけらの文字データ」** を取り出すための呪文です。

分解して説明します。

1.  **`event`**: LangGraph から送られてくる「出来事」の通知です。辞書型（JSONのような形）をしています。
2.  **`event["data"]`**: その出来事に関する詳しいデータが入っています。
3.  **`event["data"]["chunk"]`**: LLM が生成した「断片（チャンク）」です。これは `AIMessageChunk` というオブジェクトです。
4.  **`.content`**: その断片の中身（文字列）です。例えば「桃」「太」「郎」のような1文字〜数文字が入っています。

**イメージ:**
```python
event = {
    "event": "on_chat_model_stream",
    "data": {
        "chunk": AIMessageChunk(content="桃") # ← これが chunk
    }
}
# event["data"]["chunk"].content は "桃" になります
```

### 2. コード全体の流れ

#### ① 準備と非同期処理の導入 (1-7行目)

```python
import asyncio
# ...
```

*   **`import asyncio`**: ストリーミングは「待ち時間」を有効活用する技術なので、Python の「非同期処理（asyncio）」という仕組みを使います。これを使うと、データが届くのを待っている間に別のこと（画面表示など）ができます。

#### ② グラフの定義 (12-28行目)

```python
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
# ...
graph = builder.compile()
```

*   ここはこれまでの例とほぼ同じです。
*   **`streaming=True`**: 「この LLM はストリーミングで使うよ」という宣言です（最近の LangChain では自動判定されることも多いですが、明示するのが丁寧です）。

#### ③ メイン処理 (31-51行目) ★ここが主役

```python
async def main():
    # ...
    async for event in graph.astream_events(..., version="v1"):
```

*   **`async def`**: 非同期関数であることを宣言しています。
*   **`astream_events`**: これがストリーミングの心臓部です。グラフを実行しつつ、内部で起きる「イベント」をリアルタイムで受信するための特別なメソッドです。
*   **`async for`**: イベントが次々と流れてくるので、それを非同期ループで受け取ります。

#### ④ イベントの選別と表示 (40-49行目)

```python
        kind = event["event"]
        
        # 'on_chat_model_stream' というイベントが、LLMからトークンが届いた瞬間です
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                print(content, end="", flush=True)
```

*   **`kind`**: イベントの種類です。LangGraph は「ノードが始まった（on_chain_start）」「ツールが終わった（on_tool_end）」など色々なイベントを送ってきます。
*   **`if kind == "on_chat_model_stream"`**: 私たちが欲しいのは「文字が生成された瞬間」だけなので、このイベントだけをフィルタリング（選別）します。
*   **`print(..., end="", flush=True)`**:
    *   `end=""`: 通常 `print` は改行しますが、改行せずに横につなげて表示します。
    *   `flush=True`: Python は通常、ある程度文字が溜まってから画面に出そうとしますが、強制的に「今すぐ出せ！」と指示します。これがないとカクカクした表示になります。

#### ⑤ 実行 (53-54行目)

```python
if __name__ == "__main__":
    asyncio.run(main())
```

*   非同期関数 `main()` は普通には呼べないので、`asyncio.run()` を使って起動します。