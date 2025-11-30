# Time Travel (タイムトラベル)

LangGraph の強力な機能の一つに、**会話の状態を過去に戻して、別の選択肢を試す（Time Travel）** 機能があります。
これは `Checkpointer`（永続化機能）の応用で実現されます。

## 学ぶこと

1.  **`graph.get_state_history(config)`**: 特定のスレッドの過去の状態履歴を取得する。
2.  **`thread_ts` (Thread Timestamp)**: 各状態を一意に識別するID。
3.  **Forking (分岐)**: 過去の `thread_ts` を指定して新しい入力を送ることで、会話の歴史を分岐させる。

## コードの解説

### 1. 履歴の取得と特定

```python
all_states = [s async for s in graph.aget_state_history(config)]
```

*   `aget_state_history` で、そのスレッドの保存された全状態（チェックポイント）を取得できます。
*   リストは新しい順（最新が先頭）に並んでいます。
*   各状態 (`StateSnapshot`) には `config` 属性があり、そこにその時点の `thread_ts` が含まれています。

### 2. 過去からの再開（分岐）

```python
target_config = state.config  # 戻りたい時点の config (thread_ts 含む)

graph.invoke(
    {"messages": [HumanMessage(content="新しい入力")]},
    config=target_config
)
```

*   `invoke` や `astream_events` に渡す `config` に、過去の `thread_ts` を含めると、LangGraph は**その時点から処理を再開**します。
*   新しい入力を与えると、元の履歴は残ったまま、新しい履歴のブランチ（分岐）が作成されます。

## 実行方法

```bash
python timetravel_bot.py
```

### 実行の流れ

1.  **Step 1-3**: 普通に会話します。「うどんが好き」と伝えます。
2.  **Time Travel**: 履歴から「うどんが好き」と言った直後の状態を探し出します。
3.  **Step 4**: その時点に戻って、「やっぱりそばが好き」と言い直します。
4.  **Step 5**: 「私の好きな食べ物は？」と聞くと、Bot は（うどんではなく）**「そば」** と答えるはずです。

これにより、過去の事実が書き換わった（別の世界線に移動した）ことが確認できます。
