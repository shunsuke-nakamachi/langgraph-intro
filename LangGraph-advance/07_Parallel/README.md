# Parallel Execution & Configuration (並列実行と動的設定)

このレッスンでは、LangGraph の強力な2つの機能、**並列実行 (Parallel Execution)** と **動的設定 (Configuration)** を学びます。
これらを組み合わせることで、「複数の検索を同時に高速に行い」かつ「ユーザーや状況に応じて振る舞いを変える」高度なエージェントが作成できます。

## 1. Parallel Execution (並列実行)

複数のノードを同時に実行し、処理時間を短縮するテクニックです。

### 仕組み: Fan-out / Fan-in
*   **Fan-out (拡散)**: 1つのノードから複数のノードへ同時に遷移させます。LangGraph では、同じノードから複数のエッジを引くだけで自動的に並列実行されます。
*   **Fan-in (集約)**: 複数のノードの結果を1つのノードで受け取ります。LangGraph はすべての親ノードの完了を待ってから集約ノードを実行します。

### 実装のポイント: Reducer
並列に実行された結果を安全にリストにまとめるために、`Annotated` と `operator.add` を使用します。

```python
import operator
from typing import Annotated, List

class State(TypedDict):
    # 普通の List だと上書き競合が起きるため、add で追記するように指示
    results: Annotated[List[str], operator.add]
```

## 2. Configuration (動的設定)

コードを書き換えずに、実行時（Runtime）にボットの挙動を変更する機能です。
「Aさんには丁寧語で」「Bさんには関西弁で」「テスト時は GPT-3.5 で」といった切り替えが簡単にできます。

### 実装のポイント
ノード関数で `config: RunnableConfig` を受け取り、そこから設定値を読み出します。

```python
from langchain_core.runnables import RunnableConfig

def aggregator(state: State, config: RunnableConfig):
    # config["configurable"] から設定を取得
    conf = config.get("configurable", {})
    model = conf.get("model_name", "gpt-4o-mini")
    system_msg = conf.get("system_message", "You are a helpful assistant.")
    
    # 設定を使って LLM を初期化
    llm = ChatOpenAI(model=model)
    # ...
```

## 実行方法

```bash
python parallel_bot.py
```

このスクリプトは、以下の3つのパターンを順に実行します。

1.  **デフォルト設定**: 並列検索を行い、標準的な回答を生成します。
2.  **関西弁設定**: システムプロンプトを差し替え、関西弁で回答させます。
3.  **モデル切り替え**: モデルを `gpt-3.5-turbo` に切り替え、学術的な口調で回答させます。

出力ログを見ることで、`[Wikipedia]`, `[News]`, `[Blogs]` が並列に動いている様子や、設定によって回答が変化する様子が確認できます。
