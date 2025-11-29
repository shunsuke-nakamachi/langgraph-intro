# Practice ディレクトリ

このディレクトリは **LangGraph の 5 つの機能（永続化、Human‑in‑the‑loop、Streaming、Subgraphs、Reflection）** を組み合わせた **ファクトチェッカー** のサンプル実装を提供します。

> **目的**
> - 新メンバーが LangGraph の全体像と実装パターンを体感できる教材になる。
> - ターミナルだけで完結し、後から Web UI や永続化・Human‑in‑the‑loop の拡張が容易になる設計。

---

## ディレクトリ構成
```
practice/
├─ README.md                # 本ファイル（概要・手順・設計）
├─ fact_checker_bot.py      # メインスクリプト（全体グラフ）
├─ search_subgraph.py       # 検索サブグラフ（外部 API 呼び出し）
├─ verify_subgraph.py       # 事実検証サブグラフ（LLM に評価させる）
├─ utils.py                 # ログ・リトライヘルパー（任意）
├─ .env                     # OpenAI・検索 API キー（training/.env をコピー）
└─ requirements.txt         # 必要パッケージ（training/requirements.txt を流用）
```

## 1️⃣ 目的と概要
- **ファクトチェッカー**：ユーザーが投げた質問に対し、検索 → 事実検証 → スコア評価 → 必要なら再検索 のループで最終的に「信頼できる」回答を生成します。
- **LangGraph の学習ポイント**
  1. **永続化**（`MemorySaver`／`PostgresSaver`）
  2. **Human‑in‑the‑loop**（途中で手動承認を入れられる）
  3. **Streaming**（`graph.astream_events` でリアルタイム出力）
  4. **Subgraphs**（検索・検証を独立したサブグラフとして切り出す）
  5. **Reflection**（LLM が自己評価し、ループ継続か終了か決定）

## 2️⃣ 実行手順（ターミナルだけ）
```bash
# 1. 仮想環境作成・有効化（まだ無ければ）
python3 -m venv .venv
source .venv/bin/activate

# 2. 依存パッケージをインストール
pip install -r requirements.txt

# 3. .env を用意（training/.env をコピー）
cp ../training/.env .

# 4. ファクトチェッカーを実行
python fact_checker_bot.py "地球温暖化は本当に起きているか？"
```

実行例（抜粋）
```
--- Fact‑Checker 開始 ---
[search] 検索結果: …（検索 API から取得した抜粋）
[verify] 評価スコア: 0.82
[reflector] good   # ループ終了
--- 完了 ---
最終回答: 「地球温暖化は…」 (LLM がまとめた事実ベースの回答)
```

- **Streaming**：`search` と `verify` の結果がリアルタイムで表示されます。
- **Reflection**：スコアが 0.7 以上、または上限回数 (`MAX_LOOPS = 4`) に達したら自動で終了します。

## 3️⃣ 各ファイルの設計概要
### `fact_checker_bot.py`（メイングラフ）
- **State**: `messages`（LLM の入出力履歴） + `loop_count`（Reflection 用カウンタ）
- **ノード**
  - `router`：質問受取 → `search`、または `reflector` が返す verdict に応じて `finalizer` へ遷移。
  - `search`：`search_subgraph` を呼び出すラッパー。
  - `verify`：`verify_subgraph` を呼び出すラッパー。
  - `reflector`：LLM が生成したスコアを評価し、`good`/`needs_fix` を返す。
  - `finalizer`：最終回答を整形して出力。
- **エッジ**
```
START → router → search → verify → reflector → (continue) router
                                   │
                                   └─► (end) finalizer → END
```
- **Reflection ロジック**（`router` 内）
  ```python
  MAX_LOOPS = 4
  if verdict == "good" or state["loop_count"] >= MAX_LOOPS:
      return "finalizer"
  return "search"
  ```
- **Streaming**：`graph.astream_events(init_state)` で各ノードの `messages` を逐次 `print`。

### `search_subgraph.py`
- **State**: `query`, `results`, `messages`
- **ノード**
  1. `prepare_query` – ユーザー質問を検索クエリに整形。
  2. `call_api` – Wikipedia/SerpAPI へ GET リクエスト（`aiohttp` 使用）。
  3. `extract_snippet` – 取得した JSON/HTML から要点だけ抽出（LLM に要約させるプロンプト）。
  4. `output` – `messages` に `AIMessage(content=snippet)` を格納し、メインへ返す。
- **エッジ**
```
START → prepare_query → call_api → extract_snippet → output → END
```
- **永続化**（任意）: `MemorySaver` で `query → results` をキャッシュすれば同一質問の再利用が可能。

### `verify_subgraph.py`
- **State**: `evidence`, `verdict`, `messages`
- **ノード**
  1. `build_prompt` – 証拠テキストを LLM に評価させるプロンプトを作成。
  2. `call_llm` – `llm.invoke([HumanMessage(content=prompt)])`。
  3. `parse_score` – LLM の返答から数値スコアを抽出し、`score >= 0.7 → "good"`、それ以外は `"needs_fix"`。
  4. `output` – `messages` に `AIMessage(content=verdict)` を格納してメインへ返す。
- **エッジ**
```
START → build_prompt → call_llm → parse_score → output → END
```

### `utils.py`（任意）
```python
import datetime, functools, aiohttp, asyncio

def log_event(name: str, msg: str):
    ts = datetime.datetime.now().isoformat()
    print(f"[{ts}] [{name}] {msg}")

def retry(max_attempts: int = 3, backoff: float = 0.5):
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return await fn(*args, **kwargs)
                except Exception:
                    attempts += 1
                    if attempts == max_attempts:
                        raise
                    await asyncio.sleep(backoff * attempts)
        return wrapper
    return decorator
```
- `log_event` は `graph.astream_events` の出力と合わせてタイムスタンプ付きで表示。<br>- `retry` は外部 API 呼び出し（`search_subgraph`）でのリトライに利用。

## 4️⃣ 拡張アイデア（次のステップ）
| 項目 | 内容 | 実装ポイント |
|------|------|---------------|
| **永続化** | `MemorySaver` → `PostgresSaver` に差し替えて検索履歴を DB に保存 | `graph.compile(checkpointer=PostgresSaver(...))` |
| **Human‑in‑the‑loop** | `reflector` が `needs_fix` を返したら `input("承認してください (y/n): ")` で手動承認 | `if input().lower() != "y": continue` |
| **Web UI** | Vite + React で `EventSource`（Server‑Sent Events）を受信し、ストリーミング結果をリアルタイムに表示 | `fetch('/stream', {method: 'GET'})` |
| **ツール呼び出し** | 検索サブグラフで Wikipedia だけでなく、ニュース API や画像検索も組み込む | `tool` ノードを追加し `add_conditional_edges` で分岐 |
| **マルチエージェント** | 複数のサブグラフ（例：リサーチ → 要約 → 検証）を並列に走らせ、結果を統合 | `graph.add_edge(..., parallel=True)` |

## 5️⃣ 参考リンク・リソース
- LangGraph 公式ドキュメント: https://langchain-ai.github.io/langgraph/ 
- OpenAI API キー取得方法: https://platform.openai.com/account/api-keys 
- Wikipedia API（例）: https://en.wikipedia.org/api/rest_v1/#/ 
- SerpAPI（検索）: https://serpapi.com/ 

---

**この README をベースに**、まずは `fact_checker_bot.py` と 2 つのサブグラフを実装し、`python fact_checker_bot.py "質問"` で動作確認してください。拡張は後から段階的に追加していく形で問題ありません。
