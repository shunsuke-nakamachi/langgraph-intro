# LangGraph & LangSmith 学習リポジトリ

LangChainとLangGraphを使ったAIエージェント開発、およびLangSmithを使ったデバッグ・監視の学習リポジトリです。

このリポジトリでは、LangGraphの基礎から応用、実践的なアプリケーション開発まで、段階的に学習できるよう構成されています。

## 📚 プロジェクト構成

```text
langgraph-practice/
├── LangGraph/              # LangGraph学習コンテンツ
│   ├── LangGraph-training/ # 基礎編（5つのセクション）
│   ├── LangGraph-advance/   # 応用編（6つのセクション）
│   ├── LangGraph-practice/  # 実践編（ファクトチェッカー）
│   └── README.md           # LangGraph学習ガイド
│
├── LangSmith/              # LangSmith学習コンテンツ
│   ├── LangSmith-training/ # LangSmithトレーシング・デバッグ
│   ├── images/             # スクリーンショット
│   └── README.md           # LangSmith学習ガイド
│
└── README.md               # 本ファイル（全体ガイド）
```

## 🎯 学習の流れ

### ステップ1: LangGraph基礎編（LangGraph-training）

LangGraphの基本機能を段階的に学習します。

1. **00_Introduction**: 基本的なチャットボットとツール使用
2. **01_Persistence**: 会話履歴の永続化（`MemorySaver`、`thread_id`）
3. **02_Human_in_the_loop**: 人間による承認・介入（`interrupt_before`、`interrupt_after`）
4. **03_Streaming**: リアルタイム出力（`astream_events`、`astream_log`）
5. **04_Subgraphs**: サブグラフ（グラフの階層構造）
6. **05_Reflection**: 自己省察・自己修正（循環ループと脱出条件）

詳細は [LangGraph/README.md](LangGraph/README.md) を参照してください。

### ステップ2: LangGraph応用編（LangGraph-advance）

より高度な機能を学習します。

1. **06_Tools**: ツール呼び出し（`ToolNode`、`tools_condition`）
2. **07_Parallel**: 並列実行
3. **08_TimeTravel**: タイムトラベル（状態の巻き戻し）
4. **09_Supervisor**: スーパーバイザーパターン（複数エージェントのルーティング）
5. **10_ErrorHandling**: エラーハンドリング
6. **11_MultiAgent**: マルチエージェントシステム（複数エージェントの協調）

### ステップ3: LangGraph実践編（LangGraph-practice）

基礎編で学んだ5つの機能を組み合わせたファクトチェッカーの実装例です。

- 検索 → 事実検証 → スコア評価 → 必要に応じて再検索のループ

詳細は [LangGraph/LangGraph-practice/README.md](LangGraph/LangGraph-practice/README.md) を参照してください。

### ステップ4: LangSmith（デバッグ・監視）

LangSmithを使った実行ログの可視化とデバッグを学習します。

- 環境変数の設定
- 自動トレーシング
- LangSmith UIでの確認
- デバッグの活用
- パフォーマンス分析

詳細は [LangSmith/README.md](LangSmith/README.md) を参照してください。

## 🚀 セットアップ手順

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd langgraph-practice
```

### 2. Python環境の準備

Python 3.8以上が必要です。

```bash
# 仮想環境の作成
python3 -m venv .venv

# 仮想環境の有効化
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
```

### 3. 依存パッケージのインストール

各ディレクトリに `requirements.txt` があります。学習するセクションに応じてインストールしてください。

```bash
# LangGraph基礎編の場合
cd LangGraph/LangGraph-training
pip install -r requirements.txt

# LangGraph応用編の場合
cd LangGraph/LangGraph-advance
pip install -r requirements.txt

# LangSmithの場合
cd LangSmith/LangSmith-training
pip install -r requirements.txt
```

### 4. 環境変数の設定

各ディレクトリに `.env` ファイルを作成し、以下の環境変数を設定してください。

```env
# OpenAI API キー（必須）
OPENAI_API_KEY=your_openai_api_key_here

# LangSmith トレーシング（オプション）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=langgraph-practice
```

**重要**: `.env` ファイルは `.gitignore` に含まれているため、Gitにはコミットされません。各環境で個別に作成してください。

### 5. LangSmithアカウントの作成（オプション）

LangSmithを使う場合は、以下の手順でアカウントを作成してください。

1. <https://smith.langchain.com/> にアクセス
2. アカウント作成（GitHubアカウントまたはメールアドレスでサインアップ）
3. APIキーの取得（アカウント設定 → API Keys）
4. `.env` ファイルに `LANGCHAIN_API_KEY` を設定

詳細は [LangSmith/README.md](LangSmith/README.md) を参照してください。

## 📖 各ディレクトリの説明

### LangGraph/

LangGraphの学習コンテンツです。

- **LangGraph-training/**: 基礎編。永続化、Human-in-the-loop、Streaming、Subgraphs、Reflectionを学習
- **LangGraph-advance/**: 応用編。ツール、並列実行、タイムトラベル、スーパーバイザー、エラーハンドリング、マルチエージェントを学習
- **LangGraph-practice/**: 実践編。ファクトチェッカーの実装例

詳細は [LangGraph/README.md](LangGraph/README.md) を参照してください。

### LangSmith/

LangSmithの学習コンテンツです。

- **LangSmith-training/**: LangSmithを使ったトレーシング・デバッグの学習
- **images/**: LangSmith UIのスクリーンショット

詳細は [LangSmith/README.md](LangSmith/README.md) を参照してください。

## 🛠️ 必要な環境

- **Python**: 3.8以上
- **OpenAI API キー**: LangGraphの実行に必要
- **LangSmith API キー**: LangSmithのトレーシング機能を使用する場合（オプション）

## 📚 参考リンク

### LangGraph

- **LangGraph公式ドキュメント**: <https://langchain-ai.github.io/langgraph/>
- **LangChain公式ドキュメント**: <https://python.langchain.com/>
- **Qiita記事（LangGraph入門）**: <https://qiita.com/sakuraia/items/27db3f118e0ee41c54c1>

### LangSmith

- **LangSmith公式サイト**: <https://smith.langchain.com/>
- **LangSmithドキュメント**: <https://docs.smith.langchain.com/>
- **APIキーの取得**: <https://smith.langchain.com/settings>

### OpenAI

- **OpenAI Platform**: <https://platform.openai.com/>
- **APIキーの取得**: <https://platform.openai.com/account/api-keys>

## 💡 学習のコツ

1. **順番に学習**: 基礎編から順番に学習することで、段階的に理解を深められます
2. **コードを実行**: 各セクションのコードを実際に実行して、動作を確認しましょう
3. **コードを読む**: コードを読んで、どのように実装されているか理解しましょう
4. **自分で改造**: コードを改造して、自分のアイデアを試してみましょう
5. **LangSmithを活用**: LangSmithを使って実行過程を可視化し、デバッグに活用しましょう

## 🤝 貢献

このリポジトリは学習用のリポジトリです。改善提案やバグ報告は、IssueやPull Requestでお願いします。

## 📝 ライセンス

このリポジトリは学習目的で作成されています。各ライブラリのライセンスに従ってください。

## 🔗 関連リポジトリ

- [LangChain公式リポジトリ](<https://github.com/langchain-ai/langchain>)
- [LangGraph公式リポジトリ](<https://github.com/langchain-ai/langgraph>)