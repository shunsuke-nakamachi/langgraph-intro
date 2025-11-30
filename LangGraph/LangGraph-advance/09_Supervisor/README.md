# Supervisor Pattern（スーパーバイザーパターン）

このレッスンでは、**複数の専門エージェントを管理し、ユーザーの質問に応じて適切なエージェントに自動的にルーティングする** 仕組み（Supervisor Pattern）を学びます。

実務では、「質問に答えるエージェント」「計算を行うエージェント」「情報を検索するエージェント」など、複数の専門家を用意し、ユーザーの意図に応じて最適な専門家を選んで仕事を任せるという設計がよく使われます。

## 学ぶこと

1. **複数の専門エージェントの定義**: それぞれが特定のタスクに特化したグラフ（ノード）を作る
2. **Supervisor（スーパーバイザー）の実装**: ユーザーの質問を分析し、どのエージェントを使うべきか判断する
3. **動的ルーティング**: 条件分岐を使って、Supervisorの判断に基づいて適切なエージェントに振り分ける
4. **エージェント間の独立性**: 各エージェントは独立して動作し、互いに干渉しない設計

## なぜ必要か

### 1. 専門性の分離

1つのエージェントに全ての機能を持たせると、以下の問題が発生します：

- **プロンプトが複雑になる**: 「質問にも答え、計算もでき、検索もできる」という万能エージェントを作ろうとすると、プロンプトが長くなり、品質が下がる
- **判断が曖昧になる**: LLMが「これは質問なのか、計算なのか」を判断する必要があり、誤判断が起きやすい
- **拡張が困難**: 新しい機能を追加するたびに、既存のプロンプトを修正する必要がある

### 2. Supervisor Patternの利点

- **専門性の向上**: 各エージェントが1つのタスクに集中できるため、品質が向上する
- **明確な責任分離**: 「質問は question_agent」「計算は calculation_agent」と明確に分かれる
- **拡張が容易**: 新しいエージェントを追加するだけで機能を拡張できる
- **デバッグが簡単**: どのエージェントが問題を起こしているか特定しやすい

## コードの解説

### 1. State の定義

```python
class State(TypedDict):
    messages: Annotated[List, add_messages]
    next_agent: str  # 次に実行すべきエージェント名
```

- **`messages`**: 会話履歴を保持（これまでの例と同様）
- **`next_agent`**: Supervisorが決定した「次に実行すべきエージェント名」を格納するフィールド

### 2. 専門エージェントの定義

#### Question Agent（質問エージェント）

```python
def question_agent(state: State):
    """質問に答える専門エージェント"""
    messages = [
        SystemMessage(content="あなたは質問に答える専門家です。..."),
        HumanMessage(content=user_query)
    ]
    response = llm.invoke(messages)
    return {"messages": [response]}
```

- **役割**: 一般的な質問に答える
- **特徴**: シンプルで分かりやすい回答を心がける

#### Calculation Agent（計算エージェント）

```python
def calculation_agent(state: State):
    """計算を行う専門エージェント"""
    messages = [
        SystemMessage(content="あなたは計算の専門家です。..."),
        HumanMessage(content=f"以下の計算を実行してください: {user_query}")
    ]
    response = llm.invoke(messages)
    return {"messages": [response]}
```

- **役割**: 数式や計算問題を解く
- **特徴**: 計算過程も示すことで、透明性を確保

#### Search Agent（検索エージェント）

```python
def search_agent(state: State):
    """情報検索を行う専門エージェント（シミュレーション）"""
    messages = [
        SystemMessage(content="あなたは情報検索の専門家です。..."),
        HumanMessage(content=f"以下のトピックについて、最新の情報を調べて回答してください: {user_query}")
    ]
    response = llm.invoke(messages)
    return {"messages": [response]}
```

- **役割**: 最新情報や事実を調べる
- **特徴**: 実際の実装では、ここでWeb検索APIなどを呼び出す

### 3. Supervisor（スーパーバイザー）の実装 ★ここが核心

この実装では、**LangChainの Structured Output 機能**を使用して、LLMのレスポンスを確実に構造化された形式で取得します。

#### 3.1. Pydanticモデルでレスポンス構造を定義

```python
from enum import Enum
from langchain_core.pydantic_v1 import BaseModel, Field

class AgentChoice(str, Enum):
    """選択可能なエージェントの列挙型"""
    QUESTION = "question_agent"
    CALCULATION = "calculation_agent"
    SEARCH = "search_agent"

class RoutingDecision(BaseModel):
    """Supervisorのルーティング決定を表す構造化データ"""
    agent_name: AgentChoice = Field(
        description="選択されたエージェント名。question_agent, calculation_agent, search_agent のいずれか"
    )
    reason: str = Field(
        description="なぜこのエージェントを選んだかの簡潔な理由（1文程度）"
    )

# Structured Output を使うようにLLMを設定
structured_llm = llm.with_structured_output(RoutingDecision)
```

**ポイント**:
- **Enum型で選択肢を制限**: `AgentChoice` で有効なエージェント名のみを許可
- **Pydanticモデルで型を厳密に定義**: `RoutingDecision` でレスポンスの構造を明確化
- **自動バリデーション**: Pydanticが自動で型チェックとバリデーションを行う

#### 3.2. Supervisorの実装

```python
def supervisor(state: State) -> dict:
    """ユーザーの質問を分析し、適切なエージェントにルーティングする（Structured Output使用）"""
    user_message = state["messages"][-1].content
    
    routing_prompt = f"""
ユーザーの質問を読んで、どの専門エージェントに振り分けるべきか判断してください。

利用可能なエージェント:
1. question_agent: 一般的な質問に答える（例：「Pythonとは？」「AIとは？」「説明して」）
2. calculation_agent: 計算や数式を解く（例：「123 * 456は？」「計算してください」「円の面積」）
3. search_agent: 最新情報や事実を調べる（例：「2024年の最新技術」「最新のトレンド」「調べて」）

ユーザーの質問: {user_message}
"""
    
    messages = [
        SystemMessage(content="あなたは質問を分析して適切な専門家に振り分けるスーパーバイザーです。"),
        HumanMessage(content=routing_prompt)
    ]
    
    # Structured Output を使用して、確実に構造化されたデータを取得
    try:
        decision: RoutingDecision = structured_llm.invoke(messages)
        
        # デバッグ情報を表示
        print(f"  [Supervisor] 選択されたエージェント: {decision.agent_name.value}")
        print(f"  [Supervisor] 理由: {decision.reason}")
        
        agent_name = decision.agent_name.value
        
    except Exception as e:
        # エラーが発生した場合のフォールバック
        print(f"  [Supervisor] 警告: Structured Outputの取得に失敗しました: {e}")
        agent_name = "question_agent"
    
    return {"next_agent": agent_name}
```

**Structured Outputの利点**:

1. **パース処理が不要**: LangChainが自動で構造化してくれるため、手動のパース処理が不要
2. **型安全性**: Enum型で選択肢を制限し、Pydanticが自動でバリデーション
3. **追加情報の取得**: エージェント選択の理由も取得できる（デバッグやログに活用可能）
4. **エラーハンドリングが明確**: 例外処理でエラーを適切にハンドリング

**なぜStructured Outputを使うのか？**

従来の方法（文字列パース）では、以下の問題がありました：

- **LLMの出力形式に依存**: 説明が混ざる可能性がある
- **パース処理が複雑**: 3段階のフォールバック処理が必要
- **型安全性が低い**: 文字列の比較に依存

Structured Outputを使うことで、これらの問題を根本的に解決できます。

**Supervisorの役割**:
1. ユーザーの質問を読み取る
2. LLMに「どのエージェントが適切か」を判断させる
3. 決定したエージェント名を `next_agent` フィールドに格納

**なぜLLMで判断するのか？**
- ルールベース（if-else）では、「123 * 456」のような明らかな計算は判断できるが、「Pythonとは？」のような質問と「Pythonの最新バージョンは？」のような検索クエリを区別するのは難しい
- LLMは文脈を理解できるため、より柔軟で正確な判断ができる

### 4. ルーティング関数（条件分岐）

```python
def route_to_agent(state: State) -> Literal["question_agent", "calculation_agent", "search_agent", "__end__"]:
    """Supervisorが決定したエージェントにルーティングする"""
    next_agent = state.get("next_agent", "question_agent")
    
    # 既にエージェントが実行済みの場合は終了
    if len(state["messages"]) > 1:
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage):
            return "__end__"
    
    return next_agent
```

**この関数の役割**:
- Supervisorが `next_agent` に設定した値（例: `"calculation_agent"`）を読み取る
- その値に基づいて、対応するエージェントノードに遷移する
- 既にエージェントが実行済み（回答が生成済み）なら終了

### 5. グラフの構造

```
START
  ↓
supervisor (質問を分析し、next_agent を決定)
  ↓
[条件分岐: route_to_agent]
  ├─→ question_agent → END
  ├─→ calculation_agent → END
  └─→ search_agent → END
```

**実行の流れ**:
1. ユーザーが質問を入力
2. **Supervisor** が質問を分析し、「これは計算だな」と判断
3. `next_agent = "calculation_agent"` を設定
4. **route_to_agent** が `"calculation_agent"` を読み取り、calculation_agent に遷移
5. **calculation_agent** が計算を実行し、回答を生成
6. END に到達して終了

## 実行方法

```bash
python supervisor_bot.py
```

このスクリプトは、以下の4つのケースを順に実行します：

1. **一般的な質問**: "Pythonとは何ですか？" → `question_agent` にルーティング
2. **計算問題**: "123 * 456 を計算してください" → `calculation_agent` にルーティング
3. **情報検索**: "2024年のAI技術の最新トレンドについて" → `search_agent` にルーティング
4. **複雑な計算**: "半径5cmの円の面積を計算してください" → `calculation_agent` にルーティング

実行ログを見ることで、Supervisorがどのように質問を分析し、適切なエージェントに振り分けているかが確認できます。

## 実務での応用例

### 1. カスタマーサポートボット

```
supervisor
  ├─→ faq_agent (よくある質問に答える)
  ├─→ technical_support_agent (技術的な問題を解決)
  ├─→ billing_agent (請求に関する質問)
  └─→ escalation_agent (人間のオペレーターに引き継ぐ)
```

### 2. コンテンツ生成システム

```
supervisor
  ├─→ blog_writer (ブログ記事を書く)
  ├─→ code_generator (コードを生成)
  ├─→ translator (翻訳する)
  └─→ summarizer (要約する)
```

### 3. データ分析システム

```
supervisor
  ├─→ data_analyzer (データを分析)
  ├─→ chart_generator (グラフを生成)
  ├─→ report_writer (レポートを書く)
  └─→ anomaly_detector (異常を検出)
```

## 拡張アイデア

### 1. エージェントの連携

現在の実装では、1つのエージェントが実行されて終了しますが、複数のエージェントを連携させることも可能です。

例: 「Pythonの最新バージョンを調べて、そのバージョンで動作するコード例を生成してください」
- `search_agent` で最新バージョンを調べる
- `code_generator` でコード例を生成する

### 2. エージェントの品質評価

Supervisorがエージェントの回答を評価し、品質が低い場合は別のエージェントに再ルーティングする仕組みも考えられます。

### 3. エージェントの学習

各エージェントの実行結果を記録し、Supervisorの判断精度を向上させることも可能です。

## まとめ

Supervisor Patternは、**「適材適所」** を実現するための設計パターンです。

- **専門性の向上**: 各エージェントが1つのタスクに集中できる
- **拡張性**: 新しいエージェントを追加するだけで機能を拡張できる
- **保守性**: 各エージェントが独立しているため、デバッグや修正が容易

大規模なエージェントシステムを構築する際には、このパターンが非常に有効です。

---

## 補足: 以前の実装方法（参考情報）

### 文字列パースによる実装

以前は、LLMのレスポンスを文字列として受け取り、手動でパースする方法が使われていました。しかし、この方法には以下の課題がありました：

1. **パース処理が複雑**: 3段階のフォールバック処理が必要
2. **LLMの出力形式に依存**: 説明が混ざる可能性がある
3. **エラーハンドリングが不十分**: デフォルトにフォールバックするだけ
4. **型安全性が低い**: 文字列の比較に依存

**以前の実装例（参考）**:

```python
# 3段階のパース処理が必要だった
response = llm.invoke(messages)
raw_response = response.content.strip()

# 方法1: 文字列検索
for agent in valid_agents:
    if agent in raw_response.lower():
        agent_name = agent
        break

# 方法2: 行ごとにチェック
if agent_name is None:
    lines = raw_response.split('\n')
    for line in lines:
        # ... 複雑な処理

# 方法3: 正規表現
if agent_name is None:
    # ... さらに複雑な処理
```

