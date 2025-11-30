# Error Handling & Recovery（エラーハンドリングとリカバリー）

このレッスンでは、**LangGraphでノード実行時のエラー処理とリトライロジック**を学びます。

実運用では、外部APIの呼び出し、ネットワークエラー、タイムアウト、レート制限など、様々なエラーが発生する可能性があります。これらのエラーに対して適切に対処することで、堅牢なシステムを構築できます。

**注意**: デモ用にエラーを意図的に発生させています。エラーの発生頻度を調整したい場合は、`error_handling_bot.py`の`API_ERROR_PROBABILITY`定数（デフォルト: 0.5 = 50%）を変更してください。

## 学ぶこと

1. **基本的なエラーハンドリング**: try-except文を使ったエラー処理
2. **リトライロジック**: tenacityライブラリを使った自動リトライ
3. **タイムアウト処理**: 長時間処理のタイムアウト検出と対処
4. **エラー状態の管理**: Stateでエラー情報を管理し、フォールバック処理を実装
5. **エラーログの記録**: デバッグとモニタリングのためのログ記録

## なぜ必要か

### 1. 実運用でのエラーの種類

実運用では、以下のような様々なエラーが発生します：

- **ネットワークエラー**: API接続の失敗、タイムアウト
- **レート制限**: API呼び出し回数の制限
- **サービス障害**: 外部サービスの一時的な障害
- **データエラー**: 不正なデータ形式、バリデーションエラー
- **リソース不足**: メモリ不足、CPU負荷

### 2. エラーハンドリングの重要性

適切なエラーハンドリングがないと：

- **システム全体が停止**: 1つのノードのエラーで全体が止まる
- **ユーザー体験の悪化**: エラーメッセージが表示されない、処理が途中で止まる
- **デバッグの困難**: エラーの原因が特定できない
- **リソースの無駄**: 失敗した処理を何度も繰り返す

### 3. エラーハンドリングの利点

- **堅牢性の向上**: 一時的なエラーから自動的に回復
- **ユーザー体験の向上**: 適切なエラーメッセージとフォールバック処理
- **運用性の向上**: エラーログで問題を早期発見
- **コスト削減**: 不要なリトライを避け、効率的な処理

## コードの解説

### 1. State の定義

```python
class State(TypedDict):
    messages: Annotated[List, add_messages]
    query: str
    result: Optional[str]
    error_count: int  # エラー発生回数をカウント
    last_error: Optional[str]  # 最後に発生したエラーメッセージ
    retry_count: int  # リトライ回数
```

**エラー状態の管理**:
- **`error_count`**: エラーが発生した回数をカウント（リカバリー判断に使用）
- **`last_error`**: 最後に発生したエラーのメッセージ（デバッグに使用）
- **`retry_count`**: リトライした回数（モニタリングに使用）

### 2. リトライロジック（tenacityライブラリ）

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError
)

@retry(
    stop=stop_after_attempt(3),  # 最大3回までリトライ
    wait=wait_exponential(multiplier=1, min=1, max=10),  # 指数バックオフ
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),  # 特定のエラーのみリトライ
    reraise=True  # 最終的に失敗した場合は例外を再発生
)
def api_node_with_retry(state: State) -> dict:
    # ...
```

**tenacityの設定**:

1. **`stop=stop_after_attempt(3)`**: 最大3回までリトライを試みる
2. **`wait=wait_exponential(...)`**: 指数バックオフ（1秒、2秒、4秒...）で待機時間を増やす
   - これにより、一時的な負荷の問題が解決する時間を与える
3. **`retry=retry_if_exception_type(...)`**: 特定のエラーのみリトライ
   - `ConnectionError`や`TimeoutError`は一時的なエラーの可能性が高い
   - `ValueError`などはリトライしても解決しないので、リトライしない
4. **`reraise=True`**: 最終的に失敗した場合は例外を再発生させ、上位で処理できるようにする

**指数バックオフの仕組み**:
```
試行1: 即座に実行
試行2: 1秒待機
試行3: 2秒待機
試行4: 4秒待機
...
```

これにより、一時的な負荷の問題が解決する時間を与えつつ、無駄なリトライを避けます。

### 3. 基本的なエラーハンドリング

```python
def llm_node_with_error_handling(state: State) -> dict:
    try:
        response = llm.invoke(messages)
        return {"messages": [response], "error_count": 0}
    except Exception as e:
        # エラーログを記録
        error_msg = log_error("LLM Node", e)
        
        # フォールバックメッセージを返す
        fallback_message = AIMessage(
            content="申し訳ございません。一時的にサービスに接続できませんでした。"
        )
        
        return {
            "messages": [fallback_message],
            "error_count": error_count + 1,
            "last_error": error_msg
        }
```

**ポイント**:
- **try-except文**: エラーを捕捉して処理を継続
- **エラーログ**: デバッグのためのログ記録
- **フォールバック処理**: エラー時でもユーザーに適切なメッセージを返す
- **エラー状態の更新**: Stateでエラー情報を管理

### 4. タイムアウト処理

```python
def timeout_simulation_node(state: State) -> dict:
    processing_time = random.uniform(0.5, 3.0)
    
    if processing_time > TIMEOUT_THRESHOLD:
        raise TimeoutError(f"処理がタイムアウトしました")
    
    time.sleep(processing_time)
    return {"result": f"処理完了"}
```

**タイムアウトの検出**:
- 処理時間が一定時間を超えた場合、`TimeoutError`を発生
- リトライロジックと組み合わせることで、一時的な遅延から回復

### 5. エラーリカバリーノード

```python
def error_recovery_node(state: State) -> dict:
    error_count = state.get("error_count", 0)
    
    if error_count >= 3:
        # エラーが多すぎる場合は処理を中断
        return {
            "messages": [AIMessage(content="複数のエラーが発生しました。")],
            "result": "エラーが多すぎるため、処理を中断しました"
        }
    else:
        # エラーが少ない場合は処理を継続
        return {
            "messages": [AIMessage(content="一時的なエラーが発生しましたが、処理は続行します。")],
            "error_count": 0  # リカバリーしたのでエラーカウントをリセット
        }
```

**リカバリーの判断**:
- エラー回数に応じて、処理を継続するか中断するかを判断
- 一時的なエラーと永続的なエラーを区別

### 6. 条件分岐によるエラーハンドリング

```python
def check_error_state(state: State) -> str:
    error_count = state.get("error_count", 0)
    result = state.get("result")
    
    if error_count > 0 or result is None:
        return "recovery"  # エラーがある場合はリカバリーノードへ
    else:
        return "end"  # 正常な場合は終了
```

**エラー状態に応じた分岐**:
- エラーが発生している場合、リカバリーノードに遷移
- 正常な場合は終了

## 実行方法

```bash
python error_handling_bot.py
```

このスクリプトは、以下の3つのケースを順に実行します：

1. **正常な処理**: エラーが発生しない場合の動作を確認
2. **APIエラー（リトライ成功）**: APIエラーが発生するが、リトライで成功する場合
3. **タイムアウトエラー**: タイムアウトが発生する場合の動作を確認

実行ログを見ることで、エラーハンドリングとリトライの動作が確認できます。

## 実務での応用例

### 1. 外部API呼び出し

```python
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, HTTPError))
)
def call_external_api(url: str) -> dict:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()
```

### 2. データベース接続

```python
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    retry=retry_if_exception_type((OperationalError, InterfaceError))
)
def query_database(query: str) -> List[dict]:
    # データベースクエリ
    pass
```

### 3. ファイル処理

```python
def process_file(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        return {"content": content, "error_count": 0}
    except FileNotFoundError:
        return {"content": None, "error_count": 1, "last_error": "ファイルが見つかりません"}
    except PermissionError:
        return {"content": None, "error_count": 1, "last_error": "ファイルへのアクセス権限がありません"}
```

## ベストプラクティス

### 1. エラーの分類

- **一時的なエラー**: リトライすべき（ConnectionError, TimeoutError）
- **永続的なエラー**: リトライしない（ValueError, FileNotFoundError）
- **致命的なエラー**: 処理を中断すべき（MemoryError, SystemError）

### 2. リトライ戦略

- **指数バックオフ**: リトライ間隔を徐々に増やす
- **最大リトライ回数**: 無限ループを防ぐ
- **リトライ対象の限定**: リトライすべきエラーのみを対象にする

### 3. エラーログ

- **構造化ログ**: エラー情報を構造化して記録
- **コンテキスト情報**: エラー発生時の状態を記録
- **アラート**: 重要なエラーは通知する

### 4. フォールバック処理

- **デフォルト値**: エラー時でもデフォルト値を返す
- **代替処理**: 別の方法で処理を試みる
- **ユーザーへの通知**: 適切なエラーメッセージを表示

## まとめ

Error Handling & Recoveryは、**堅牢なシステムを構築するための重要な技術**です。

- **エラーの予測**: どのようなエラーが発生する可能性があるかを予測
- **適切な対処**: エラーの種類に応じて適切な対処を行う
- **自動回復**: 一時的なエラーから自動的に回復する仕組みを実装
- **監視とログ**: エラーを記録し、問題を早期発見

実運用では、エラーハンドリングの品質がシステムの信頼性を大きく左右します。適切なエラーハンドリングを実装することで、ユーザー体験を向上させ、運用コストを削減できます。

