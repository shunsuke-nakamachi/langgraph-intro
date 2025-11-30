"""
Error Handling & Recovery（エラーハンドリングとリカバリー）

この実装では、LangGraphでノード実行時のエラー処理とリトライロジックを学びます。

【学ぶこと】
1. 基本的なエラーハンドリング（try-except）
2. リトライロジック（tenacityライブラリの使用）
3. タイムアウト処理
4. エラー状態の管理とフォールバック
5. エラーログの記録
"""
import asyncio
import time
import random
from typing import TypedDict, Annotated, List, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError
)

# -------------------------------------------------
# 1. 環境設定
# -------------------------------------------------
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# -------------------------------------------------
# 2. State 定義
# -------------------------------------------------
class State(TypedDict):
    messages: Annotated[List, add_messages]
    query: str
    result: Optional[str]
    error_count: int  # エラー発生回数をカウント
    last_error: Optional[str]  # 最後に発生したエラーメッセージ
    retry_count: int  # リトライ回数

# -------------------------------------------------
# 3. エラーハンドリング用のユーティリティ関数
# -------------------------------------------------

def log_error(node_name: str, error: Exception, attempt: int = 1):
    """エラーログを記録"""
    error_msg = f"[{node_name}] エラー発生 (試行回数: {attempt}): {type(error).__name__}: {str(error)}"
    print(f"  ⚠️  {error_msg}")
    return error_msg

# -------------------------------------------------
# 4. ノード定義（エラーハンドリング付き）
# -------------------------------------------------

def unreliable_api_call(query: str) -> str:
    """
    不安定な外部API呼び出しをシミュレート
    50%の確率でエラーを発生させる（デモ用）
    """
    # ランダムにエラーを発生させる（デモ用）
    if random.random() < 0.5:
        raise ConnectionError(f"API接続エラー: {query} へのリクエストが失敗しました")
    
    # 正常な場合
    return f"API Response for: {query}"

@retry(
    stop=stop_after_attempt(3),  # 最大3回までリトライ
    wait=wait_exponential(multiplier=1, min=1, max=10),  # 指数バックオフ（1秒、2秒、4秒...）
    retry=retry_if_exception_type((ConnectionError, TimeoutError)),  # これらのエラーのみリトライ
    reraise=True  # 最終的に失敗した場合は例外を再発生
)
def api_node_with_retry(state: State) -> dict:
    """リトライ機能付きのAPI呼び出しノード"""
    print("\n[API Node] 外部APIを呼び出し中...")
    
    query = state.get("query", "")
    retry_count = state.get("retry_count", 0)
    
    try:
        # 不安定なAPI呼び出しをシミュレート
        result = unreliable_api_call(query)
        print(f"  ✅ [API Node] 成功: {result}")
        return {
            "result": result,
            "retry_count": retry_count + 1,
            "error_count": 0  # 成功したのでエラーカウントをリセット
        }
    except (ConnectionError, TimeoutError) as e:
        # tenacityが自動的にリトライするが、ログは記録
        log_error("API Node", e, retry_count + 1)
        raise  # tenacityにリトライを委ねる
    except Exception as e:
        # リトライしないエラー（例: ValueError）
        log_error("API Node", e, retry_count + 1)
        return {
            "result": None,
            "error_count": state.get("error_count", 0) + 1,
            "last_error": str(e),
            "retry_count": retry_count + 1
        }

def llm_node_with_error_handling(state: State) -> dict:
    """エラーハンドリング付きのLLM呼び出しノード"""
    print("\n[LLM Node] LLMを呼び出し中...")
    
    messages = state["messages"]
    error_count = state.get("error_count", 0)
    
    try:
        # LLM呼び出し（タイムアウトやレート制限エラーが発生する可能性がある）
        response = llm.invoke(messages)
        print(f"  ✅ [LLM Node] 成功")
        return {
            "messages": [response],
            "error_count": 0  # 成功したのでエラーカウントをリセット
        }
    except Exception as e:
        # LLM呼び出しのエラー（レート制限、タイムアウトなど）
        error_msg = log_error("LLM Node", e)
        
        # エラー時のフォールバックメッセージ
        fallback_message = AIMessage(
            content="申し訳ございません。一時的にサービスに接続できませんでした。しばらくしてから再度お試しください。"
        )
        
        return {
            "messages": [fallback_message],
            "error_count": error_count + 1,
            "last_error": error_msg
        }

def timeout_simulation_node(state: State) -> dict:
    """タイムアウトをシミュレートするノード"""
    print("\n[Timeout Node] 長時間処理をシミュレート中...")
    
    # ランダムに長時間処理をシミュレート（デモ用）
    processing_time = random.uniform(0.5, 3.0)
    
    if processing_time > 2.0:
        # タイムアウトとみなす
        raise TimeoutError(f"処理がタイムアウトしました（処理時間: {processing_time:.2f}秒）")
    
    time.sleep(processing_time)
    print(f"  ✅ [Timeout Node] 成功（処理時間: {processing_time:.2f}秒）")
    return {"result": f"処理完了（{processing_time:.2f}秒）"}

@retry(
    stop=stop_after_attempt(2),  # 最大2回までリトライ
    wait=wait_exponential(multiplier=0.5, min=0.5, max=2),
    retry=retry_if_exception_type(TimeoutError),
    reraise=True
)
def timeout_node_with_retry(state: State) -> dict:
    """タイムアウトリトライ付きノード"""
    try:
        return timeout_simulation_node(state)
    except TimeoutError as e:
        log_error("Timeout Node", e)
        raise  # tenacityにリトライを委ねる

def error_recovery_node(state: State) -> dict:
    """エラー状態を確認し、必要に応じてリカバリーするノード"""
    print("\n[Recovery Node] エラー状態を確認中...")
    
    error_count = state.get("error_count", 0)
    last_error = state.get("last_error", "")
    result = state.get("result")
    
    if error_count > 0:
        print(f"  ⚠️  [Recovery Node] エラーが検出されました（回数: {error_count}）")
        print(f"  📝 [Recovery Node] 最後のエラー: {last_error}")
        
        # エラーが多すぎる場合は、代替処理を提案
        if error_count >= 3:
            recovery_message = AIMessage(
                content="複数のエラーが発生しました。システム管理者に連絡するか、別の方法をお試しください。"
            )
            return {
                "messages": [recovery_message],
                "result": "エラーが多すぎるため、処理を中断しました"
            }
        else:
            # エラーが少ない場合は、再試行を促す
            recovery_message = AIMessage(
                content="一時的なエラーが発生しましたが、処理は続行します。"
            )
            return {
                "messages": [recovery_message],
                "error_count": 0  # リカバリーしたのでエラーカウントをリセット
            }
    else:
        print(f"  ✅ [Recovery Node] エラーは検出されませんでした")
        return {}

# -------------------------------------------------
# 5. 条件分岐関数
# -------------------------------------------------

def check_error_state(state: State) -> str:
    """エラー状態をチェックして次のノードを決定"""
    error_count = state.get("error_count", 0)
    result = state.get("result")
    
    # エラーが発生している、または結果がない場合はリカバリーノードへ
    if error_count > 0 or result is None:
        return "recovery"
    else:
        return "end"

# -------------------------------------------------
# 6. グラフ構築
# -------------------------------------------------
builder = StateGraph(State)

# ノードの追加
builder.add_node("api_call", api_node_with_retry)
builder.add_node("llm_call", llm_node_with_error_handling)
builder.add_node("timeout_test", timeout_node_with_retry)
builder.add_node("recovery", error_recovery_node)

# エッジの追加
builder.add_edge(START, "api_call")
builder.add_edge("api_call", "llm_call")
builder.add_edge("llm_call", "timeout_test")

# エラー状態に応じた条件分岐
builder.add_conditional_edges(
    "timeout_test",
    check_error_state,
    {
        "recovery": "recovery",
        "end": END
    }
)

builder.add_edge("recovery", END)

graph = builder.compile()

# -------------------------------------------------
# 7. 実行
# -------------------------------------------------
async def main():
    print("--- Error Handling & Recovery Bot 開始 ---\n")
    
    test_cases = [
        {
            "name": "ケース1: 正常な処理",
            "query": "正常なクエリってなんですか？",
            "description": "エラーが発生しない場合の動作を確認"
        },
        {
            "name": "ケース2: APIエラー（リトライ成功）",
            "query": "APIエラーテストってなんですか？",
            "description": "APIエラーが発生するが、リトライで成功する場合"
        },
        {
            "name": "ケース3: タイムアウトエラー",
            "query": "タイムアウトテストってなんですか？",
            "description": "タイムアウトが発生する場合の動作を確認"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"{test_case['name']}")
        print(f"説明: {test_case['description']}")
        print(f"{'='*60}\n")
        
        initial_state = {
            "messages": [HumanMessage(content=test_case["query"])],
            "query": test_case["query"],
            "result": None,
            "error_count": 0,
            "last_error": None,
            "retry_count": 0
        }
        
        try:
            # グラフを実行
            final_state = None
            async for event in graph.astream_events(initial_state, version="v1"):
                if event["event"] == "on_chain_end":
                    name = event.get("name", "")
                    if name in ["llm_call", "recovery"]:
                        output = event["data"]["output"]
                        if "messages" in output and output["messages"]:
                            last_msg = output["messages"][-1]
                            if isinstance(last_msg, AIMessage):
                                print(f"\n[Final Message]\n{last_msg.content}")
            
            print(f"\n[最終状態]")
            print(f"  エラー回数: {initial_state.get('error_count', 0)}")
            print(f"  リトライ回数: {initial_state.get('retry_count', 0)}")
            if initial_state.get("last_error"):
                print(f"  最後のエラー: {initial_state['last_error']}")
        
        except RetryError as e:
            print(f"\n  ❌ 最大リトライ回数に達しました: {e}")
        except Exception as e:
            print(f"\n  ❌ 予期しないエラー: {type(e).__name__}: {e}")
        finally:
            # エラーが発生してもグラフ構造を表示
            print(f"\n[グラフ構造]")
            graph_ascii = graph.get_graph().print_ascii()
            print(graph_ascii)
        
        print("\n" + "-"*60)
        
        # テストケース間で少し待機（APIレート制限対策）
        if i < len(test_cases):
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())

