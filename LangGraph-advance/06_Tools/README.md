# ToolNode & ToolCalling (ツール利用)

LangGraph には、エージェントが自律的にツール（関数）を呼び出すための便利な機能が組み込まれています。
これを使うと、自分で「ツールを呼び出す条件分岐」や「ツールの実行処理」を書く手間が大幅に省けます。

## 学ぶこと

1.  **`@tool` デコレータ**: Python 関数を LangChain ツールとして定義する方法
2.  **`bind_tools`**: LLM にツールを認識させる方法
3.  **`ToolNode`**: ツール実行を自動化する LangGraph の組み込みノード
4.  **`tools_condition`**: 「ツールを使うべきか、回答を返すべきか」を自動判定する条件分岐

## コードの解説

### 1. ツール定義とバインディング

```python
@tool
def multiply(a: int, b: int) -> int:
    """2つの整数を掛け算します。"""
    return a * b

# ...

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)
```

*   `@tool` をつけるだけで、関数の docstring（説明文）や型ヒントから自動的にツールの定義情報（JSON Schema）が生成されます。
*   `bind_tools` で LLM にその定義を渡すと、LLM は必要に応じて「このツールを、この引数で呼び出したい」というレスポンス（`tool_calls`）を返すようになります。

### 2. ToolNode（ツール実行ノード）

```python
from langgraph.prebuilt import ToolNode

tool_node = ToolNode(tools)
```

*   `ToolNode` は、直前のメッセージに含まれる `tool_calls` を読み取り、**対応する関数を自動で実行**し、結果を `ToolMessage` として返してくれる便利なノードです。
*   これがないと、自分で `if "tool_calls" in msg:` のような分岐を書いて関数を実行する処理を書かなければなりません。

### 3. tools_condition（条件分岐）

```python
from langgraph.prebuilt import tools_condition

builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
```

*   `tools_condition` は、LLM のレスポンスを見て次のように振り分けてくれる関数です：
    *   **ツール呼び出しがある場合** → `"tools"` ノードへ（`ToolNode` が動く）
    *   **ない場合（普通の回答）** → `END` へ（終了）
*   `add_conditional_edges` の第3引数（遷移先マップ）を省略すると、デフォルトで `{"tools": "tools", "__end__": END}` のように解釈されます（ノード名が `"tools"` である必要があります）。



実行すると、以下の3パターンの挙動が確認できます。

1.  **普通の会話**: ツールを使わずに返答します。
2.  **計算**: `multiply` ツールを呼び出して計算結果を返します。
3.  **天気**: `get_weather` ツールを呼び出して（ダミーの）天気を返します。

---

## 備考: より汎用的な計算機の実装について

今回の `multiply` ツールは「2つの整数の掛け算」しかできませんが、実務では「どんな複雑な計算もできる」汎用的な計算機が欲しくなることがあります。

その場合、**Python REPL (Read-Eval-Print Loop)** をツールとして LLM に渡すアプローチが一般的です。
LangChain には `PythonREPL` というユーティリティがあり、これを使うと LLM が自分で Python コードを書いて実行し、その結果を使って回答できるようになります。

```python
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool

python_repl = PythonREPL()

@tool
def python_interpreter(code: str) -> str:
    """Python コードを実行して結果を返します。
    複雑な計算やデータ処理が必要な場合に使ってください。
    print() で出力した内容が結果として返されます。
    """
    try:
        return python_repl.run(code)
    except Exception as e:
        return f"Error: {e}"
```

これを使えば、「100番目のフィボナッチ数を計算して」や「このリストの平均値を求めて」といった高度な指示にも対応できるようになります。
※ ただし、任意のコードが実行できるため、セキュリティには十分な注意が必要です（サンドボックス環境での実行を推奨）。
