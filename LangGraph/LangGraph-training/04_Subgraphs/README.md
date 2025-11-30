# 04. Subgraphs（サブグラフ）

このコードは、**「グラフの中に別のグラフを埋め込む（入れ子にする）」** 方法（サブグラフ）を学ぶサンプルです。
大規模なエージェントを作る際、全ての処理を1つのグラフに書くと複雑になりすぎるため、このように機能ごとに分割して管理します。

## 構成

このサンプルでは、以下の2つのチーム（グラフ）が連携します。

1.  **親グラフ（編集部）**: 全体の指揮をとります。
    *   `topic_receiver`: テーマを受け取る。
    *   `writer`: 記事を書く。
2.  **子グラフ（リサーチチーム）**: 特定のタスク（調査）だけを専門に行います。
    *   `researcher`: 情報を集める。
    *   `reviewer`: 内容をチェックする。

## 仕組みのポイント

### 1. 子グラフの作成とコンパイル

```python
research_builder = StateGraph(ResearchState)
# ... ノードやエッジを追加 ...
research_graph = research_builder.compile() # ← これが「部品」になる
```

*   子グラフも、普通のグラフと同じように作ります。
*   `compile()` することで、他のグラフから呼び出せるオブジェクトになります。

### 2. 親グラフへの組み込み

```python
# ★ここで子グラフを「1つのノード」として追加！
main_builder.add_node("research_team", research_graph)
```

*   親グラフの `add_node` に、関数ではなく **コンパイル済みの子グラフ** を渡します。
*   これだけで、親グラフはそのノードに来たとき、自動的に子グラフを起動し、子グラフが終了するまで待ちます。

### 3. データの受け渡し

*   **親 → 子**: 親の `state` がそのまま子グラフの入力（`START`）として渡されます。
*   **子 → 親**: 子グラフの最後の出力（`END`時点の `state`）が、親グラフの次のノードへの入力になります。
    *   ※ 親と子で `State` の定義（スキーマ）に互換性がある必要があります（今回は `messages` という共通のキーを持たせています）。

## コードの詳細解説

このコードは、**「大きな仕事を、専門チーム（子グラフ）に丸投げする」** という構造（サブグラフ）を実現しています。

上から順に、何をしているのか詳しく解説します。

### 1. 子グラフ（リサーチチーム）の作成 (13-50行目)

まず、「調査」だけを専門に行う小さなグラフを作ります。

```python
# 子グラフ用のState
class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    research_summary: str
```

*   子グラフ専用の State を定義します。親グラフとデータのやり取りをするため、共通のキー（`messages`）を持たせています。

```python
def researcher(state: ResearchState):
    # ... 調査して結果を返す ...
def reviewer(state: ResearchState):
    # ... 結果を確認する ...

research_builder = StateGraph(ResearchState)
# ... ノードとエッジを追加 ...
research_graph = research_builder.compile() # ★ここでコンパイル！
```

*   `researcher` → `reviewer` という流れのグラフを作ります。
*   最後に `compile()` することで、このグラフ全体が**「1つの呼び出し可能なオブジェクト（部品）」**になります。

### 2. 親グラフ（編集部）の作成 (53-97行目)

次に、全体の指揮をとる親グラフを作ります。

```python
# 親グラフ用のState
class MainState(TypedDict):
    messages: Annotated[list, add_messages]
    final_article: str
```

*   親グラフ用の State です。

```python
def topic_receiver(state: MainState):
    # ... テーマを受け取る ...

def writer(state: MainState):
    # ... 記事を書く ...
```

*   親グラフ独自のノード（編集長、ライター）を定義します。

### 3. 子グラフの組み込み (86-87行目) ★ここが一番重要

```python
# ★ここで子グラフを「1つのノード」として追加！
main_builder.add_node("research_team", research_graph)
```

*   ここがサブグラフの核心です。
*   `add_node` には通常、関数（`def ...`）を渡しますが、代わりに **コンパイル済みの子グラフ (`research_graph`)** を渡しています。
*   これにより、LangGraph は「あ、ここに来たら別のグラフを起動すればいいんだな」と理解します。

### 4. 全体の流れ (91-94行目)

```python
main_builder.add_edge(START, "topic_receiver")
main_builder.add_edge("topic_receiver", "research_team") # 親 -> 子
main_builder.add_edge("research_team", "writer")         # 子 -> 親
main_builder.add_edge("writer", END)
```

1.  **topic_receiver**: ユーザーから「コーヒーの健康効果」というテーマを受け取ります。
2.  **research_team (子グラフ)**:
    *   ここで処理が子グラフに移ります。
    *   親の State（`messages`）が子グラフに渡されます。
    *   子グラフ内で `researcher`（調査）→ `reviewer`（確認）が実行されます。
    *   子グラフが終了すると、その結果（最後のメッセージ）を持って親グラフに戻ります。
3.  **writer**: 子グラフから戻ってきた調査結果を見て、記事を書きます。

## まとめ

この仕組みを使うメリットは、**「複雑さを閉じ込められること」** です。
親グラフ（編集長）は、「リサーチチーム」の中身がどうなっているか（誰が何回チェックしているかなど）を知る必要がありません。「頼んだら結果が返ってくる」ということだけ知っていればいいのです。
これにより、大規模なエージェント開発でもスパゲッティコードにならずに済みます。
