"""
検索サブグラフ：Tavily で Web 検索し、各結果にドメイン・信頼度 tier を付与する。
(Search subgraph: runs a Tavily web search and tags each result with its domain and trust tier.)
"""

from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Optional
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage
from langchain_tavily import TavilySearch

from utils import log_event, retry, get_domain, get_tier

load_dotenv()


class Evidence(TypedDict):
    """1件の検索結果を表す証拠データ。(A single search result, represented as evidence.)"""
    url: str
    domain: str
    title: str
    snippet: str
    tier: int  # 1=公的機関・通信社, 2=主要メディア, 3=その他 (1=official/wire, 2=mainstream, 3=other)


class SearchState(TypedDict):
    """検索サブグラフの State。(State for the search subgraph.)"""
    question: str
    retry_hint: Optional[str]
    query: str
    raw_results: List[dict]
    evidence: Annotated[List[Evidence], operator.add]
    messages: Annotated[list, add_messages]


tavily_search = TavilySearch(max_results=5)

# 万が一 retry_hint が想定より長くなった場合の保険（本来は verify_subgraph 側で
# 短い検索キーワードとして生成される想定）。
# (Safety cap in case retry_hint ends up longer than expected — it's meant to
# already be a short search keyword phrase, generated as such by verify_subgraph.)
MAX_RETRY_HINT_CHARS = 60
MAX_QUERY_CHARS = 380


def prepare_query(state: SearchState) -> dict:
    """
    質問を検索クエリに変換する。再検索時は前回の verify_subgraph が生成した
    短い search_suggestion（検索用キーワード）をクエリに足し合わせ、
    同じ上位結果ばかりが返らないようクエリを多様化する。
    （以前は自由記述の rationale をそのまま使っていたが、ライブテストで
    実際に検索が脱線したため、短いキーワードを使う設計に変更した。）
    (Turns the question into a search query. On retry, appends the short
    search_suggestion keyword phrase that verify_subgraph generated, so the
    query is diversified rather than returning the same top result.
    Previously this used the free-text rationale directly, but that actually
    derailed a live search run, so the design was changed to use a short
    keyword phrase instead.)

    Args:
        state (SearchState): 検索サブグラフの現在の State (current search subgraph state)

    Returns:
        dict: 更新された `query` (dict with the updated `query`)
    """
    question = state["question"]
    retry_hint = state.get("retry_hint")
    if retry_hint:
        hint = retry_hint.replace("\n", " ").strip()[:MAX_RETRY_HINT_CHARS]
        query = f"{question} {hint}"[:MAX_QUERY_CHARS]
        log_event("prepare_query", f"再検索クエリ (retry query): {query}")
    else:
        query = question
        log_event("prepare_query", f"初回クエリ (initial query): {query}")
    return {"query": query}


@retry(max_attempts=3, backoff=0.5)
async def _call_tavily(query: str) -> dict:
    """
    Tavily 検索を呼び出す（リトライ付き）。
    (Invokes the Tavily search tool, wrapped with retry.)

    Args:
        query (str): 検索クエリ (the search query)

    Returns:
        dict: Tavily のレスポンス（"results" キーを含む） (Tavily's response, containing a "results" key)

    Raises:
        RuntimeError: Tavily がエラーを返した場合。"結果0件"として静かに握りつぶさない
                      （偽陰性——本当は失敗しているのに「根拠なし」と誤認されるのを防ぐ）。
                      (If Tavily returns an error. Never silently swallowed as "0 results" —
                      that would be a false negative, mistaking a real failure for "no evidence".)
    """
    response = await tavily_search.ainvoke({"query": query})
    if isinstance(response, dict) and response.get("error"):
        raise RuntimeError(f"Tavily error: {response['error']}")
    return response


async def call_api(state: SearchState) -> dict:
    """
    検索クエリで Tavily API を呼び出し、生の検索結果を取得するノード。
    (Node that calls the Tavily API with the prepared query and stores the raw results.)

    Args:
        state (SearchState): 検索サブグラフの現在の State (current search subgraph state)

    Returns:
        dict: 更新された `raw_results` (dict with the updated `raw_results`)
    """
    response = await _call_tavily(state["query"])
    raw_results = response.get("results", []) if isinstance(response, dict) else (response or [])
    log_event("call_api", f"{len(raw_results)} 件取得 ({len(raw_results)} results retrieved)")
    return {"raw_results": raw_results}


def tag_and_extract(state: SearchState) -> dict:
    """
    各検索結果にドメインと信頼度 tier を付与する。tier は静的テーブルによる
    決め打ちであり、LLM には一切判断させない。
    (Tags each result with its domain and trust tier, via the static table only —
    never by asking the LLM to judge credibility.)

    Args:
        state (SearchState): 検索サブグラフの現在の State (current search subgraph state)

    Returns:
        dict: 新規に見つかった `evidence` のリスト (list of newly found `evidence`)
    """
    tagged: List[Evidence] = []
    for r in state["raw_results"]:
        url = r.get("url", "")
        domain = get_domain(url)
        tagged.append(Evidence(
            url=url,
            domain=domain,
            title=r.get("title", ""),
            snippet=(r.get("content") or "")[:500],
            tier=get_tier(domain),
        ))
    log_event("tag_and_extract", f"tiers={[e['tier'] for e in tagged]}")
    return {"evidence": tagged}


def output(state: SearchState) -> dict:
    """
    検索結果の要約をメッセージ履歴に追加し、メイングラフへ返すノード。
    (Node that appends a summary of the search results to the message history,
    returning control to the main graph.)

    Args:
        state (SearchState): 検索サブグラフの現在の State (current search subgraph state)

    Returns:
        dict: 更新された `messages` (dict with the updated `messages`)
    """
    new_evidence = state["evidence"]
    summary = "\n".join(f"- [tier{e['tier']}] {e['domain']}: {e['title']}" for e in new_evidence) or "（結果なし / no results）"
    return {"messages": [AIMessage(content=f"[search] 検索結果 (search results):\n{summary}")]}


builder = StateGraph(SearchState)
builder.add_node("prepare_query", prepare_query)
builder.add_node("call_api", call_api)
builder.add_node("tag_and_extract", tag_and_extract)
builder.add_node("output", output)

builder.add_edge(START, "prepare_query")
builder.add_edge("prepare_query", "call_api")
builder.add_edge("call_api", "tag_and_extract")
builder.add_edge("tag_and_extract", "output")
builder.add_edge("output", END)

search_graph = builder.compile()
