"""
ファクトチェッカー用の共通ヘルパー（ログ、リトライ、ドメイン信頼度テーブル）。
(Shared helpers for the fact-checker: logging, retry, and the domain-tiering table.)
"""

import datetime
import functools
import asyncio
from urllib.parse import urlparse


def log_event(name: str, msg: str) -> None:
    """
    タイムスタンプ付きでイベントをコンソールに出力する。
    (Prints an event to the console with a timestamp.)

    Args:
        name (str): イベントの発生元（ノード名など） (source of the event, e.g. a node name)
        msg (str): 表示するメッセージ (the message to display)
    """
    ts = datetime.datetime.now().isoformat()
    print(f"[{ts}] [{name}] {msg}")


def retry(max_attempts: int = 3, backoff: float = 0.5):
    """
    非同期関数を対象に、失敗時に指数的な待機を挟んで再試行させるデコレータ。
    (Decorator that retries an async function on failure, with linear/backoff wait between attempts.)

    Args:
        max_attempts (int): 最大試行回数 (maximum number of attempts)
        backoff (float): 試行ごとの待機時間の基準値（秒） (base backoff time in seconds per attempt)

    Returns:
        Callable: デコレートされた非同期関数 (the decorated async function)
    """
    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    attempts += 1
                    log_event(fn.__name__, f"試行 {attempts}/{max_attempts} 失敗: {exc} (attempt {attempts}/{max_attempts} failed)")
                    if attempts == max_attempts:
                        raise
                    await asyncio.sleep(backoff * attempts)
        return wrapper
    return decorator


# ドメイン信頼度テーブル（静的・決め打ち）。
# LLM に「このソースは信頼できるか」を判断させるのではなく、
# 検索結果のドメインを固定ルールで分類し、複数ドメインでの裏取り（corroboration）を
# 判定するために使う。日本国内向け事業のため、日本の公的機関・主要メディアを
# 国際的な情報源と同格の tier に含める。
#
# (Static, hard-coded domain-tiering table. Rather than asking the LLM to judge
# whether a source is "credible", search-result domains are classified by fixed
# rule and used to check cross-domain corroboration. Since the business operates
# in Japan, Japanese official bodies and mainstream outlets sit at the same tiers
# as their international counterparts, not below them.)
#
# ".gov" 系のパターンは get_tier() 内で "gov" ラベル一致として汎用的に処理する
# （gov.uk のような apex ドメインと whitehouse.gov のような suffix の両方を拾うため）。
# ここでは "gov" ラベル判定でカバーできない suffix のみを列挙する。
# (".gov"-style patterns are handled generically in get_tier() via an exact "gov"
# label match, covering both apex domains like gov.uk and suffixed domains like
# whitehouse.gov. Only list suffixes here that aren't covered by that label check.)
TIER_1_SUFFIXES = (
    ".go.jp", ".edu", ".ac.jp",
)
TIER_1_DOMAINS = {
    "who.int", "un.org",
    "reuters.com", "apnews.com",
    "kyodonews.net",   # 共同通信
    "jiji.com",         # 時事通信
    "nhk.or.jp",        # NHK
    "canada.ca",        # カナダ政府公式ポータル（.gov 系 suffix に一致しないため個別追加）
                         # (Canada's official government portal — doesn't match a .gov-style suffix)
}
TIER_2_DOMAINS = {
    "nytimes.com", "bbc.com",
    "asahi.com",        # 朝日新聞
    "yomiuri.co.jp",    # 読売新聞
    "nikkei.com",       # 日本経済新聞
    "mainichi.jp",       # 毎日新聞
    "britannica.com",   # 編集管理された百科事典（Wikipedia とは異なり誰でも編集できるわけではない）
                         # (editorially-controlled encyclopedia — unlike Wikipedia, not open-edit)
}


def get_domain(url: str) -> str:
    """
    URL からドメイン部分（例: www. を除いたホスト名）を抽出する。
    (Extracts the domain (hostname, minus a leading "www.") from a URL.)

    Args:
        url (str): 検索結果の URL (a search result's URL)

    Returns:
        str: ドメイン名 (the domain name)
    """
    host = urlparse(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def get_tier(domain: str) -> int:
    """
    ドメインを静的テーブルに基づいて信頼度 tier（1〜3）に分類する。
    LLM の判断は一切使わない。
    (Classifies a domain into a trust tier (1-3) using the static table only —
    no LLM judgment involved.)

    Args:
        domain (str): `get_domain` で抽出したドメイン (domain from `get_domain`)

    Returns:
        int: 1 = 公的機関・通信社 (official/wire), 2 = 主要メディア (mainstream news),
             3 = その他（Wikipedia を含む） (everything else, incl. Wikipedia)
    """
    # "gov" がドット区切りのラベルとして単独で出現するかで判定する。
    # ".gov" で終わるか（whitehouse.gov）だけでなく、"gov.uk" のように "gov" 自体が
    # トップレベルの直下ラベルであるケースも同じロジックで拾える
    # （実際に "gov.uk" が ".gov." 部分文字列マッチでは拾えず、ライブテストで判明した）。
    # "govtech.com" のような誤検出は起きない（ラベル全体が "gov" と一致する場合のみ）。
    # (Checks whether "gov" appears as its own dot-separated label, which catches both
    # ".gov"-suffixed domains (whitehouse.gov) and cases where "gov" itself is the
    # apex label (gov.uk) — the latter was missed by a ".gov." substring check in live
    # testing. Avoids false positives like "govtech.com", since it requires an exact
    # label match, not a substring match.)
    if (
        domain in TIER_1_DOMAINS
        or any(domain.endswith(suffix) for suffix in TIER_1_SUFFIXES)
        or "gov" in domain.split(".")
    ):
        return 1
    if domain in TIER_2_DOMAINS:
        return 2
    return 3
