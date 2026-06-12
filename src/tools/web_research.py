"""Small, cached web research helpers for Verdict Room agents."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, TypedDict

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

CACHE_DIR = Path("data/cache")
DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_RESULTS = 5
MAX_PAGE_CHARS = 12_000
USER_AGENT = "VerdictRoomResearch/0.1 (+https://github.com/verdict-room)"
REDDIT_SEARCH_URL = "https://www.reddit.com/search.json"


class SearchResult(TypedDict):
    """Normalized result returned by :func:`web_search`."""

    title: str
    url: str
    snippet: str


class PageResult(TypedDict):
    """Structured page content returned by :func:`fetch_page`."""

    url: str
    title: str
    text: str
    status_code: int | None
    error: str | None


class RedditResult(TypedDict):
    """Normalized Reddit post returned by :func:`reddit_search`."""

    title: str
    url: str
    subreddit: str
    author: str
    score: int
    num_comments: int
    created_utc: float
    snippet: str


def _cache_path(operation: str, parameters: dict[str, object]) -> Path:
    payload = json.dumps(
        {"operation": operation, "parameters": parameters},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def _read_cache(operation: str, parameters: dict[str, object]) -> Any | None:
    path = _cache_path(operation, parameters)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(operation: str, parameters: dict[str, object], value: object) -> None:
    path = _cache_path(operation, parameters)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    except OSError:
        # Research should still work when the cache directory is not writable.
        return


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _ddgs_text(query: str, max_results: int, timeout: float) -> list[dict[str, Any]]:
    with DDGS(timeout=max(1, int(timeout))) as client:
        return client.text(query, max_results=max_results)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def _http_get(
    url: str,
    *,
    timeout: float,
    headers: dict[str, str],
    params: dict[str, str | int] | None = None,
) -> httpx.Response:
    response = httpx.get(
        url,
        params=params,
        headers=headers,
        timeout=timeout,
        follow_redirects=True,
    )
    response.raise_for_status()
    return response


def web_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[SearchResult]:
    """Search the web with DDGS and return normalized JSON-compatible results."""

    cleaned_query = query.strip()
    if not cleaned_query or max_results <= 0:
        return []

    parameters: dict[str, object] = {
        "query": cleaned_query,
        "max_results": max_results,
    }
    cached = _read_cache("web_search", parameters)
    if isinstance(cached, list):
        return cached

    try:
        raw_results = _ddgs_text(cleaned_query, max_results, timeout)
    except Exception:
        return []

    results: list[SearchResult] = []
    for item in raw_results[:max_results]:
        if not isinstance(item, dict):
            continue
        url = str(item.get("href") or item.get("url") or "").strip()
        if not url:
            continue
        results.append(
            {
                "title": str(item.get("title") or "").strip(),
                "url": url,
                "snippet": str(item.get("body") or item.get("snippet") or "").strip(),
            }
        )

    _write_cache("web_search", parameters, results)
    return results


def fetch_page(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_chars: int = MAX_PAGE_CHARS,
) -> PageResult:
    """Fetch a page and extract its title and readable text."""

    cleaned_url = url.strip()
    empty_result: PageResult = {
        "url": cleaned_url,
        "title": "",
        "text": "",
        "status_code": None,
        "error": None,
    }
    if not cleaned_url:
        return {**empty_result, "error": "URL must not be empty"}
    if max_chars <= 0:
        return {**empty_result, "error": "max_chars must be greater than zero"}

    parameters: dict[str, object] = {"url": cleaned_url, "max_chars": max_chars}
    cached = _read_cache("fetch_page", parameters)
    if isinstance(cached, dict):
        return cached

    try:
        response = _http_get(
            cleaned_url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "noscript", "template"]):
            element.decompose()

        title = soup.title.get_text(" ", strip=True) if soup.title else ""
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
        result: PageResult = {
            "url": str(response.url),
            "title": title,
            "text": text[:max_chars],
            "status_code": response.status_code,
            "error": None,
        }
    except Exception as exc:
        return {**empty_result, "error": f"{type(exc).__name__}: {exc}"}

    _write_cache("fetch_page", parameters, result)
    return result


def reddit_search(
    query: str,
    limit: int = DEFAULT_MAX_RESULTS,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> list[RedditResult]:
    """Search public Reddit posts through Reddit's JSON endpoint."""

    cleaned_query = query.strip()
    if not cleaned_query or limit <= 0:
        return []

    parameters: dict[str, object] = {"query": cleaned_query, "limit": limit}
    cached = _read_cache("reddit_search", parameters)
    if isinstance(cached, list):
        return cached

    try:
        response = _http_get(
            REDDIT_SEARCH_URL,
            timeout=timeout,
            headers={
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
            params={
                "q": cleaned_query,
                "limit": limit,
                "sort": "relevance",
                "raw_json": 1,
            },
        )
        payload = response.json()
        payload_data = payload.get("data", {}) if isinstance(payload, dict) else {}
        children = (
            payload_data.get("children", []) if isinstance(payload_data, dict) else []
        )
        if not isinstance(children, list):
            children = []
    except Exception:
        return []

    results: list[RedditResult] = []
    for child in children[:limit]:
        child_data = child.get("data", {}) if isinstance(child, dict) else {}
        data = child_data if isinstance(child_data, dict) else {}
        permalink = str(data.get("permalink") or "")
        url = (
            f"https://www.reddit.com{permalink}"
            if permalink.startswith("/")
            else str(data.get("url") or permalink)
        )
        results.append(
            {
                "title": str(data.get("title") or "").strip(),
                "url": url,
                "subreddit": str(data.get("subreddit_name_prefixed") or ""),
                "author": str(data.get("author") or ""),
                "score": _as_int(data.get("score")),
                "num_comments": _as_int(data.get("num_comments")),
                "created_utc": _as_float(data.get("created_utc")),
                "snippet": str(data.get("selftext") or "").strip()[:500],
            }
        )

    _write_cache("reddit_search", parameters, results)
    return results


def main(argv: list[str] | None = None) -> int:
    """Run a web search from the command line and print JSON."""

    parser = argparse.ArgumentParser(description="Search the web with DDGS.")
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--max-results",
        type=int,
        default=DEFAULT_MAX_RESULTS,
        help=f"maximum results to return (default: {DEFAULT_MAX_RESULTS})",
    )
    args = parser.parse_args(argv)
    output = {
        "query": args.query,
        "results": web_search(args.query, max_results=args.max_results),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
