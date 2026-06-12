from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tools import web_research


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(web_research, "CACHE_DIR", cache_dir)
    return cache_dir


def test_web_search_normalizes_and_caches(
    monkeypatch: pytest.MonkeyPatch, isolated_cache: Path
) -> None:
    calls = 0

    def fake_search(
        query: str, max_results: int, timeout: float
    ) -> list[dict[str, str]]:
        nonlocal calls
        calls += 1
        assert query == "Notion pricing"
        assert max_results == 2
        assert timeout == 4
        return [
            {"title": "Plans", "href": "https://example.com", "body": "Pricing"},
            {"title": "No URL", "body": "Skipped"},
        ]

    monkeypatch.setattr(web_research, "_ddgs_text", fake_search)

    expected = [
        {
            "title": "Plans",
            "url": "https://example.com",
            "snippet": "Pricing",
        }
    ]
    assert web_research.web_search(" Notion pricing ", 2, timeout=4) == expected
    assert web_research.web_search("Notion pricing", 2, timeout=99) == expected
    assert calls == 1

    cache_files = list(isolated_cache.glob("*.json"))
    assert len(cache_files) == 1
    assert len(cache_files[0].stem) == 64
    assert json.loads(cache_files[0].read_text(encoding="utf-8")) == expected


def test_web_search_gracefully_handles_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> list[dict[str, Any]]:
        raise RuntimeError("search unavailable")

    monkeypatch.setattr(web_research, "_ddgs_text", fail)
    assert web_research.web_search("query") == []


def test_ddgs_text_uses_bounded_backends_without_retrying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, float]] = []

    class FakeDDGS:
        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> FakeDDGS:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def text(
            self, query: str, *, max_results: int, backend: str
        ) -> list[dict[str, str]]:
            calls.append((backend, self.timeout))
            assert query == "Notion pricing"
            assert max_results == 3
            if backend == "brave":
                raise RuntimeError("brave unavailable")
            return [{"title": "Plans", "href": "https://example.com"}]

    monkeypatch.setattr(web_research, "DDGS", FakeDDGS)

    assert web_research._ddgs_text("Notion pricing", 3, timeout=20) == [
        {"title": "Plans", "href": "https://example.com"}
    ]
    assert [backend for backend, _ in calls] == ["brave", "mojeek"]
    assert sum(timeout for _, timeout in calls) <= web_research.MAX_SEARCH_TIMEOUT


def test_ddgs_text_supports_older_api_without_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str | None] = []

    class LegacyDDGS:
        def __init__(self, *, timeout: float) -> None:
            assert timeout <= web_research.MAX_SEARCH_TIMEOUT

        def __enter__(self) -> LegacyDDGS:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def text(
            self,
            query: str,
            *,
            max_results: int,
            **kwargs: str,
        ) -> list[dict[str, str]]:
            backend = kwargs.get("backend")
            calls.append(backend)
            if backend is not None:
                raise TypeError("unexpected keyword argument 'backend'")
            return [{"title": query, "href": f"https://example.com/{max_results}"}]

    monkeypatch.setattr(web_research, "DDGS", LegacyDDGS)

    assert web_research._ddgs_text("legacy", 2, timeout=3) == [
        {"title": "legacy", "href": "https://example.com/2"}
    ]
    assert calls == ["brave", None]


def test_web_search_caps_results_for_agent_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_limits: list[int] = []

    def fake_search(
        query: str, max_results: int, timeout: float
    ) -> list[dict[str, str]]:
        requested_limits.append(max_results)
        return [
            {
                "title": f"Result {index}",
                "href": f"https://example.com/{index}",
            }
            for index in range(20)
        ]

    monkeypatch.setattr(web_research, "_ddgs_text", fake_search)

    results = web_research.web_search("query", max_results=100)
    assert requested_limits == [web_research.MAX_SEARCH_RESULTS]
    assert len(results) == web_research.MAX_SEARCH_RESULTS


def test_fetch_page_extracts_text_and_uses_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    request = httpx.Request("GET", "https://example.com/final")
    response = httpx.Response(
        200,
        request=request,
        html="<html><head><title> Example </title><style>x</style></head>"
        "<body><h1>Hello</h1><script>bad()</script><p>World</p></body></html>",
    )

    def fake_get(
        url: str,
        *,
        timeout: float,
        headers: dict[str, str],
        params: dict[str, str | int] | None = None,
    ) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert url == "https://example.com"
        assert headers["User-Agent"] == web_research.USER_AGENT
        assert params is None
        return response

    monkeypatch.setattr(web_research, "_http_get", fake_get)

    result = web_research.fetch_page("https://example.com", max_chars=15)
    assert result == {
        "url": "https://example.com/final",
        "title": "Example",
        "text": "Example Hello W",
        "status_code": 200,
        "error": None,
    }
    assert web_research.fetch_page("https://example.com", max_chars=15) == result
    assert calls == 1


def test_fetch_page_returns_structured_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(web_research, "_http_get", fail)
    result = web_research.fetch_page("https://example.com")

    assert result["url"] == "https://example.com"
    assert result["status_code"] is None
    assert result["text"] == ""
    assert result["error"] == "TimeoutException: timed out"


def test_reddit_search_sets_user_agent_and_normalizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("GET", web_research.REDDIT_SEARCH_URL)
    response = httpx.Response(
        200,
        request=request,
        json={
            "data": {
                "children": [
                    {
                        "data": {
                            "title": "A post",
                            "permalink": "/r/SaaS/comments/abc/a_post/",
                            "subreddit_name_prefixed": "r/SaaS",
                            "author": "tester",
                            "score": 42,
                            "num_comments": 7,
                            "created_utc": 123.5,
                            "selftext": "Useful details",
                        }
                    }
                ]
            }
        },
    )

    def fake_get(
        url: str,
        *,
        timeout: float,
        headers: dict[str, str],
        params: dict[str, str | int] | None = None,
    ) -> httpx.Response:
        assert url == web_research.REDDIT_SEARCH_URL
        assert "VerdictRoomResearch" in headers["User-Agent"]
        assert params == {
            "q": "Notion reviews",
            "limit": 3,
            "sort": "relevance",
            "raw_json": 1,
        }
        return response

    monkeypatch.setattr(web_research, "_http_get", fake_get)
    assert web_research.reddit_search("Notion reviews", limit=3) == [
        {
            "title": "A post",
            "url": "https://www.reddit.com/r/SaaS/comments/abc/a_post/",
            "subreddit": "r/SaaS",
            "author": "tester",
            "score": 42,
            "num_comments": 7,
            "created_utc": 123.5,
            "snippet": "Useful details",
        }
    ]


def test_reddit_search_tolerates_malformed_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("GET", web_research.REDDIT_SEARCH_URL)
    response = httpx.Response(
        200,
        request=request,
        json={
            "data": {
                "children": [
                    {
                        "data": {
                            "title": "Partial post",
                            "url": "https://reddit.com/example",
                            "score": "unknown",
                            "created_utc": None,
                        }
                    }
                ]
            }
        },
    )
    monkeypatch.setattr(web_research, "_http_get", lambda *args, **kwargs: response)

    result = web_research.reddit_search("partial")
    assert result[0]["score"] == 0
    assert result[0]["created_utc"] == 0.0


def test_http_get_retries_three_times(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fail_get(*args: object, **kwargs: object) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("offline")

    monkeypatch.setattr(web_research.httpx, "get", fail_get)
    monkeypatch.setattr(web_research._http_get.retry, "sleep", lambda _: None)

    with pytest.raises(httpx.ConnectError):
        web_research._http_get(
            "https://example.com",
            timeout=1,
            headers={"User-Agent": "test"},
        )
    assert calls == 3


def test_main_prints_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        web_research,
        "web_search",
        lambda query, max_results: [
            {"title": query, "url": "https://example.com", "snippet": str(max_results)}
        ],
    )

    assert web_research.main(["Notion pricing", "--max-results", "1"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "query": "Notion pricing",
        "results": [
            {
                "title": "Notion pricing",
                "url": "https://example.com",
                "snippet": "1",
            }
        ],
    }
