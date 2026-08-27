"""
MCP 联网搜索服务。支持多种后端：

- duckduckgo: 免费，无需 API key（默认）
- brave: Brave Search API，需要 API key

降级链 Level 2 使用此服务搜索外部信息。
"""
from __future__ import annotations

import logging
import urllib.parse
from dataclasses import dataclass, field

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    title: str
    url: str
    snippet: str


def search_web(query: str, max_results: int | None = None) -> list[dict]:
    """联网搜索主入口。根据配置选择后端。

    后端优先级：配置的 provider → Bing（国内可访问）→ DuckDuckGo（需代理）

    Returns:
        [{"title": str, "url": str, "snippet": str}, ...]
    """
    if not settings.web_search_enabled:
        logger.info("Web search disabled by config")
        return []

    max_results = max_results or settings.web_search_max_results
    provider = settings.web_search_provider

    if provider == "brave":
        results = _search_brave(query, max_results)
        if results:
            return results
        # Brave 失败 → 尝试 Bing
        logger.info("Brave failed, falling back to Bing")
        bing_results = _search_bing(query, max_results)
        if bing_results:
            return bing_results
        # Bing 也失败 → 尝试 DuckDuckGo
        logger.info("Bing also failed, falling back to DuckDuckGo")
        return _search_duckduckgo(query, max_results)

    elif provider == "bing":
        results = _search_bing(query, max_results)
        if results:
            return results
        # Bing 失败 → 尝试 DuckDuckGo
        logger.info("Bing failed, falling back to DuckDuckGo")
        return _search_duckduckgo(query, max_results)

    else:
        # DuckDuckGo（国内通常不可达）→ 优先尝试 Bing
        results = _search_duckduckgo(query, max_results)
        if results:
            return results
        # DuckDuckGo 超时/失败 → 尝试 Bing（国内可访问）
        logger.info("DuckDuckGo unavailable, falling back to Bing")
        bing_results = _search_bing(query, max_results)
        if bing_results:
            return bing_results
        return []


def _search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    """DuckDuckGo Instant Answer API（免费，无需 API key）。

    使用 DuckDuckGo 的 HTML 搜索页面抓取。免费且稳定。
    """
    try:
        url = "https://html.duckduckgo.com/html/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        resp = requests.post(
            url,
            data={"q": query, "kl": "cn-zh"},
            headers=headers,
            timeout=5,  # 国内 DuckDuckGo 通常不可达，5s 快速失败
        )
        resp.raise_for_status()

        # 简单解析 HTML（避免依赖 bs4）
        results = []
        from html.parser import HTMLParser

        class DDHtmlParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.results = []
                self.current = {}
                self.in_result = False
                self.in_link = False
                self.in_snippet = False
                self.tag_stack = []

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                cls = attrs.get("class", "")
                if tag == "div" and "result" in cls:
                    self.in_result = True
                    self.current = {}
                if self.in_result and tag == "a" and "snippet" not in self.current:
                    self.in_link = True
                    self.current["url"] = attrs.get("href", "")
                if self.in_result and tag == "a" and "snippet" in str(attrs.get("class", "")):
                    self.in_snippet = True

            def handle_data(self, data):
                if self.in_link:
                    self.current["title"] = (self.current.get("title", "") + data).strip()
                if self.in_snippet:
                    self.current["snippet"] = (self.current.get("snippet", "") + data).strip()

            def handle_endtag(self, tag):
                if tag == "a":
                    self.in_link = False
                    self.in_snippet = False
                if tag == "div" and self.in_result:
                    self.in_result = False
                    if self.current.get("title") and self.current.get("url"):
                        self.results.append(dict(self.current))

        parser = DDHtmlParser()
        parser.feed(resp.text)

        for r in parser.results[:max_results]:
            # 清理 DuckDuckGo 的 URL 重定向
            url = r.get("url", "")
            if "//duckduckgo.com/l/?" in url:
                parsed = urllib.parse.urlparse(url)
                qs = urllib.parse.parse_qs(parsed.query)
                url = qs.get("uddg", [url])[0]
                r["url"] = url

        logger.info("DuckDuckGo search | q=%.40s → %d results", query, len(parser.results))
        return parser.results[:max_results]

    except Exception:
        logger.warning("DuckDuckGo search failed (expected if network restricted)")
        return []


def _search_brave(query: str, max_results: int = 5) -> list[dict]:
    """Brave Search API（需要 API key）。"""
    api_key = settings.brave_search_api_key
    if not api_key:
        logger.warning("Brave Search API key not configured")
        return []

    try:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": min(max_results, 10)},
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for r in data.get("web", {}).get("results", [])[:max_results]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("description", ""),
            })

        logger.info("Brave search | q=%.40s → %d results", query, len(results))
        return results

    except Exception:
        logger.exception("Brave Search failed")
        return []


def _search_bing(query: str, max_results: int = 5) -> list[dict]:
    """Bing Web Search API v7（微软，中国可访问，每月 1000 次免费）。

    需要 API key，在 Azure Portal 创建 Bing Search 资源获取。
    https://portal.azure.com → 创建 Bing Search v7 资源 → Keys and Endpoint
    """
    api_key = settings.bing_search_api_key
    if not api_key:
        logger.warning("Bing Search API key not configured")
        return []

    try:
        resp = requests.get(
            "https://api.bing.microsoft.com/v7.0/search",
            params={"q": query, "count": min(max_results, 10), "mkt": "zh-CN"},
            headers={"Ocp-Apim-Subscription-Key": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for r in data.get("webPages", {}).get("value", [])[:max_results]:
            results.append({
                "title": r.get("name", ""),
                "url": r.get("url", ""),
                "snippet": r.get("snippet", ""),
            })

        logger.info("Bing search | q=%.40s → %d results", query, len(results))
        return results

    except Exception:
        logger.exception("Bing Search failed")
        return []
