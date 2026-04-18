from __future__ import annotations

import ast
import json
from typing import Callable, Dict, Optional

import requests
import wikipedia
from bs4 import BeautifulSoup
from tavily import TavilyClient

from .config import Settings
from .models import ResearchMemory


class ToolError(RuntimeError):
    """Raised when a tool execution fails."""


def _safe_eval(expr: str) -> float:
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.FloorDiv,
    )

    node = ast.parse(expr, mode="eval")
    for subnode in ast.walk(node):
        if not isinstance(subnode, allowed_nodes):
            raise ToolError("Calculator supports only arithmetic expressions.")

    return eval(compile(node, "<calc>", "eval"), {"__builtins__": {}}, {})


class Toolset:
    def __init__(self, memory: ResearchMemory, settings: Settings) -> None:
        self.memory = memory
        self.settings = settings
        self._tavily_client: Optional[TavilyClient] = None

    def get_tools(self) -> Dict[str, Callable[[str], str]]:
        return {
            "web_search": self.web_search,
            "wikipedia": self.wikipedia_lookup,
            "url_reader": self.url_reader,
            "calculator": self.calculator,
            "note_taker": self.note_taker,
        }

    def _get_tavily_client(self) -> TavilyClient:
        if self._tavily_client is None:
            self._tavily_client = TavilyClient()
        return self._tavily_client

    def web_search(self, query: str) -> str:
        if not query.strip():
            raise ToolError("web_search requires a query string.")

        client = self._get_tavily_client()
        results = client.search(
            query=query,
            max_results=self.settings.max_search_results,
            search_depth="advanced",
            include_answer=False,
            include_raw_content=False,
        )
        rows = results.get("results", [])
        if not rows:
            raise ToolError("No search results returned.")

        formatted = []
        for i, row in enumerate(rows, start=1):
            title = row.get("title", "Untitled")
            url = row.get("url", "")
            content = (row.get("content") or "").strip()
            if url:
                citation_id = self.memory.add_source(url)
            else:
                citation_id = 0
            snippet = content[:350].replace("\n", " ")
            citation_text = f" [source {citation_id}]" if citation_id else ""
            formatted.append(f"{i}. {title}{citation_text}\nURL: {url}\nSnippet: {snippet}")

        return "\n\n".join(formatted)

    def wikipedia_lookup(self, query: str) -> str:
        if not query.strip():
            raise ToolError("wikipedia requires a query string.")

        matches = wikipedia.search(query, results=3)
        if not matches:
            raise ToolError("No matching Wikipedia pages found.")

        page = wikipedia.page(matches[0], auto_suggest=False)
        citation_id = self.memory.add_source(page.url)
        summary = page.summary.replace("\n", " ")
        return f"Title: {page.title} [source {citation_id}]\nURL: {page.url}\nSummary: {summary[:1200]}"

    def url_reader(self, url: str) -> str:
        url = url.strip()
        if not url.startswith("http"):
            raise ToolError("url_reader requires a valid URL starting with http/https.")

        response = requests.get(url, timeout=self.settings.request_timeout_s)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        title = (soup.title.string or "Untitled") if soup.title else "Untitled"
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(paragraphs)
        text = " ".join(text.split())

        citation_id = self.memory.add_source(url)
        snippet = text[:2200] if text else "No paragraph text extracted."
        return f"Title: {title} [source {citation_id}]\nURL: {url}\nContent: {snippet}"

    def calculator(self, expression: str) -> str:
        expression = expression.strip()
        if not expression:
            raise ToolError("calculator requires an arithmetic expression.")

        value = _safe_eval(expression)
        return f"Result: {value}"

    def note_taker(self, raw_input: str) -> str:
        raw_input = raw_input.strip()
        if not raw_input:
            raise ToolError("note_taker requires JSON input or plain text.")

        claim = ""
        evidence = ""
        source = ""

        try:
            payload = json.loads(raw_input)
            claim = str(payload.get("claim", ""))
            evidence = str(payload.get("evidence", ""))
            source = str(payload.get("source", ""))
        except json.JSONDecodeError:
            claim = raw_input
            evidence = raw_input

        if not claim:
            claim = evidence or "Unspecified claim"

        note = self.memory.add_note(claim=claim, evidence=evidence or claim, source=source)
        source_hint = f" source={source}" if source else ""
        return f"Saved note #{note.note_id}:{source_hint} claim='{note.claim[:120]}'"
