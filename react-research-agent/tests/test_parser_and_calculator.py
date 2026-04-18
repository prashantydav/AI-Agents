from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from react_research_agent.agent import ReActResearchAgent
from react_research_agent.tools import ToolError, _safe_eval


def test_parse_react_output() -> None:
    text = """Thought: Need a source first.
Action: web_search
Action Input: impact of LFP batteries on grid storage"""
    parsed = ReActResearchAgent.parse_react_output(text)
    assert parsed.thought == "Need a source first."
    assert parsed.action == "web_search"
    assert parsed.action_input == "impact of LFP batteries on grid storage"


def test_safe_eval_ok() -> None:
    assert _safe_eval("2 + 3 * 4") == 14


def test_safe_eval_rejects_calls() -> None:
    try:
        _safe_eval("__import__('os').system('echo hi')")
    except ToolError:
        return
    assert False, "Expected ToolError for disallowed expression"


def test_coerce_chunk_text() -> None:
    assert ReActResearchAgent._coerce_chunk_text("abc") == "abc"
    assert ReActResearchAgent._coerce_chunk_text([{"text": "a"}, {"text": "b"}]) == "ab"
