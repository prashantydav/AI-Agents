from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .config import Settings
from .logging_utils import JsonlStepLogger
from .models import ResearchMemory, ResearchStep
from .tools import Toolset

try:
    from langsmith import traceable
except Exception:  # pragma: no cover
    def traceable(*args, **kwargs):
        def deco(func):
            return func

        if args and callable(args[0]):
            return args[0]
        return deco


@dataclass
class ParsedAction:
    thought: str
    action: str
    action_input: str


class ReActResearchAgent:
    VALID_ACTIONS = {"web_search", "wikipedia", "url_reader", "calculator", "note_taker", "finish"}

    def __init__(
        self,
        model_name: Optional[str] = None,
        max_steps: Optional[int] = None,
        log_file: str = "logs/steps.jsonl",
    ) -> None:
        self.settings = Settings()
        self.max_steps = max_steps or self.settings.max_steps
        self.memory = ResearchMemory()
        self.logger = JsonlStepLogger(log_file)
        self.toolset = Toolset(memory=self.memory, settings=self.settings)
        self.tools = self.toolset.get_tools()
        self.llm = ChatOpenAI(model=model_name or self.settings.openai_model, temperature=0)
        self.streaming_llm = ChatOpenAI(
            model=model_name or self.settings.openai_model,
            temperature=0,
            streaming=True,
        )

    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "You are a ReAct research agent. Solve complex questions by iterating in steps.\n"
            "At every step, respond with exactly this format:\n"
            "Thought: <reasoning about next best step>\n"
            "Action: <one of web_search|wikipedia|url_reader|calculator|note_taker|finish>\n"
            "Action Input: <single line tool input>\n"
            "Rules:\n"
            "- Use tools to gather evidence before finishing.\n"
            "- Use note_taker to store important claims with evidence and source when possible.\n"
            "- If enough evidence is collected, use Action: finish with a short reason in Action Input.\n"
            "- Do not invent sources.\n"
        )

    def _build_user_prompt(self, question: str) -> str:
        step_lines: List[str] = []
        for s in self.memory.steps:
            step_lines.append(
                f"Step {s.step}\nThought: {s.thought}\nAction: {s.action}\n"
                f"Action Input: {s.action_input}\nObservation: {s.observation}\n"
            )

        notes_lines = [
            f"#{n.note_id} claim={n.claim} | evidence={n.evidence} | source={n.source or 'n/a'}"
            for n in self.memory.notes
        ]

        return (
            f"Question: {question}\n\n"
            f"Previous steps:\n{chr(10).join(step_lines) if step_lines else 'None'}\n\n"
            f"Notes:\n{chr(10).join(notes_lines) if notes_lines else 'None'}\n\n"
            f"Take the next best action."
        )

    @staticmethod
    def parse_react_output(text: str) -> ParsedAction:
        thought_match = re.search(r"Thought:\s*(.*)", text)
        action_match = re.search(r"Action:\s*([a-zA-Z_]+)", text)
        input_match = re.search(r"Action Input:\s*([\s\S]*)", text)

        if not action_match or not input_match:
            raise ValueError("Could not parse Action / Action Input from model output.")

        thought = thought_match.group(1).strip() if thought_match else ""
        action = action_match.group(1).strip()
        action_input = input_match.group(1).strip()
        return ParsedAction(thought=thought, action=action, action_input=action_input)

    @traceable(name="react_llm_step")
    def _invoke_llm_step(self, question: str) -> str:
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(question)
        response = self.llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        if isinstance(response.content, list):
            return "\n".join(str(item) for item in response.content)
        return str(response.content)

    @traceable(name="tool_run")
    def _run_tool_once(self, action: str, action_input: str) -> str:
        tool = self.tools[action]
        return tool(action_input)

    def _run_tool_with_retry_and_fallback(self, action: str, action_input: str) -> str:
        attempts = self.settings.max_tool_retries + 1
        last_error: Optional[Exception] = None

        for _ in range(attempts):
            try:
                return self._run_tool_once(action=action, action_input=action_input)
            except Exception as exc:
                last_error = exc
                time.sleep(0.5)

        fallback_map = {
            "web_search": "wikipedia",
            "wikipedia": "web_search",
            "url_reader": "web_search",
        }
        fallback_tool = fallback_map.get(action)
        if fallback_tool and fallback_tool in self.tools:
            try:
                fallback_observation = self._run_tool_once(action=fallback_tool, action_input=action_input)
                return (
                    f"Primary tool '{action}' failed after retries ({type(last_error).__name__}: {last_error}). "
                    f"Fallback '{fallback_tool}' succeeded.\n{fallback_observation}"
                )
            except Exception as fallback_error:
                return (
                    f"ToolError: '{action}' failed ({type(last_error).__name__}: {last_error}). "
                    f"Fallback '{fallback_tool}' also failed ({type(fallback_error).__name__}: {fallback_error})."
                )

        return f"ToolError: '{action}' failed after retries ({type(last_error).__name__}: {last_error})."

    def _build_report_prompt(self, question: str) -> str:
        notes_block = "\n".join(
            [f"- Note {n.note_id}: {n.claim} | evidence: {n.evidence} | source: {n.source or 'n/a'}" for n in self.memory.notes]
        )
        if not notes_block:
            notes_block = "- No notes captured."

        steps_block = "\n".join(
            [
                f"- Step {s.step}: action={s.action}, input={s.action_input}, observation={s.observation[:500]}"
                for s in self.memory.steps
            ]
        )

        sources = sorted(self.memory.sources.items(), key=lambda x: x[1])
        source_block = "\n".join([f"[{i}] {url}" for url, i in sources])
        if not source_block:
            source_block = "[No external sources captured]"

        return (
            "Create a concise markdown research report with this structure:\n"
            "# Research Report\n"
            "## Question\n"
            "## Executive Summary\n"
            "## Key Findings\n"
            "## Reasoning Trail\n"
            "## Sources\n\n"
            "Use citation markers like [1], [2] in findings where possible.\n"
            "Do not invent references; only use provided sources list.\n\n"
            f"Question:\n{question}\n\n"
            f"Captured notes:\n{notes_block}\n\n"
            f"Action trail:\n{steps_block if steps_block else '- None'}\n\n"
            f"Sources:\n{source_block}\n"
        )

    @traceable(name="synthesize_report")
    def _synthesize_report(self, question: str) -> str:
        prompt = self._build_report_prompt(question)
        response = self.llm.invoke([HumanMessage(content=prompt)])
        if isinstance(response.content, list):
            return "\n".join(str(item) for item in response.content)
        return str(response.content)

    @staticmethod
    def _coerce_chunk_text(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    @traceable(name="synthesize_report_stream")
    def _synthesize_report_stream(
        self,
        question: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        prompt = self._build_report_prompt(question)
        chunks: List[str] = []
        for chunk in self.streaming_llm.stream([HumanMessage(content=prompt)]):
            text = self._coerce_chunk_text(chunk.content)
            if not text:
                continue
            chunks.append(text)
            if on_token:
                on_token(text)
        return "".join(chunks)

    @traceable(name="react_research_agent_run")
    def run(
        self,
        question: str,
        stream_final: bool = False,
        on_final_token: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[Dict[str, object]], None]] = None,
    ) -> str:
        self._emit_status(on_status, state="thinking", message="Thinking about the next best step.")
        for step in range(1, self.max_steps + 1):
            self._emit_status(
                on_status,
                state="thinking",
                message=f"Thinking (step {step}).",
                step=step,
            )
            raw = self._invoke_llm_step(question=question)
            try:
                parsed = self.parse_react_output(raw)
            except ValueError as parse_error:
                observation = f"ParseError: {parse_error}. Raw output: {raw[:400]}"
                self.memory.steps.append(
                    ResearchStep(
                        step=step,
                        thought="",
                        action="parse_error",
                        action_input="",
                        observation=observation,
                    )
                )
                self.logger.log(
                    {
                        "step": step,
                        "thought": "",
                        "action": "parse_error",
                        "action_input": "",
                        "observation": observation,
                    }
                )
                self._emit_status(
                    on_status,
                    state="thinking",
                    message=f"Model output parse issue at step {step}; retrying.",
                    step=step,
                )
                continue

            action = parsed.action.strip().lower()
            if action not in self.VALID_ACTIONS:
                observation = f"InvalidAction: '{action}'. Choose one of {sorted(self.VALID_ACTIONS)}"
                self._emit_status(
                    on_status,
                    state="thinking",
                    message=f"Invalid action '{action}' at step {step}; retrying.",
                    step=step,
                )
            elif action == "finish":
                observation = f"Finish requested: {parsed.action_input}"
                self.memory.steps.append(
                    ResearchStep(
                        step=step,
                        thought=parsed.thought,
                        action=action,
                        action_input=parsed.action_input,
                        observation=observation,
                    )
                )
                self.logger.log(
                    {
                        "step": step,
                        "thought": parsed.thought,
                        "action": action,
                        "action_input": parsed.action_input,
                        "observation": observation,
                    }
                )
                self._emit_status(
                    on_status,
                    state="finalizing",
                    message="Sufficient evidence collected. Finalizing report.",
                    step=step,
                )
                break
            else:
                self._emit_status(
                    on_status,
                    state="using_tool",
                    message=f"Using tool: {action}",
                    tool=action,
                    step=step,
                )
                observation = self._run_tool_with_retry_and_fallback(action=action, action_input=parsed.action_input)
                self._emit_status(
                    on_status,
                    state="thinking",
                    message=f"Analyzing output from {action}.",
                    step=step,
                )

            if len(observation) > self.settings.max_observation_chars:
                observation = observation[: self.settings.max_observation_chars] + "... [truncated]"

            self.memory.steps.append(
                ResearchStep(
                    step=step,
                    thought=parsed.thought,
                    action=action,
                    action_input=parsed.action_input,
                    observation=observation,
                )
            )
            self.logger.log(
                {
                    "step": step,
                    "thought": parsed.thought,
                    "action": action,
                    "action_input": parsed.action_input,
                    "observation": observation,
                }
            )

        self._emit_status(on_status, state="finalizing", message="Writing final report.")
        if stream_final:
            report = self._synthesize_report_stream(question=question, on_token=on_final_token)
        else:
            report = self._synthesize_report(question=question)
        self._emit_status(on_status, state="completed", message="Research complete.")
        return report

    @staticmethod
    def _emit_status(
        on_status: Optional[Callable[[Dict[str, object]], None]],
        state: str,
        message: str,
        tool: Optional[str] = None,
        step: Optional[int] = None,
    ) -> None:
        if not on_status:
            return
        payload: Dict[str, object] = {"state": state, "message": message}
        if tool:
            payload["tool"] = tool
        if step is not None:
            payload["step"] = step
        on_status(payload)

    def format_debug_trace(self) -> str:
        if not self.memory.steps:
            return "No ReAct steps captured."

        lines: List[str] = ["# Debug Trace", "## ReAct Steps"]
        for s in self.memory.steps:
            lines.extend(
                [
                    f"### Step {s.step}",
                    f"- Thought: {s.thought or '[empty]'}",
                    f"- Action: {s.action}",
                    f"- Action Input: {s.action_input or '[empty]'}",
                    f"- Observation: {s.observation}",
                ]
            )

        lines.append("## Notes")
        if self.memory.notes:
            for n in self.memory.notes:
                lines.append(
                    f"- #{n.note_id} claim={n.claim} | evidence={n.evidence} | source={n.source or 'n/a'}"
                )
        else:
            lines.append("- None")

        lines.append("## Sources")
        if self.memory.sources:
            for url, idx in sorted(self.memory.sources.items(), key=lambda x: x[1]):
                lines.append(f"- [{idx}] {url}")
        else:
            lines.append("- None")

        return "\n".join(lines)

    def export_debug_json(self) -> str:
        payload = {
            "steps": [
                {
                    "step": s.step,
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": s.observation,
                }
                for s in self.memory.steps
            ],
            "notes": [
                {
                    "note_id": n.note_id,
                    "claim": n.claim,
                    "evidence": n.evidence,
                    "source": n.source,
                }
                for n in self.memory.notes
            ],
            "sources": self.memory.sources,
        }
        return json.dumps(payload, ensure_ascii=True, indent=2)
