from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from react_research_agent.agent import ReActResearchAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ReAct research agent.")
    parser.add_argument("question", type=str, help="Research question to answer")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum ReAct iterations")
    parser.add_argument("--model", type=str, default=None, help="OpenAI model name")
    parser.add_argument("--log-file", type=str, default="logs/steps.jsonl", help="JSONL step log path")
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable token streaming for final report generation",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print Thought/Action/Observation trace after final output",
    )
    parser.add_argument(
        "--debug-json",
        action="store_true",
        help="Print internal steps/notes JSON after the report",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    agent = ReActResearchAgent(model_name=args.model, max_steps=args.max_steps, log_file=args.log_file)

    def on_token(token: str) -> None:
        print(token, end="", flush=True)

    if args.no_stream:
        report = agent.run(args.question)
        print(report)
    else:
        report = agent.run(args.question, stream_final=True, on_final_token=on_token)
        print()

    if args.debug:
        print("\n--- DEBUG TRACE ---")
        print(agent.format_debug_trace())

    if args.debug_json:
        print("\n--- DEBUG JSON ---")
        print(agent.export_debug_json())


if __name__ == "__main__":
    main()
