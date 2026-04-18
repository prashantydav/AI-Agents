# ReAct Research Agent (Project 2.1)

A Python research agent that uses the ReAct pattern (`Thought -> Action -> Observation`) to answer multi-hop questions.

## Features
- Full ReAct loop with iterative reasoning and tool use
- Five custom tools:
  - `web_search` (Tavily Search API)
  - `wikipedia` (Wikipedia lookup)
  - `url_reader` (fetch and parse article content)
  - `calculator` (safe arithmetic evaluator)
  - `note_taker` (captures structured research notes)
- Tool retry + fallback behavior
- LangSmith tracing hooks for intermediate step observability
- Final structured markdown report with citations

## Project Structure
```text
react-research-agent/
  src/react_research_agent/
    agent.py
    config.py
    logging_utils.py
    models.py
    tools.py
  tests/
    test_parser_and_calculator.py
  .env.example
  requirements.txt
  main.py
```

## Setup
1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy env file and add API keys:

```bash
cp .env.example .env
```

Required for full functionality:
- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

Optional (for LangSmith observability):
- `LANGSMITH_API_KEY`
- `LANGSMITH_TRACING=true`
- `LANGSMITH_PROJECT=react-research-agent`

## Run
```bash
python main.py "What are the economic and environmental trade-offs of lithium iron phosphate batteries versus NMC batteries in grid storage?"
```

Optional flags:
- `--max-steps` (default: 10)
- `--model` (default: `gpt-4.1-mini`)
- `--log-file` (default: `logs/steps.jsonl`)
- `--no-stream` (disable final report token streaming)
- `--debug` (print Thought/Action/Observation trace after final output)
- `--debug-json` (print raw JSON steps/notes after final output)

## Notes on citations
- Tools register sources as they are discovered.
- The final report includes numbered citations in a `## Sources` section.

## Basic test
```bash
python -m pytest -q
```
