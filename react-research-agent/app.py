from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from react_research_agent.agent import ReActResearchAgent

load_dotenv()

app = FastAPI(title="ReAct Research Agent API", version="1.0.0")


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=3)
    max_steps: Optional[int] = Field(default=None, ge=1, le=30)
    model: Optional[str] = None


class ResearchResponse(BaseModel):
    report: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest) -> ResearchResponse:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")

    agent = ReActResearchAgent(model_name=req.model, max_steps=req.max_steps)
    report = agent.run(req.question, stream_final=False)
    return ResearchResponse(report=report)
