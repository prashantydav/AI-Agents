from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from react_research_agent.agent import ReActResearchAgent

load_dotenv()

app = FastAPI(title="ReAct Research Agent API", version="1.0.0")
logger = logging.getLogger("uvicorn.error")


def _parse_cors_origins() -> list[str]:
    raw = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
    if not raw:
        return ["*"]
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["*"]


cors_origins = _parse_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=3)
    max_steps: Optional[int] = Field(default=None, ge=1, le=30)
    model: Optional[str] = None


class ResearchResponse(BaseModel):
    report: str


def _error_message(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        message = "No additional error details available."
    return message[:700]


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "ReAct Research Agent API",
        "health": "/health",
        "research": "/research",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest) -> ResearchResponse:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")

    try:
        agent = ReActResearchAgent(model_name=req.model, max_steps=req.max_steps)
        report = agent.run(req.question, stream_final=False)
        return ResearchResponse(report=report)
    except Exception as exc:
        logger.exception("Unhandled error while processing /research request")
        error_type = type(exc).__name__
        detail = _error_message(exc)

        if error_type == "AuthenticationError":
            raise HTTPException(status_code=401, detail=f"OpenAI authentication failed: {detail}") from exc
        if error_type == "RateLimitError":
            raise HTTPException(status_code=429, detail=f"OpenAI rate limit reached: {detail}") from exc
        if error_type in {"BadRequestError", "NotFoundError"}:
            raise HTTPException(status_code=400, detail=f"Model/request error: {detail}") from exc
        if error_type in {"APIConnectionError", "APITimeoutError"}:
            raise HTTPException(status_code=503, detail=f"OpenAI connection/timeout error: {detail}") from exc

        raise HTTPException(status_code=500, detail=f"Research failed ({error_type}): {detail}") from exc
