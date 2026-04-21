from __future__ import annotations

from datetime import datetime, timezone
import os
import sys
import logging
from pathlib import Path
from threading import Lock, Thread
from typing import Optional
from uuid import uuid4

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
JOBS_LOCK = Lock()
JOBS: dict[str, dict[str, object]] = {}


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


class ResearchJobCreateResponse(BaseModel):
    job_id: str
    state: str
    done: bool


class ResearchJobStatusResponse(BaseModel):
    job_id: str
    state: str
    message: str
    done: bool
    tool: Optional[str] = None
    step: Optional[int] = None
    report: Optional[str] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str
    finished_at: Optional[str] = None


def _error_message(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        message = "No additional error details available."
    return message[:700]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_job(job_id: str) -> Optional[dict[str, object]]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return None
        return dict(job)


def _update_job(job_id: str, **updates: object) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = _utc_now_iso()


def _create_job() -> dict[str, object]:
    job_id = uuid4().hex
    now = _utc_now_iso()
    record: dict[str, object] = {
        "job_id": job_id,
        "state": "queued",
        "message": "Queued.",
        "done": False,
        "tool": None,
        "step": None,
        "report": None,
        "error": None,
        "created_at": now,
        "updated_at": now,
        "finished_at": None,
    }
    with JOBS_LOCK:
        JOBS[job_id] = record
    return dict(record)


def _run_research_job(job_id: str, req: ResearchRequest) -> None:
    def on_status(payload: dict[str, object]) -> None:
        _update_job(
            job_id,
            state=str(payload.get("state") or "thinking"),
            message=str(payload.get("message") or ""),
            tool=payload.get("tool"),
            step=payload.get("step"),
        )

    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("Missing OPENAI_API_KEY")

        agent = ReActResearchAgent(model_name=req.model, max_steps=req.max_steps)
        report = agent.run(req.question, stream_final=False, on_status=on_status)
        _update_job(
            job_id,
            state="completed",
            message="Research complete.",
            done=True,
            report=report,
            tool=None,
            finished_at=_utc_now_iso(),
        )
    except Exception as exc:
        logger.exception("Unhandled error while processing research job")
        _update_job(
            job_id,
            state="failed",
            message="Research failed.",
            error=f"{type(exc).__name__}: {_error_message(exc)}",
            done=True,
            tool=None,
            finished_at=_utc_now_iso(),
        )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "ReAct Research Agent API",
        "health": "/health",
        "research": "/research",
        "research_jobs_create": "/research/jobs",
        "research_job_status": "/research/jobs/{job_id}",
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


@app.post("/research/jobs", response_model=ResearchJobCreateResponse)
def create_research_job(req: ResearchRequest) -> ResearchJobCreateResponse:
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY")

    record = _create_job()
    thread = Thread(target=_run_research_job, args=(record["job_id"], req), daemon=True)
    thread.start()
    return ResearchJobCreateResponse(job_id=str(record["job_id"]), state=str(record["state"]), done=bool(record["done"]))


@app.get("/research/jobs/{job_id}", response_model=ResearchJobStatusResponse)
def get_research_job(job_id: str) -> ResearchJobStatusResponse:
    record = _get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Research job not found")

    return ResearchJobStatusResponse(
        job_id=str(record["job_id"]),
        state=str(record["state"]),
        message=str(record["message"]),
        done=bool(record["done"]),
        tool=str(record["tool"]) if record["tool"] is not None else None,
        step=int(record["step"]) if record["step"] is not None else None,
        report=str(record["report"]) if record["report"] is not None else None,
        error=str(record["error"]) if record["error"] is not None else None,
        created_at=str(record["created_at"]),
        updated_at=str(record["updated_at"]),
        finished_at=str(record["finished_at"]) if record["finished_at"] is not None else None,
    )
