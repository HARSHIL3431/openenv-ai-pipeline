"""
FastAPI server — OpenEnv compliant.
Exposes: /reset  /step  /state  /grade  /tasks  /health
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from files.models import Action, DifficultyLevel, ResetRequest
from files.environment import session_manager

app = FastAPI(
    title="Data Pipeline Repair Environment",
    description=(
        "An OpenEnv-compliant RL environment where an LLM agent diagnoses "
        "and repairs broken data pipelines across three difficulty levels."
    ),
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request / Response wrappers
# ─────────────────────────────────────────────

class ResetResponse(BaseModel):
    session_id: str
    observation: dict


class StepRequest(BaseModel):
    session_id: str
    action: Action


class StepResponse(BaseModel):
    session_id: str
    observation: dict
    reward: float
    done: bool
    info: dict


class StateResponse(BaseModel):
    session_id: str
    observation: dict


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "pipeline-repair-env"}


@app.get("/tasks")
def list_tasks():
    """List all available tasks."""
    return {"tasks": session_manager.list_tasks()}


@app.post("/reset", response_model=ResetResponse)
def reset(
    task_id: Optional[str] = Query(None, description="Specific task ID"),
    difficulty: Optional[DifficultyLevel] = Query(None, description="easy | medium | hard"),
):
    """
    Start a new episode. Returns a session_id and the initial observation.
    Optionally pass task_id or difficulty to select a specific task.
    """
    try:
        request = ResetRequest(task_id=task_id, difficulty=difficulty)
        session_id, obs = session_manager.reset(request)
        return ResetResponse(session_id=session_id, observation=obs.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(body: StepRequest):
    """
    Take one action in the environment.
    
    Valid action_types:
      identify_issue, fix_null, fix_type, fix_schema, drop_duplicates,
      rewrite_query, normalize_column, validate_pipeline,
      add_missing_column, fix_date_format
    
    Example:
      {"session_id": "...", "action": {"action_type": "fix_null", "target": "age", "value": "0"}}
    """
    try:
        result = session_manager.step(body.session_id, body.action)
        return StepResponse(
            session_id=body.session_id,
            observation=result.observation.model_dump(),
            reward=result.reward,
            done=result.done,
            info=result.info,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/state", response_model=StateResponse)
def state(session_id: str = Query(..., description="Session ID from /reset")):
    """Get the current observation without taking an action."""
    try:
        obs = session_manager.state(session_id)
        return StateResponse(session_id=session_id, observation=obs.model_dump())
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/grade")
def grade(session_id: str = Query(..., description="Session ID from /reset")):
    """
    Grade the current session. Returns deterministic score 0.0–1.0.
    Can be called at any point during or after the episode.
    """
    try:
        result = session_manager.grade(session_id)
        return {"session_id": session_id, **result}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/")
def root():
    return {
        "name": "Data Pipeline Repair Environment",
        "version": "1.0.0",
        "openenv": True,
        "endpoints": ["/reset", "/step", "/state", "/grade", "/tasks", "/health", "/docs"],
        "tasks": [t["task_id"] for t in session_manager.list_tasks()],
        "tag": "openenv",
    }
