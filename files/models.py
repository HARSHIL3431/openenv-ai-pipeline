from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from enum import Enum


class DifficultyLevel(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class PipelineStage(str, Enum):
    ingest = "ingest"
    validate = "validate"
    transform = "transform"
    load = "load"
    done = "done"


class ActionType(str, Enum):
    identify_issue = "identify_issue"
    fix_null = "fix_null"
    fix_type = "fix_type"
    fix_schema = "fix_schema"
    drop_duplicates = "drop_duplicates"
    rewrite_query = "rewrite_query"
    normalize_column = "normalize_column"
    validate_pipeline = "validate_pipeline"
    add_missing_column = "add_missing_column"
    fix_date_format = "fix_date_format"


class Action(BaseModel):
    action_type: ActionType
    target: Optional[str] = Field(None, description="Column name, table name, or stage to act on")
    value: Optional[str] = Field(None, description="Cast type, fill value, format string, etc.")

    model_config = {"json_schema_extra": {
        "examples": [
            {"action_type": "fix_null", "target": "age", "value": "0"},
            {"action_type": "fix_type", "target": "salary", "value": "float"},
            {"action_type": "validate_pipeline", "target": None, "value": None},
        ]
    }}


class DataSample(BaseModel):
    columns: List[str]
    rows: List[List[Any]]
    dtypes: Dict[str, str]


class Observation(BaseModel):
    task_id: str
    difficulty: DifficultyLevel
    pipeline_stage: PipelineStage
    errors: List[str]
    warnings: List[str]
    schema: Dict[str, str]
    data_sample: DataSample
    logs: List[str]
    steps_taken: List[str]
    steps_remaining: int
    done: bool
    reward_so_far: float


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    difficulty: Optional[DifficultyLevel] = None


class GraderResult(BaseModel):
    task_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    passed: bool
    breakdown: Dict[str, float]
    feedback: str
