"""
Task registry: 3 tasks (easy / medium / hard) for the Data Pipeline Repair environment.
Each task defines:
  - initial state
  - required action sequence (for grading)
  - grader (deterministic, 0.0–1.0)
  - partial reward table
"""

from typing import Any, Dict, List, Optional, Tuple
from files.models import (
    Action, ActionType, DataSample, DifficultyLevel,
    GraderResult, Observation, PipelineStage,
)


# ─────────────────────────────────────────────
# Base task class
# ─────────────────────────────────────────────
class BaseTask:
    task_id: str
    difficulty: DifficultyLevel
    description: str

    # Ordered list of (action_type, target) pairs that constitute a full solution.
    # Multiple valid orderings can be expressed by subclasses overriding `grade()`.
    solution_steps: List[Tuple[ActionType, Optional[str]]]

    def initial_observation(self) -> Observation:
        raise NotImplementedError

    def grade(self, steps_taken: List[str], final_stage: PipelineStage) -> GraderResult:
        raise NotImplementedError

    def reward_for_action(self, action: Action, current_state: dict) -> float:
        raise NotImplementedError


# ─────────────────────────────────────────────
# EASY — Missing values in a single column
# ─────────────────────────────────────────────
class EasyTask(BaseTask):
    task_id = "easy_null_fix"
    difficulty = DifficultyLevel.easy
    description = (
        "The 'age' column in the users table has NULL values. "
        "Identify the issue, apply a fix (fill nulls with 0), then validate."
    )
    solution_steps = [
        (ActionType.identify_issue, "age"),
        (ActionType.fix_null, "age"),
        (ActionType.validate_pipeline, None),
    ]

    def initial_observation(self) -> Observation:
        return Observation(
            task_id=self.task_id,
            difficulty=self.difficulty,
            pipeline_stage=PipelineStage.validate,
            errors=["NullValueError: column 'age' contains 12 NULL values (out of 100 rows)"],
            warnings=["Pipeline halted at VALIDATE stage"],
            schema={"user_id": "int", "name": "str", "age": "int", "email": "str"},
            data_sample=DataSample(
                columns=["user_id", "name", "age", "email"],
                rows=[
                    [1, "Alice", None, "alice@example.com"],
                    [2, "Bob", 30, "bob@example.com"],
                    [3, "Carol", None, "carol@example.com"],
                    [4, "Dave", 25, "dave@example.com"],
                ],
                dtypes={"user_id": "int64", "name": "object", "age": "float64", "email": "object"},
            ),
            logs=[
                "INFO  [ingest]    Loaded 100 rows from users.csv",
                "INFO  [validate]  Schema check started",
                "ERROR [validate]  NullValueError in column 'age': 12 nulls detected",
                "ERROR [validate]  Pipeline halted",
            ],
            steps_taken=[],
            steps_remaining=5,
            done=False,
            reward_so_far=0.0,
        )

    def reward_for_action(self, action: Action, current_state: dict) -> float:
        taken = current_state.get("steps_taken", [])
        # +0.2 identify issue on correct column
        if action.action_type == ActionType.identify_issue and action.target == "age":
            if "identify_issue:age" not in taken:
                return 0.2
        # +0.5 correct fix
        if action.action_type == ActionType.fix_null and action.target == "age":
            if "identify_issue:age" in taken and "fix_null:age" not in taken:
                return 0.5
        # +0.3 validate at end
        if action.action_type == ActionType.validate_pipeline:
            if "fix_null:age" in taken and "validate_pipeline" not in taken:
                return 0.3
        # Wrong action or premature action
        return -0.2

    def grade(self, steps_taken: List[str], final_stage: PipelineStage) -> GraderResult:
        breakdown: Dict[str, float] = {}
        breakdown["identified_issue"] = 1.0 if "identify_issue:age" in steps_taken else 0.0
        breakdown["fixed_null"] = 1.0 if "fix_null:age" in steps_taken else 0.0
        breakdown["validated"] = 1.0 if "validate_pipeline" in steps_taken else 0.0
        breakdown["pipeline_complete"] = 1.0 if final_stage == PipelineStage.done else 0.0

        weights = {"identified_issue": 0.2, "fixed_null": 0.5, "validated": 0.2, "pipeline_complete": 0.1}
        score = sum(breakdown[k] * weights[k] for k in breakdown)
        score = round(min(score, 1.0), 4)

        return GraderResult(
            task_id=self.task_id,
            score=score,
            passed=score >= 0.8,
            breakdown=breakdown,
            feedback=(
                "✅ Full pipeline repaired." if score >= 0.8
                else f"⚠️ Partial fix. Score: {score}. Missing steps: " +
                     ", ".join(k for k, v in breakdown.items() if v == 0.0)
            ),
        )


# ─────────────────────────────────────────────
# MEDIUM — Wrong dtype + duplicate rows
# ─────────────────────────────────────────────
class MediumTask(BaseTask):
    task_id = "medium_type_dedup"
    difficulty = DifficultyLevel.medium
    description = (
        "The 'salary' column is stored as string instead of float, "
        "and the dataset contains 20 duplicate rows. "
        "Fix the type, drop duplicates, then validate."
    )
    solution_steps = [
        (ActionType.identify_issue, "salary"),
        (ActionType.fix_type, "salary"),
        (ActionType.identify_issue, "duplicates"),
        (ActionType.drop_duplicates, None),
        (ActionType.validate_pipeline, None),
    ]

    def initial_observation(self) -> Observation:
        return Observation(
            task_id=self.task_id,
            difficulty=self.difficulty,
            pipeline_stage=PipelineStage.transform,
            errors=[
                "TypeError: column 'salary' expected float, got object (str)",
                "DuplicateRowError: 20 duplicate rows detected in dataset",
            ],
            warnings=["Downstream aggregations will fail on string salary"],
            schema={"emp_id": "int", "name": "str", "salary": "float", "dept": "str"},
            data_sample=DataSample(
                columns=["emp_id", "name", "salary", "dept"],
                rows=[
                    [101, "Alice", "75000.0", "Engineering"],
                    [102, "Bob", "82000.0", "Marketing"],
                    [101, "Alice", "75000.0", "Engineering"],  # duplicate
                    [103, "Carol", "91000.5", "Engineering"],
                ],
                dtypes={"emp_id": "int64", "name": "object", "salary": "object", "dept": "object"},
            ),
            logs=[
                "INFO  [ingest]     Loaded 200 rows from employees.csv",
                "INFO  [transform]  Applying schema coercion",
                "ERROR [transform]  TypeError: salary must be float, found object",
                "ERROR [transform]  DuplicateRowError: 20 duplicates",
                "ERROR [transform]  Pipeline halted",
            ],
            steps_taken=[],
            steps_remaining=8,
            done=False,
            reward_so_far=0.0,
        )

    def reward_for_action(self, action: Action, current_state: dict) -> float:
        taken = current_state.get("steps_taken", [])
        if action.action_type == ActionType.identify_issue and action.target == "salary":
            if "identify_issue:salary" not in taken:
                return 0.15
        if action.action_type == ActionType.fix_type and action.target == "salary":
            if "identify_issue:salary" in taken and "fix_type:salary" not in taken:
                return 0.3
        if action.action_type == ActionType.identify_issue and action.target == "duplicates":
            if "identify_issue:duplicates" not in taken:
                return 0.15
        if action.action_type == ActionType.drop_duplicates:
            if "identify_issue:duplicates" in taken and "drop_duplicates" not in taken:
                return 0.25
        if action.action_type == ActionType.validate_pipeline:
            if "fix_type:salary" in taken and "drop_duplicates" in taken and "validate_pipeline" not in taken:
                return 0.25
        return -0.2

    def grade(self, steps_taken: List[str], final_stage: PipelineStage) -> GraderResult:
        breakdown: Dict[str, float] = {
            "identified_salary_type": 1.0 if "identify_issue:salary" in steps_taken else 0.0,
            "fixed_salary_type": 1.0 if "fix_type:salary" in steps_taken else 0.0,
            "identified_duplicates": 1.0 if "identify_issue:duplicates" in steps_taken else 0.0,
            "dropped_duplicates": 1.0 if "drop_duplicates" in steps_taken else 0.0,
            "validated": 1.0 if "validate_pipeline" in steps_taken else 0.0,
            "pipeline_complete": 1.0 if final_stage == PipelineStage.done else 0.0,
        }
        weights = {
            "identified_salary_type": 0.1,
            "fixed_salary_type": 0.3,
            "identified_duplicates": 0.1,
            "dropped_duplicates": 0.25,
            "validated": 0.15,
            "pipeline_complete": 0.1,
        }
        score = round(min(sum(breakdown[k] * weights[k] for k in breakdown), 1.0), 4)
        return GraderResult(
            task_id=self.task_id,
            score=score,
            passed=score >= 0.8,
            breakdown=breakdown,
            feedback=(
                "✅ Both issues resolved and pipeline validated."
                if score >= 0.8 else
                f"⚠️ Incomplete. Score: {score}. Missing: " +
                ", ".join(k for k, v in breakdown.items() if v == 0.0)
            ),
        )


# ─────────────────────────────────────────────
# HARD — Schema mismatch + bad dates + bad query + multi-stage failure
# ─────────────────────────────────────────────
class HardTask(BaseTask):
    task_id = "hard_multi_stage"
    difficulty = DifficultyLevel.hard
    description = (
        "A multi-stage pipeline has failed with three distinct issues: "
        "(1) 'created_at' date column uses inconsistent formats (MM/DD/YYYY vs YYYY-MM-DD), "
        "(2) the 'region' column is missing from the schema but present in raw data, "
        "(3) the downstream SQL query references a non-existent column alias. "
        "Identify all issues, apply all fixes in the correct order, then validate."
    )
    solution_steps = [
        (ActionType.identify_issue, "created_at"),
        (ActionType.fix_date_format, "created_at"),
        (ActionType.identify_issue, "region"),
        (ActionType.add_missing_column, "region"),
        (ActionType.identify_issue, "query"),
        (ActionType.rewrite_query, None),
        (ActionType.validate_pipeline, None),
    ]

    def initial_observation(self) -> Observation:
        return Observation(
            task_id=self.task_id,
            difficulty=self.difficulty,
            pipeline_stage=PipelineStage.transform,
            errors=[
                "DateFormatError: column 'created_at' has mixed formats (MM/DD/YYYY and YYYY-MM-DD)",
                "SchemaError: column 'region' present in source but missing from schema definition",
                "QueryError: SELECT alias 'user_count' not defined — referenced before assignment",
            ],
            warnings=[
                "Data loaded from 3 source files with differing schemas",
                "Pipeline will fail at LOAD stage even if TRANSFORM passes",
            ],
            schema={
                "order_id": "int", "customer_id": "int",
                "amount": "float", "created_at": "str",
            },
            data_sample=DataSample(
                columns=["order_id", "customer_id", "amount", "created_at", "region"],
                rows=[
                    [1001, 1, 49.99, "2024-01-15", "APAC"],
                    [1002, 2, 120.0, "03/22/2024", "EMEA"],
                    [1003, 3, 75.5, "2024-03-01", "APAC"],
                    [1004, 4, 200.0, "11/05/2024", "AMER"],
                ],
                dtypes={
                    "order_id": "int64", "customer_id": "int64",
                    "amount": "float64", "created_at": "object", "region": "object",
                },
            ),
            logs=[
                "INFO  [ingest]     Loaded 850 rows from orders_Q1.csv, orders_Q2.csv, orders_Q3.csv",
                "INFO  [validate]   Schema validation started",
                "ERROR [validate]   SchemaError: 'region' not in schema definition",
                "INFO  [transform]  Date normalisation started",
                "ERROR [transform]  DateFormatError: mixed date formats in 'created_at'",
                "INFO  [load]       SQL query execution started",
                "ERROR [load]       QueryError: undefined alias 'user_count'",
                "ERROR [load]       Pipeline halted — 3 errors require resolution",
            ],
            steps_taken=[],
            steps_remaining=12,
            done=False,
            reward_so_far=0.0,
        )

    def reward_for_action(self, action: Action, current_state: dict) -> float:
        taken = current_state.get("steps_taken", [])
        rewards = {
            ("identify_issue", "created_at"): ("identify_issue:created_at" not in taken, 0.1),
            ("fix_date_format", "created_at"): ("identify_issue:created_at" in taken and "fix_date_format:created_at" not in taken, 0.2),
            ("identify_issue", "region"): ("identify_issue:region" not in taken, 0.1),
            ("add_missing_column", "region"): ("identify_issue:region" in taken and "add_missing_column:region" not in taken, 0.2),
            ("identify_issue", "query"): ("identify_issue:query" not in taken, 0.1),
            ("rewrite_query", None): ("identify_issue:query" in taken and "rewrite_query" not in taken, 0.2),
            ("validate_pipeline", None): (
                all(s in taken for s in ["fix_date_format:created_at", "add_missing_column:region", "rewrite_query"])
                and "validate_pipeline" not in taken, 0.3
            ),
        }
        key = (action.action_type.value, action.target)
        if key in rewards:
            condition, r = rewards[key]
            if condition:
                return r
        return -0.15

    def grade(self, steps_taken: List[str], final_stage: PipelineStage) -> GraderResult:
        breakdown: Dict[str, float] = {
            "identified_date_issue": 1.0 if "identify_issue:created_at" in steps_taken else 0.0,
            "fixed_date_format": 1.0 if "fix_date_format:created_at" in steps_taken else 0.0,
            "identified_schema_issue": 1.0 if "identify_issue:region" in steps_taken else 0.0,
            "added_missing_column": 1.0 if "add_missing_column:region" in steps_taken else 0.0,
            "identified_query_issue": 1.0 if "identify_issue:query" in steps_taken else 0.0,
            "rewrote_query": 1.0 if "rewrite_query" in steps_taken else 0.0,
            "validated": 1.0 if "validate_pipeline" in steps_taken else 0.0,
            "pipeline_complete": 1.0 if final_stage == PipelineStage.done else 0.0,
        }
        weights = {
            "identified_date_issue": 0.05,
            "fixed_date_format": 0.2,
            "identified_schema_issue": 0.05,
            "added_missing_column": 0.2,
            "identified_query_issue": 0.05,
            "rewrote_query": 0.2,
            "validated": 0.15,
            "pipeline_complete": 0.1,
        }
        score = round(min(sum(breakdown[k] * weights[k] for k in breakdown), 1.0), 4)
        return GraderResult(
            task_id=self.task_id,
            score=score,
            passed=score >= 0.8,
            breakdown=breakdown,
            feedback=(
                "✅ All three pipeline issues resolved."
                if score >= 0.8 else
                f"⚠️ Partial fix. Score: {score}. Missing: " +
                ", ".join(k for k, v in breakdown.items() if v == 0.0)
            ),
        )


# ─────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────
TASK_REGISTRY: Dict[str, BaseTask] = {
    EasyTask.task_id: EasyTask(),
    MediumTask.task_id: MediumTask(),
    HardTask.task_id: HardTask(),
}

def get_task(task_id: str) -> BaseTask:
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"Unknown task_id '{task_id}'. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_id]

def get_task_by_difficulty(difficulty: DifficultyLevel) -> BaseTask:
    for task in TASK_REGISTRY.values():
        if task.difficulty == difficulty:
            return task
    raise ValueError(f"No task found for difficulty '{difficulty}'")
