"""
PipelineRepairEnv — the stateful environment.
Session-scoped: each session_id gets its own isolated env state.
"""

import uuid
from typing import Dict, Optional
from files.models import (
    Action, ActionType, DifficultyLevel, Observation,
    PipelineStage, ResetRequest, StepResult,
)
from files.registry import BaseTask, get_task, get_task_by_difficulty, TASK_REGISTRY


class EnvSession:
    """Single agent session state."""

    def __init__(self, task: BaseTask):
        self.task = task
        self.obs: Observation = task.initial_observation()
        self.steps_taken: list[str] = []
        self.total_reward: float = 0.0
        self.done: bool = False

    def _step_key(self, action: Action) -> str:
        if action.target:
            return f"{action.action_type.value}:{action.target}"
        return action.action_type.value

    def step(self, action: Action) -> StepResult:
        if self.done:
            return StepResult(
                observation=self.obs,
                reward=0.0,
                done=True,
                info={"message": "Episode already done. Call /reset to start a new task."},
            )

        key = self._step_key(action)

        # Compute reward
        reward = self.task.reward_for_action(action, {"steps_taken": self.steps_taken})

        # Record step (idempotent — don't double-record)
        if key not in self.steps_taken:
            self.steps_taken.append(key)

        self.total_reward = round(self.total_reward + reward, 4)

        # Determine pipeline stage progression
        stage = self._compute_stage()
        done = self._compute_done(stage)
        self.done = done

        # Build updated observation
        self.obs = self._build_obs(stage, done, action, reward)
        return StepResult(
            observation=self.obs,
            reward=reward,
            done=done,
            info={
                "step_key": key,
                "reward_delta": reward,
                "cumulative_reward": self.total_reward,
                "valid_action": reward > 0,
            },
        )

    def _compute_stage(self) -> PipelineStage:
        """Advance the pipeline stage based on steps completed."""
        taken = set(self.steps_taken)
        task_id = self.task.task_id

        if task_id == "easy_null_fix":
            if "validate_pipeline" in taken:
                return PipelineStage.done
            if "fix_null:age" in taken:
                return PipelineStage.load
            if "identify_issue:age" in taken:
                return PipelineStage.transform
            return PipelineStage.validate

        if task_id == "medium_type_dedup":
            if "validate_pipeline" in taken:
                return PipelineStage.done
            if "fix_type:salary" in taken and "drop_duplicates" in taken:
                return PipelineStage.load
            if "fix_type:salary" in taken or "drop_duplicates" in taken:
                return PipelineStage.transform
            return PipelineStage.transform

        if task_id == "hard_multi_stage":
            all_fixed = all(s in taken for s in [
                "fix_date_format:created_at", "add_missing_column:region", "rewrite_query"
            ])
            if "validate_pipeline" in taken and all_fixed:
                return PipelineStage.done
            if all_fixed:
                return PipelineStage.load
            if "fix_date_format:created_at" in taken or "add_missing_column:region" in taken:
                return PipelineStage.transform
            return PipelineStage.transform

        return PipelineStage.validate

    def _compute_done(self, stage: PipelineStage) -> bool:
        if stage == PipelineStage.done:
            return True
        steps_left = self.obs.steps_remaining - 1
        return steps_left <= 0

    def _build_obs(self, stage: PipelineStage, done: bool, action: Action, reward: float) -> Observation:
        prev = self.obs
        steps_remaining = max(0, prev.steps_remaining - 1)

        # Compute live errors based on what's been fixed
        errors = list(prev.errors)
        for step in self.steps_taken:
            errors = [e for e in errors if not self._error_resolved_by(step, e)]

        log_entry = (
            f"INFO  [{stage.value}]  Action '{action.action_type.value}'"
            + (f" on '{action.target}'" if action.target else "")
            + f"  →  reward={reward:+.2f}"
        )

        return Observation(
            task_id=prev.task_id,
            difficulty=prev.difficulty,
            pipeline_stage=stage,
            errors=errors,
            warnings=prev.warnings if errors else [],
            schema=prev.schema,
            data_sample=prev.data_sample,
            logs=prev.logs + [log_entry],
            steps_taken=list(self.steps_taken),
            steps_remaining=steps_remaining,
            done=done,
            reward_so_far=self.total_reward,
        )

    @staticmethod
    def _error_resolved_by(step_key: str, error: str) -> bool:
        resolution_map = {
            "fix_null:age": "NullValueError",
            "fix_type:salary": "TypeError",
            "drop_duplicates": "DuplicateRowError",
            "fix_date_format:created_at": "DateFormatError",
            "add_missing_column:region": "SchemaError",
            "rewrite_query": "QueryError",
        }
        keyword = resolution_map.get(step_key)
        return keyword is not None and keyword in error

    def grade(self) -> dict:
        stage = self._compute_stage()
        result = self.task.grade(self.steps_taken, stage)
        return result.model_dump()


# ─────────────────────────────────────────────
# Session manager (in-memory)
# ─────────────────────────────────────────────
class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, EnvSession] = {}

    def reset(self, request: ResetRequest) -> tuple[str, Observation]:
        if request.task_id:
            task = get_task(request.task_id)
        elif request.difficulty:
            task = get_task_by_difficulty(request.difficulty)
        else:
            # Default: cycle through tasks
            task = get_task("easy_null_fix")

        session_id = str(uuid.uuid4())
        session = EnvSession(task)
        self._sessions[session_id] = session
        return session_id, session.obs

    def step(self, session_id: str, action: Action) -> StepResult:
        session = self._get(session_id)
        return session.step(action)

    def state(self, session_id: str) -> Observation:
        return self._get(session_id).obs

    def grade(self, session_id: str) -> dict:
        return self._get(session_id).grade()

    def _get(self, session_id: str) -> EnvSession:
        if session_id not in self._sessions:
            raise KeyError(f"Session '{session_id}' not found. Call /reset first.")
        return self._sessions[session_id]

    def list_tasks(self) -> list:
        return [
            {
                "task_id": t.task_id,
                "difficulty": t.difficulty.value,
                "description": t.description,
            }
            for t in TASK_REGISTRY.values()
        ]


session_manager = SessionManager()
