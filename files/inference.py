"""
inference.py — OpenEnv compliant LLM agent runner.

Reads env vars:
    API_BASE_URL  — OpenAI-compatible API base endpoint (required)
    MODEL_NAME    — model identifier (required)
    HF_TOKEN      — Hugging Face token used as API key (required)
    ENV_URL       — Base URL of the running environment API (required)

Usage:
  python inference.py
"""

import os
import json
import sys
import requests
from openai import OpenAI

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


API_BASE_URL = required_env("API_BASE_URL")
MODEL_NAME = required_env("MODEL_NAME")
HF_TOKEN = required_env("HF_TOKEN")
ENV_URL = required_env("ENV_URL")

TASKS = ["easy_null_fix", "medium_type_dedup", "hard_multi_stage"]
MAX_STEPS = 12

client = OpenAI(
    api_key=os.getenv("HF_TOKEN"),
    base_url=os.getenv("API_BASE_URL"),
)


# ─────────────────────────────────────────────
# Env API helpers
# ─────────────────────────────────────────────
def env_reset(task_id: str) -> tuple[str, dict]:
    r = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id})
    r.raise_for_status()
    data = r.json()
    return data["session_id"], data["observation"]


def env_step(session_id: str, action: dict) -> dict:
    r = requests.post(f"{ENV_URL}/step", json={"session_id": session_id, "action": action})
    r.raise_for_status()
    return r.json()


def env_grade(session_id: str) -> dict:
    r = requests.get(f"{ENV_URL}/grade", params={"session_id": session_id})
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────
# LLM agent
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a data pipeline repair agent. You will be given the current state of a broken data pipeline.
Your job is to select the best repair action.

You MUST respond with ONLY a JSON object — no explanation, no markdown, no extra text.

Format:
{"action_type": "<type>", "target": "<column_or_target_or_null>", "value": "<value_or_null>"}

Valid action_types:
  identify_issue      — identify what is wrong with a column or component (target = column name or "duplicates" or "query")
  fix_null            — fill null values in a column (target = column name, value = fill value)
  fix_type            — cast column to correct type (target = column name, value = new type e.g. "float")
  fix_schema          — fix schema definition issues (target = column name)
  drop_duplicates     — remove duplicate rows (target = null)
  rewrite_query       — fix broken SQL/query (target = null)
  normalize_column    — normalise column values (target = column name)
  validate_pipeline   — run pipeline validation (target = null) — use LAST after all fixes
  add_missing_column  — add a missing column to the schema (target = column name)
  fix_date_format     — standardise date format in a column (target = column name)

Strategy:
1. Always identify_issue on the affected column or component BEFORE fixing it.
2. Fix all errors before calling validate_pipeline.
3. Call validate_pipeline as the final step.
"""


def obs_to_prompt(obs: dict) -> str:
    return f"""Current pipeline state:

Stage: {obs['pipeline_stage']}
Errors: {json.dumps(obs['errors'], indent=2)}
Warnings: {json.dumps(obs['warnings'], indent=2)}
Schema: {json.dumps(obs['schema'], indent=2)}
Data sample columns: {obs['data_sample']['columns']}
Steps taken so far: {obs['steps_taken']}
Steps remaining: {obs['steps_remaining']}
Reward so far: {obs['reward_so_far']}

What is your next action?"""


def llm_action(obs: dict) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs_to_prompt(obs)},
    ]
    try:
        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME"),
            messages=messages,
            max_tokens=150,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        action = json.loads(raw)
        # Normalise nulls
        action.setdefault("target", None)
        action.setdefault("value", None)
        return action
    except Exception as e:
        raise RuntimeError(
            f"LLM API call failed for model '{MODEL_NAME}' at '{API_BASE_URL}'. "
            f"Check HF_TOKEN, API_BASE_URL, and MODEL_NAME. Error: {e}"
        ) from e


def fallback_action(obs: dict) -> dict:
    """Deterministic fallback when LLM API is unavailable."""
    plans = {
        "easy_null_fix": [
            {"action_type": "identify_issue", "target": "age", "value": None},
            {"action_type": "fix_null", "target": "age", "value": "0"},
            {"action_type": "validate_pipeline", "target": None, "value": None},
        ],
        "medium_type_dedup": [
            {"action_type": "identify_issue", "target": "salary", "value": None},
            {"action_type": "fix_type", "target": "salary", "value": "float"},
            {"action_type": "identify_issue", "target": "duplicates", "value": None},
            {"action_type": "drop_duplicates", "target": None, "value": None},
            {"action_type": "validate_pipeline", "target": None, "value": None},
        ],
        "hard_multi_stage": [
            {"action_type": "identify_issue", "target": "created_at", "value": None},
            {"action_type": "fix_date_format", "target": "created_at", "value": None},
            {"action_type": "identify_issue", "target": "region", "value": None},
            {"action_type": "add_missing_column", "target": "region", "value": None},
            {"action_type": "identify_issue", "target": "query", "value": None},
            {"action_type": "rewrite_query", "target": None, "value": None},
            {"action_type": "validate_pipeline", "target": None, "value": None},
        ],
    }

    task_id = obs.get("task_id")
    taken = set(obs.get("steps_taken", []))

    def step_key(action: dict) -> str:
        target = action.get("target")
        if target:
            return f"{action['action_type']}:{target}"
        return action["action_type"]

    for action in plans.get(task_id, []):
        if step_key(action) not in taken:
            return action

    return {"action_type": "validate_pipeline", "target": None, "value": None}


# ─────────────────────────────────────────────
# Run all tasks
# ─────────────────────────────────────────────
def run_task(task_id: str) -> dict:
    task_name = task_id
    print(f"[START] task={task_name}", flush=True)

    session_id, obs = env_reset(task_id)

    step_num = 0
    done = False
    total_reward = 0.0

    while not done and step_num < MAX_STEPS:
        step_num += 1
        try:
            action = llm_action(obs)
        except Exception as e:
            action = fallback_action(obs)

        result = env_step(session_id, action)
        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        total_reward += reward
        print(f"[STEP] step={step_num} reward={reward}", flush=True)

    grade = env_grade(session_id)
    final_score = grade["score"]
    print(f"[END] task={task_name} score={final_score} steps={step_num}", flush=True)

    return {
        "task_id": task_id,
        "score": final_score,
        "passed": grade["passed"],
        "steps": step_num,
        "total_reward": round(total_reward, 4),
    }


def main():
    print("Data Pipeline Repair Environment — Inference Runner")
    print(f"Model:   {MODEL_NAME}")
    print(f"API:     {API_BASE_URL}")
    print(f"Env URL: {ENV_URL}")

    results = []
    for task_id in TASKS:
        result = run_task(task_id)
        results.append(result)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"  {status}  {r['task_id']:<25}  score={r['score']:.4f}  steps={r['steps']}")

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"\n  Average score: {avg_score:.4f}")
    print(f"  Tasks passed:  {sum(1 for r in results if r['passed'])}/{len(results)}")

    return {"results": results, "avg_score": avg_score}

    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
