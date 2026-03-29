---
title: Data Pipeline Repair Environment
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - data-engineering
  - llm-agent
  - pipeline-repair
pinned: false
---

# 🔧 Data Pipeline Repair Environment

> **OpenEnv-compliant RL environment** where an LLM agent diagnoses and repairs broken real-world data pipelines.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)](https://openenv.dev)
[![HuggingFace](https://img.shields.io/badge/🤗-Space-yellow)](https://huggingface.co/spaces)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](./Dockerfile)

---

## 🧠 Problem Description

Data pipelines break constantly in the real world — NULL values, wrong dtypes, duplicate rows, malformed dates, missing schema columns, broken SQL queries. Engineers spend hours debugging these one by one.

This environment simulates that exact scenario: the agent observes a broken pipeline's state (errors, logs, schema, data sample) and must select the correct sequence of structured repair actions to restore the pipeline to a working state.

**Why this is a strong environment:**
- Multi-step reasoning required (identify → fix → validate)
- Partial rewards at each step (not just final score)
- Deterministic grading
- Three well-separated difficulty levels
- Directly maps to work data engineers do every day

---

## 🏗️ Environment Design

### Architecture

```
Agent
  │
  ├── POST /reset   → get initial observation + session_id
  ├── POST /step    → take action, get (obs, reward, done)
  ├── GET  /state   → inspect current state
  └── GET  /grade   → deterministic score 0.0–1.0
```

### Session Lifecycle

```python
obs = env.reset(task_id="easy_null_fix")
session_id = obs["session_id"]

while not done:
    action = LLM(obs["observation"])
    result = env.step(session_id, action)
    obs, reward, done = result["observation"], result["reward"], result["done"]

score = env.grade(session_id)  # 0.0 → 1.0
```

### State Machine

```
INGEST → VALIDATE → TRANSFORM → LOAD → DONE
```
Pipeline stage advances as errors are resolved. `done=True` when all issues are fixed and `validate_pipeline` is called successfully.

---

## ⚙️ Action Space

All actions are structured JSON — deterministic, parseable, gradeable.

| Action Type | Target | Value | Description |
|---|---|---|---|
| `identify_issue` | column / "duplicates" / "query" | — | Diagnose a specific problem |
| `fix_null` | column name | fill value | Fill NULL values |
| `fix_type` | column name | type (e.g. "float") | Cast to correct dtype |
| `fix_schema` | column name | — | Fix schema definition |
| `drop_duplicates` | — | — | Remove duplicate rows |
| `rewrite_query` | — | — | Fix broken SQL query |
| `normalize_column` | column name | — | Normalise column values |
| `validate_pipeline` | — | — | Run final validation (**use last**) |
| `add_missing_column` | column name | — | Add column missing from schema |
| `fix_date_format` | column name | — | Standardise date formats |

**Example action:**
```json
{"action_type": "fix_null", "target": "age", "value": "0"}
```

---

## 👁️ Observation Space

```json
{
  "task_id": "easy_null_fix",
  "difficulty": "easy",
  "pipeline_stage": "validate",
  "errors": ["NullValueError: column 'age' contains 12 NULL values"],
  "warnings": ["Pipeline halted at VALIDATE stage"],
  "schema": {"user_id": "int", "name": "str", "age": "int", "email": "str"},
  "data_sample": {
    "columns": ["user_id", "name", "age", "email"],
    "rows": [[1, "Alice", null, "alice@example.com"], ...],
    "dtypes": {"user_id": "int64", "age": "float64", ...}
  },
  "logs": ["ERROR [validate] NullValueError in column 'age'"],
  "steps_taken": [],
  "steps_remaining": 5,
  "done": false,
  "reward_so_far": 0.0
}
```

---

## 🎯 Tasks

### 🟢 Easy — `easy_null_fix`
**Scenario:** The `age` column in a users table has 12 NULL values. The pipeline halted at the VALIDATE stage.

**Required steps:**
1. `identify_issue` on `age`
2. `fix_null` on `age`
3. `validate_pipeline`

**Max steps:** 5

---

### 🟡 Medium — `medium_type_dedup`
**Scenario:** The `salary` column is stored as `object` (string) instead of `float`, AND the dataset has 20 duplicate rows. Two independent issues to fix.

**Required steps:**
1. `identify_issue` on `salary`
2. `fix_type` on `salary` → `"float"`
3. `identify_issue` on `"duplicates"`
4. `drop_duplicates`
5. `validate_pipeline`

**Max steps:** 8

---

### 🔴 Hard — `hard_multi_stage`
**Scenario:** Three concurrent failures across two pipeline stages:
- Mixed date formats in `created_at` (MM/DD/YYYY vs YYYY-MM-DD)
- `region` column present in raw data but missing from schema
- Downstream SQL query references an undefined alias

**Required steps:**
1. `identify_issue` on `created_at` → `fix_date_format`
2. `identify_issue` on `region` → `add_missing_column`
3. `identify_issue` on `query` → `rewrite_query`
4. `validate_pipeline`

**Max steps:** 12

---

## 🎁 Reward Function

Rewards are partial — given at each correct step, not just at the end.

| Step | Easy | Medium | Hard |
|---|---|---|---|
| identify_issue (correct column) | +0.20 | +0.15 | +0.10 |
| apply correct fix | +0.50 | +0.25–0.30 | +0.20 |
| validate_pipeline (after all fixes) | +0.30 | +0.25 | +0.30 |
| wrong / premature action | −0.20 | −0.20 | −0.15 |

---

## 📊 Grader

All graders are **deterministic** — same steps always produce the same score.

| Criterion | Weight |
|---|---|
| Issue identification | 5–20% |
| Correct fix applied | 20–50% |
| Pipeline validated | 15–20% |
| Pipeline stage = DONE | 10% |

**Passing threshold:** score ≥ 0.8

---

## 📦 Baseline Scores

Tested with a simple rule-based agent that always follows the optimal path:

| Task | Score | Passed |
|---|---|---|
| easy_null_fix | 1.00 | ✅ |
| medium_type_dedup | 1.00 | ✅ |
| hard_multi_stage | 1.00 | ✅ |

Random-action baseline (shuffled actions, no strategy):

| Task | Score | Passed |
|---|---|---|
| easy_null_fix | ~0.30 | ❌ |
| medium_type_dedup | ~0.20 | ❌ |
| hard_multi_stage | ~0.10 | ❌ |

---

## 🚀 Setup

### Local (uv / pip)

```bash
# Clone / cd into project
cd pipeline_env

# Install
pip install -r requirements.txt

# Run server
python -m uvicorn files.main:app --host 0.0.0.0 --port 7860 --reload

# Open docs
open http://localhost:7860/docs
```

### Docker

```bash
docker build -t pipeline-env .
docker run -p 7860:7860 pipeline-env
```

### Run Inference (LLM agent)

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
export HF_TOKEN="hf_..."
export ENV_URL="http://localhost:7860"

python -m files.inference
```

---

## ☁️ Hugging Face Deployment

```bash
# Push to HF Spaces (Docker SDK)
openenv push --repo-id username/pipeline-repair-env

# Or manually:
huggingface-cli upload username/pipeline-repair-env . --repo-type space
```

Space configuration (`README.md` header for HF):
```yaml
---
title: Data Pipeline Repair Environment
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - data-engineering
  - llm-agent
---
```

---

## 📁 Project Structure

```
pipeline_env/
├── files/
│   ├── __init__.py
│   ├── main.py          # FastAPI server
│   ├── models.py        # Pydantic models (Action, Observation, etc.)
│   ├── environment.py   # Core env logic + session manager
│   ├── registry.py      # Task definitions + graders
│   ├── inference.py     # LLM agent runner
│   ├── openenv.yaml     # OpenEnv spec
│   ├── requirements.txt
│   ├── Dockerfile
│   └── README.md
```

---

## 🏷️ Tags

`openenv` `reinforcement-learning` `llm-agent` `data-engineering` `pipeline-repair` `fastapi` `pydantic`
