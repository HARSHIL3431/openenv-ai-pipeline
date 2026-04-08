"""
demo.py — Rule-based optimal agent. No LLM or API key required.

Runs all 3 tasks against the live server using the hardcoded optimal
action sequences, printing a full run trace and final scores.

Usage:
  # With server already running:
  python demo.py

  # Or start server inline (requires uvicorn):
    python -m uvicorn files.main:app --port 7860 &
  python demo.py
"""

import json
import math
import sys
import time
import requests

ENV_URL = "http://localhost:7860"


def clamp_score(score):
    try:
        score = float(score)
    except Exception:
        score = 0.5
    if not math.isfinite(score):
        return 0.5
    if score <= 0.0:
        return 0.5
    if score >= 1.0:
        return 0.5
    return score

# ─────────────────────────────────────────────
# Optimal action sequences per task
# ─────────────────────────────────────────────
OPTIMAL_PLANS = {
    "easy_null_fix": [
        {"action_type": "identify_issue",    "target": "age",        "value": None},
        {"action_type": "fix_null",          "target": "age",        "value": "0"},
        {"action_type": "validate_pipeline", "target": None,         "value": None},
    ],
    "medium_type_dedup": [
        {"action_type": "identify_issue",    "target": "salary",     "value": None},
        {"action_type": "fix_type",          "target": "salary",     "value": "float"},
        {"action_type": "identify_issue",    "target": "duplicates", "value": None},
        {"action_type": "drop_duplicates",   "target": None,         "value": None},
        {"action_type": "validate_pipeline", "target": None,         "value": None},
    ],
    "hard_multi_stage": [
        {"action_type": "identify_issue",    "target": "created_at", "value": None},
        {"action_type": "fix_date_format",   "target": "created_at", "value": None},
        {"action_type": "identify_issue",    "target": "region",     "value": None},
        {"action_type": "add_missing_column","target": "region",     "value": None},
        {"action_type": "identify_issue",    "target": "query",      "value": None},
        {"action_type": "rewrite_query",     "target": None,         "value": None},
        {"action_type": "validate_pipeline", "target": None,         "value": None},
    ],
}


def wait_for_server(max_tries: int = 10):
    for i in range(max_tries):
        try:
            r = requests.get(f"{ENV_URL}/health", timeout=2)
            if r.status_code == 200:
                print(f"✅ Server ready at {ENV_URL}")
                return True
        except Exception:
            pass
        print(f"  Waiting for server... ({i+1}/{max_tries})")
        time.sleep(1)
    print("❌ Server not reachable. Start it with: python -m uvicorn files.main:app --port 7860")
    return False


def run_task(task_id: str, plan: list) -> dict:
    bar = "─" * 55
    print(f"\n┌{bar}┐")
    print(f"│  TASK: {task_id:<46}│")
    print(f"└{bar}┘")

    # Reset
    r = requests.post(f"{ENV_URL}/reset", params={"task_id": task_id})
    r.raise_for_status()
    data = r.json()
    session_id = data["session_id"]
    obs = data["observation"]

    print(f"  Session : {session_id}")
    print(f"  Stage   : {obs['pipeline_stage']}")
    print(f"  Errors  : {obs['errors']}")
    print()

    total_reward = 0.0

    for i, action in enumerate(plan, 1):
        r = requests.post(f"{ENV_URL}/step", json={"session_id": session_id, "action": action})
        r.raise_for_status()
        result = r.json()

        reward  = result["reward"]
        done    = result["done"]
        obs     = result["observation"]
        total_reward += reward

        action_str = f"{action['action_type']}" + (f"({action['target']})" if action["target"] else "()")
        sign = "+" if reward > 0 else ""
        print(f"  Step {i:>2}: {action_str:<35}  reward={sign}{reward:.2f}  stage={obs['pipeline_stage']}")

        if obs["errors"]:
            print(f"          ↳ errors remaining: {obs['errors']}")

        if done:
            break

    # Grade
    r = requests.get(f"{ENV_URL}/grade", params={"session_id": session_id})
    r.raise_for_status()
    grade = r.json()

    status = "✅ PASS" if grade["passed"] else "❌ FAIL"
    print()
    print(f"  {status}  score={grade['score']:.4f}  total_reward={total_reward:.2f}")
    print(f"  Feedback: {grade['feedback']}")
    print(f"  Breakdown:")
    for k, v in grade["breakdown"].items():
        tick = "✓" if v == 1.0 else "✗"
        print(f"    {tick} {k}: {v}")

    return {
        "task_id": task_id,
        "score": clamp_score(grade["score"]),
        "passed": grade["passed"],
        "total_reward": round(total_reward, 4),
        "steps": len(plan),
    }


def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Data Pipeline Repair Environment — Demo Agent      ║")
    print("║   Rule-based optimal agent (no LLM required)         ║")
    print("╚══════════════════════════════════════════════════════╝")

    if not wait_for_server():
        sys.exit(1)

    # Show available tasks
    r = requests.get(f"{ENV_URL}/tasks")
    tasks_info = r.json()["tasks"]
    print(f"\n  Available tasks: {[t['task_id'] for t in tasks_info]}")

    results = []
    for task_id, plan in OPTIMAL_PLANS.items():
        result = run_task(task_id, plan)
        results.append(result)

    # Summary
    print(f"\n╔══════════════════════════════════════════════════════╗")
    print(f"║  FINAL SUMMARY                                       ║")
    print(f"╠══════════════════════════════════════════════════════╣")
    for r in results:
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"║  {status}  {r['task_id']:<25} score={r['score']:.4f}  ║")

    avg = sum(clamp_score(r["score"]) for r in results) / len(results)
    avg = clamp_score(avg)
    passed = sum(1 for r in results if r["passed"])
    print(f"╠══════════════════════════════════════════════════════╣")
    print(f"║  Average score : {avg:.4f}                              ║")
    print(f"║  Tasks passed  : {passed}/{len(results)}                                ║")
    print(f"╚══════════════════════════════════════════════════════╝")

    with open("demo_results.json", "w") as f:
        json.dump({"results": results, "avg_score": avg, "agent": "rule-based-optimal"}, f, indent=2)
    print("\n  Results saved → demo_results.json")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
