"""
test_env.py — deterministic unit tests for all 3 tasks.
No external services required. Tests graders, rewards, and state transitions.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from files.models import Action, ActionType, DifficultyLevel, ResetRequest
from files.environment import SessionManager


def make_manager() -> SessionManager:
    return SessionManager()


# ─────────────────────────────────────────────
# EASY TASK
# ─────────────────────────────────────────────
def test_easy_optimal_path():
    mgr = make_manager()
    sid, obs = mgr.reset(ResetRequest(task_id="easy_null_fix"))
    assert obs.pipeline_stage.value == "validate"
    assert not obs.done

    # Step 1: identify issue
    r1 = mgr.step(sid, Action(action_type=ActionType.identify_issue, target="age"))
    assert r1.reward > 0, f"Expected positive reward, got {r1.reward}"

    # Step 2: fix null
    r2 = mgr.step(sid, Action(action_type=ActionType.fix_null, target="age", value="0"))
    assert r2.reward > 0

    # Step 3: validate
    r3 = mgr.step(sid, Action(action_type=ActionType.validate_pipeline))
    assert r3.done

    grade = mgr.grade(sid)
    assert grade["score"] >= 0.8, f"Score too low: {grade['score']}"
    assert grade["passed"]
    print(f"  ✅ easy_null_fix: score={grade['score']}")


def test_easy_wrong_action_penalty():
    mgr = make_manager()
    sid, _ = mgr.reset(ResetRequest(task_id="easy_null_fix"))

    # Validate before fixing anything — should be penalised
    r = mgr.step(sid, Action(action_type=ActionType.validate_pipeline))
    assert r.reward < 0, f"Expected negative reward for premature validate, got {r.reward}"
    print(f"  ✅ easy_null_fix wrong action penalty: {r.reward}")


def test_easy_grade_zero_on_no_steps():
    mgr = make_manager()
    sid, _ = mgr.reset(ResetRequest(task_id="easy_null_fix"))
    grade = mgr.grade(sid)
    assert grade["score"] == 0.0
    print(f"  ✅ easy_null_fix zero grade on no steps: {grade['score']}")


# ─────────────────────────────────────────────
# MEDIUM TASK
# ─────────────────────────────────────────────
def test_medium_optimal_path():
    mgr = make_manager()
    sid, obs = mgr.reset(ResetRequest(task_id="medium_type_dedup"))
    assert "TypeError" in obs.errors[0]

    actions = [
        Action(action_type=ActionType.identify_issue, target="salary"),
        Action(action_type=ActionType.fix_type, target="salary", value="float"),
        Action(action_type=ActionType.identify_issue, target="duplicates"),
        Action(action_type=ActionType.drop_duplicates),
        Action(action_type=ActionType.validate_pipeline),
    ]
    total_reward = 0.0
    for a in actions:
        r = mgr.step(sid, a)
        total_reward += r.reward

    grade = mgr.grade(sid)
    assert grade["score"] >= 0.8, f"Score too low: {grade['score']}"
    assert grade["passed"]
    print(f"  ✅ medium_type_dedup: score={grade['score']}, total_reward={total_reward:.2f}")


def test_medium_partial_fix_score():
    mgr = make_manager()
    sid, _ = mgr.reset(ResetRequest(task_id="medium_type_dedup"))

    # Only fix salary type, don't handle duplicates
    mgr.step(sid, Action(action_type=ActionType.identify_issue, target="salary"))
    mgr.step(sid, Action(action_type=ActionType.fix_type, target="salary", value="float"))

    grade = mgr.grade(sid)
    assert 0.0 < grade["score"] < 0.8, f"Expected partial score, got {grade['score']}"
    print(f"  ✅ medium partial fix score: {grade['score']}")


# ─────────────────────────────────────────────
# HARD TASK
# ─────────────────────────────────────────────
def test_hard_optimal_path():
    mgr = make_manager()
    sid, obs = mgr.reset(ResetRequest(task_id="hard_multi_stage"))
    assert len(obs.errors) == 3

    actions = [
        Action(action_type=ActionType.identify_issue, target="created_at"),
        Action(action_type=ActionType.fix_date_format, target="created_at"),
        Action(action_type=ActionType.identify_issue, target="region"),
        Action(action_type=ActionType.add_missing_column, target="region"),
        Action(action_type=ActionType.identify_issue, target="query"),
        Action(action_type=ActionType.rewrite_query),
        Action(action_type=ActionType.validate_pipeline),
    ]
    for a in actions:
        mgr.step(sid, a)

    grade = mgr.grade(sid)
    assert grade["score"] >= 0.8, f"Score too low: {grade['score']}"
    assert grade["passed"]
    print(f"  ✅ hard_multi_stage: score={grade['score']}")


def test_hard_errors_clear_as_fixed():
    mgr = make_manager()
    sid, obs = mgr.reset(ResetRequest(task_id="hard_multi_stage"))
    initial_errors = len(obs.errors)

    mgr.step(sid, Action(action_type=ActionType.identify_issue, target="created_at"))
    r = mgr.step(sid, Action(action_type=ActionType.fix_date_format, target="created_at"))
    assert len(r.observation.errors) < initial_errors, "Errors should decrease after fix"
    print(f"  ✅ hard errors clear: {initial_errors} → {len(r.observation.errors)}")


def test_determinism():
    """Same actions always produce same scores."""
    for _ in range(3):
        mgr = make_manager()
        sid, _ = mgr.reset(ResetRequest(task_id="easy_null_fix"))
        mgr.step(sid, Action(action_type=ActionType.identify_issue, target="age"))
        mgr.step(sid, Action(action_type=ActionType.fix_null, target="age", value="0"))
        mgr.step(sid, Action(action_type=ActionType.validate_pipeline))
        grade = mgr.grade(sid)
        assert grade["score"] == 1.0, f"Non-deterministic score: {grade['score']}"
    print("  ✅ Grader is deterministic across 3 runs")


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        test_easy_optimal_path,
        test_easy_wrong_action_penalty,
        test_easy_grade_zero_on_no_steps,
        test_medium_optimal_path,
        test_medium_partial_fix_score,
        test_hard_optimal_path,
        test_hard_errors_clear_as_fixed,
        test_determinism,
    ]

    passed = 0
    failed = 0
    print("\n🧪 Running environment tests...\n")
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"  Passed: {passed}/{len(tests)}")
    if failed:
        print(f"  Failed: {failed}")
        sys.exit(1)
    else:
        print("  All tests passed ✅")
