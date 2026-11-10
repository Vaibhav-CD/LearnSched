"""Unit tests for the evaluation harness and scheduler integration."""
from __future__ import annotations

import pytest

from learnsched import workloads
from learnsched.baselines import FIXED_MLFQ, TUNED_MLFQ, run_rule_based
from learnsched.metrics import composite_score, compute_metrics
from learnsched.simulator import Simulator
from learnsched.train import ppo_schedule
from scripts.evaluate import evaluate_scenario, run_one


def test_all_schedulers_execute_cleanly():
    """Verify that every supported scheduler runs without exception on a test trace."""
    procs = workloads.generate("mixed", n=8, seed=42)
    schedulers = ["sjf", "fcfs", "rr", "mlfq_fixed", "mlfq_tuned", "rule_adaptive", "ppo_untrained"]
    
    for s in schedulers:
        p_copy = workloads.generate("mixed", n=8, seed=42)
        m = run_one(s, p_copy)
        assert "composite_score" in m
        assert "avg_waiting" in m
        assert "avg_response" in m
        assert "fairness_jain" in m
        assert m["avg_waiting"] >= 0.0
        assert 0.0 <= m["fairness_jain"] <= 1.0


def test_composite_score_properties():
    """Verify composite score mathematical behavior (lower is better, fairness rewarded)."""
    # Baseline metrics
    base = {"avg_waiting": 100.0, "avg_response": 10.0, "fairness_jain": 0.50}
    c_base = composite_score(base)
    # Expected: 100 + 0.5*10 - 100*0.50 = 100 + 5 - 50 = 55.0
    assert pytest.approx(c_base, 0.01) == 55.0

    # Higher fairness must strictly improve (lower) the composite score
    more_fair = {"avg_waiting": 100.0, "avg_response": 10.0, "fairness_jain": 0.60}
    c_fair = composite_score(more_fair)
    assert c_fair < c_base
    assert pytest.approx(c_fair, 0.01) == 45.0  # 0.10 fairness gain = 10 point improvement

    # Lower latency must strictly improve (lower) the composite score
    less_wait = {"avg_waiting": 90.0, "avg_response": 10.0, "fairness_jain": 0.50}
    assert composite_score(less_wait) < c_base


def test_evaluate_scenario_structure():
    """Test the evaluate_scenario function returns structured, sorted results."""
    res = evaluate_scenario(
        "bursty",
        seed=1,
        n_procs=10,
        schedulers=[("sjf", "SJF"), ("fcfs", "FCFS"), ("rr", "Round Robin")],
    )
    assert len(res) == 3
    # Ensure results are sorted by composite score ascending
    for i in range(len(res) - 1):
        assert res[i]["composite"] <= res[i + 1]["composite"]
