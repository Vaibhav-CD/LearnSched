"""Fairness: Jain definition, composite ranking, reward and tuning wiring."""
import numpy as np
import pytest

from learnsched import workloads
from learnsched.env import MLFQTuningEnv, ACTION_CONFIGS, W_FAIR
from learnsched.experiments import _score
from learnsched.metrics import (FAIRNESS_SCORE_WEIGHT, composite_score,
                                current_jain, jain_index)
from learnsched.process import make_processes
from learnsched.simulator import Simulator


def test_jain_equal_service_is_one_and_starved_is_low():
    assert jain_index([0.5, 0.5, 0.5]) == pytest.approx(1.0)
    assert jain_index([1.0]) == pytest.approx(1.0)
    assert jain_index([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.25)
    assert jain_index([]) == 0.0


def test_metrics_fairness_and_slowdown_on_hand_case():
    # FCFS: P1 cpu5 tat5 (x=1), P2 cpu3 tat7 (x=3/7) -> unequal service.
    m = Simulator(make_processes([(0, 5), (1, 3)]), algorithm="fcfs").run().metrics()
    x = np.array([1.0, 3 / 7])
    assert m["fairness_jain"] == pytest.approx(float((x.sum() ** 2) / (2 * (x ** 2).sum())))
    assert m["max_slowdown"] == pytest.approx(7 / 3)
    assert m["composite_score"] == pytest.approx(
        m["avg_waiting"] + 0.5 * m["avg_response"]
        - FAIRNESS_SCORE_WEIGHT * m["fairness_jain"])


def test_composite_prefers_fairer_at_equal_latency():
    base = {"avg_waiting": 100.0, "avg_response": 10.0}
    fair = composite_score({**base, "fairness_jain": 0.9})
    unfair = composite_score({**base, "fairness_jain": 0.5})
    assert fair < unfair
    assert unfair - fair == pytest.approx(0.4 * FAIRNESS_SCORE_WEIGHT)
    # Tuning objective is the same composite, not a separate latency score.
    assert _score([{**base, "fairness_jain": 0.9}]) == pytest.approx(fair)


def test_current_jain_uses_received_service_only():
    procs = make_processes([(0, 10), (0, 10)])
    sim = Simulator(procs, algorithm="fcfs")
    sim.step_ticks(5)  # only P1 served so far: x = [5/5, 0/5]
    assert sim.current_jain() == pytest.approx(jain_index([1.0, 0.0]))
    assert 0.0 <= current_jain(sim.procs, sim.time) <= 1.0


def test_env_reward_penalises_unfair_window_and_reports_jain():
    assert W_FAIR > 0
    env = MLFQTuningEnv(fixed_trace=workloads.generate("mixed", n=15, seed=3))
    env.reset(seed=0)
    _, reward, _, _, info = env.step(0)
    assert np.isfinite(reward)
    assert 0.0 <= info["jain_window"] <= 1.0


def test_fixed_action_reward_winner_is_fairness_aware():
    # Every fixed action's episodic return must be computable and the best
    # return must not be the long-quanta/no-boost config that starves
    # interactive jobs on the mixed validation trace.
    trace = workloads.generate("mixed", n=20, seed=101)
    returns = []
    for a in range(len(ACTION_CONFIGS)):
        env = MLFQTuningEnv(fixed_trace=trace)
        env.reset(seed=0)
        total, done = 0.0, False
        while not done:
            _, r, term, trunc, info = env.step(a)
            total += r
            done = term or trunc
        returns.append(total)
        assert info["metrics"]["fairness_jain"] >= 0.0
    assert int(np.argmax(returns)) != 7  # not (8,16,32,inf), boost off
