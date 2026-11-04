"""Workload reproducibility, Gymnasium env contract, reward sanity."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from learnsched import workloads
from learnsched.env import MLFQTuningEnv, ACTION_CONFIGS


@pytest.mark.parametrize("scenario", workloads.SCENARIOS)
def test_workload_reproducible(scenario):
    a = workloads.generate(scenario, n=20, seed=7)
    b = workloads.generate(scenario, n=20, seed=7)
    assert [(p.arrival, p.cpu_bursts, p.io_bursts) for p in a] == \
           [(p.arrival, p.cpu_bursts, p.io_bursts) for p in b]


def test_different_seeds_differ():
    a = workloads.generate("mixed", n=20, seed=1)
    b = workloads.generate("mixed", n=20, seed=2)
    assert [p.cpu_bursts for p in a] != [p.cpu_bursts for p in b]


def test_trace_roundtrip(tmp_path):
    procs = workloads.generate("io_intensive", n=15, seed=3)
    path = str(tmp_path / "t.csv")
    workloads.save_trace(procs, path)
    loaded = workloads.load_trace(path)
    assert [(p.pid, p.arrival, p.cpu_bursts) for p in loaded] == \
           [(p.pid, p.arrival, p.cpu_bursts) for p in procs]


def test_env_check():
    check_env(MLFQTuningEnv(n_processes=15), skip_render_check=True)


def test_env_episode_terminates_and_reports_metrics():
    env = MLFQTuningEnv(n_processes=15)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    done, info = False, {}
    steps = 0
    rng = np.random.default_rng(0)
    while not done:
        obs, reward, terminated, truncated, info = env.step(int(rng.integers(len(ACTION_CONFIGS))))
        assert np.isfinite(reward)
        done = terminated or truncated
        steps += 1
        assert steps < 5000
    assert info["metrics"]["n_processes"] == 15
    assert info["metrics"]["avg_waiting"] >= 0


def test_env_no_future_leak_in_obs():
    # observation dimension is fixed and bounded regardless of trace length
    env = MLFQTuningEnv(n_processes=30)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (10,) and np.all((0 <= obs) & (obs <= 1))
