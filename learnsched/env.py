"""Gymnasium environment: PPO tunes MLFQ parameters, not per-tick dispatch.

MDP  M = (S, A, P, R, gamma)
  S: 10 observable, normalised features (no future information).
  A: Discrete(K) -- index into ACTION_CONFIGS (valid MLFQ configurations).
  P: the CPU simulator advanced ``decision_interval`` ticks per step.
  R: completions rewarded; ready-queue waiting, context switches and
     fairness penalised (see step()). No reward for merely ending the
     episode, so premature termination / starvation is not rewarded.
  gamma: handled by PPO (default 0.99 in training config).
"""
from __future__ import annotations

import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .simulator import MLFQConfig, Simulator
from . import workloads

# Valid MLFQ configurations the agent may select.
ACTION_CONFIGS: list[MLFQConfig] = [
    MLFQConfig((2, 4, 8, float("inf")), 25),
    MLFQConfig((2, 4, 8, float("inf")), 50),
    MLFQConfig((2, 4, 8, float("inf")), 100),
    MLFQConfig((4, 8, 16, float("inf")), 50),
    MLFQConfig((4, 8, 16, float("inf")), 100),
    MLFQConfig((8, 16, 32, float("inf")), 50),
    MLFQConfig((1, 2, 4, float("inf")), 25),
    MLFQConfig((8, 16, 32, float("inf")), 0),  # long quanta, no boost
    MLFQConfig((1, 2, 4, float("inf")), 0),    # short quanta, no boost (tuned winner family)
    MLFQConfig((2, 4, 8, float("inf")), 0),    # medium quanta, no boost
]

# Reward v4: v3 (built from accrued waiting time, so the episodic return
# tracks the reported latency metrics) plus a fairness penalty. Fairness is
# Jain's index over cpu_received/elapsed for arrived processes, computed
# from observable state only at the end of each window (see
# metrics.current_jain, same cpu/turnaround definition as the episodic
# fairness_jain). The penalty is bounded per window: -W_FAIR * (1 - jain),
# so a perfectly fair window costs 0 and prolonging an episode cannot farm
# fairness reward (there is no positive fairness bonus to collect).
#   sum over windows of wait_ticks/n == average waiting time.
# W_FAIR is calibrated to the composite score: episodic fairness pressure
# is W_FAIR * n_windows (~45) against W_WAIT per waiting tick, i.e. a ratio
# of ~90, matching FAIRNESS_SCORE_WEIGHT / 1.0 in metrics.composite_score.
# At 0.50 the ratio was ~225 and the trained policy bought +0.10 Jain for
# +51 ticks of waiting (worse composite) -- see docs/REPORT.md.
W_WAIT, W_RESP, W_CS, W_COMPLETE, W_FAIR = 0.10, 0.25, 0.01, 0.10, 0.20


class MLFQTuningEnv(gym.Env):
    """One episode = one workload trace scheduled to completion."""

    metadata = {"render_modes": []}

    def __init__(self, scenarios=("cpu_bound", "io_intensive", "mixed", "bursty"),
                 n_processes: int = 40, decision_interval: int = 20,
                 cs_overhead: float = 0.0, max_ticks: int = 20000,
                 seed: int = 0, fixed_trace=None, controller_overhead_s: float = 0.0):
        super().__init__()
        self.scenarios = tuple(scenarios)
        self.n_processes = n_processes
        self.decision_interval = int(decision_interval)
        self.cs_overhead = float(cs_overhead)
        self.max_ticks = int(max_ticks)
        self._seed = seed
        self.fixed_trace = fixed_trace
        self.controller_overhead_s = controller_overhead_s
        self.action_space = spaces.Discrete(len(ACTION_CONFIGS))
        # all features clipped to [0, 1]-ish ranges documented in docs/MDP.md
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)
        self.sim: Simulator | None = None
        self._episode = 0
        self.decision_time_s = 0.0
        self.current_action = 1

    # ------------------------------------------------------------------
    def _new_sim(self, seed: int) -> Simulator:
        if self.fixed_trace is not None:
            procs = self.fixed_trace
        else:
            scenario = self.scenarios[seed % len(self.scenarios)]
            procs = workloads.generate(scenario, n=self.n_processes, seed=seed)
        sim = Simulator(procs, algorithm="mlfq",
                        mlfq=ACTION_CONFIGS[self.current_action],
                        cs_overhead=self.cs_overhead)
        return sim

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ep_seed = int(seed if seed is not None else self._seed + self._episode)
        self._episode += 1
        self.current_action = 1
        self.sim = self._new_sim(ep_seed)
        self.decision_time_s = 0.0
        return self._obs(), {"scenario_seed": ep_seed}

    def _obs(self) -> np.ndarray:
        s = self.sim
        procs = s.procs
        completed = [p for p in procs if p.completion is not None]
        bursts = [p.cpu_bursts[p.burst_idx] for p in procs if p.completion is None] or [0.0]
        cfg = ACTION_CONFIGS[self.current_action]
        recent_wait = np.mean([p.waiting for p in completed]) if completed else 0.0
        obs = np.array([
            min(s.ready_count() / 20.0, 1.0),                       # ready queue length
            s.busy_ticks / max(s.window_ticks, 1.0),                # recent CPU utilisation
            min(recent_wait / 100.0, 1.0),                          # avg waiting (completed)
            min(len(completed) / max(len(procs), 1) , 1.0),         # completion progress
            min(float(np.mean(bursts)) / 30.0, 1.0),                # observed burst mean
            min(float(np.std(bursts)) / 30.0, 1.0),                 # observed burst spread
            min(cfg.quanta[0] / 8.0, 1.0),                          # current Q0 quantum
            min((cfg.boost_interval or 100.0) / 100.0, 1.0),        # current boost interval
            min(s.time / max(self.max_ticks, 1), 1.0),              # time progress
            min(sum(len(p.io_bursts) > 0 for p in procs) / max(len(procs), 1), 1.0),  # I/O share
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    def step(self, action):
        t0 = time.perf_counter()
        assert self.sim is not None, "call reset() first"
        self.current_action = int(action)
        self.sim.set_mlfq_config(ACTION_CONFIGS[self.current_action])
        done_before = {p.pid for p in self.sim.procs if p.completion is not None}
        self.sim.reset_window_stats()
        self.sim.step_ticks(self.decision_interval)
        self.decision_time_s += time.perf_counter() - t0 + self.controller_overhead_s

        s = self.sim
        wait_cost = s.wait_ticks_window / max(self.decision_interval, 1)  # avg ready length
        n = max(len(s.procs), 1)
        jain_window = s.current_jain()
        reward = (-W_WAIT * (s.wait_ticks_window / n)
                  - W_RESP * (s.response_wait_window / n)
                  - W_CS * s.cs_window
                  + W_COMPLETE * s.completed_window
                  - W_FAIR * (1.0 - jain_window))
        terminated = s.done
        truncated = (not terminated) and s.time >= self.max_ticks
        # NOTE: no wall-clock values in info (keeps step() deterministic
        # under Gymnasium's env checker); overhead is on the attribute.
        info = {"time": s.time, "completed": s.done_count,
                "config": ACTION_CONFIGS[self.current_action].label(),
                "wait_cost": float(wait_cost), "cs": s.cs_window,
                "jain_window": float(jain_window)}
        if terminated or truncated:
            from .simulator import SimResult
            result = SimResult(processes=s.procs, timeline=s.timeline,
                               context_switches=s.context_switches, makespan=s.time,
                               overhead_time=s.overhead_time,
                               config_history=list(s.config_history))
            result.decision_overhead = self.decision_time_s
            info["metrics"] = result.metrics()
            info["result"] = result
        return self._obs(), float(reward), terminated, truncated, info


def make_env(**kwargs):
    return MLFQTuningEnv(**kwargs)
