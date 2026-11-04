"""Comparative experiments on identical traces.

Rules enforced here:
- every scheduler sees the SAME saved trace file for a (scenario, seed);
- the tuned fixed-MLFQ is selected on validation seeds only, then frozen
  before test seeds are evaluated;
- trained vs untrained PPO use the same architecture and action space;
- all numbers written to CSV come from actual Simulator runs.
"""
from __future__ import annotations

import itertools
import os
import time

import numpy as np
import pandas as pd

from . import workloads
from .baselines import FIXED_MLFQ, run_rule_based
from .simulator import MLFQConfig, Simulator
from .train import ppo_schedule

SCHEDULERS = ("fcfs", "sjf", "rr", "mlfq_fixed", "mlfq_tuned",
              "rule_adaptive", "ppo_untrained", "ppo_trained")
METRIC_COLS = ["avg_response", "p95_response", "avg_waiting", "avg_turnaround",
               "throughput", "fairness_jain", "context_switches",
               "overhead_time", "decision_overhead_s", "makespan"]


def candidate_configs():
    quanta_sets = [(2, 4, 8, float("inf")), (4, 8, 16, float("inf")),
                   (8, 16, 32, float("inf")), (1, 2, 4, float("inf"))]
    boosts = [25, 50, 100, 0]
    return [MLFQConfig(q, b) for q, b in itertools.product(quanta_sets, boosts)]


def _score(metrics_list):
    """Validation objective: waiting + response (lower is better)."""
    return float(np.mean([m["avg_waiting"] + 0.5 * m["avg_response"] for m in metrics_list]))


def tune_fixed_mlfq(scenarios=("cpu_bound", "io_intensive", "mixed"),
                    seeds=(101, 102), n=40):
    """Grid-search fixed MLFQ on validation traces only. Returns winner."""
    best, rows = None, []
    for cfg in candidate_configs():
        ms = []
        for sc, sd in itertools.product(scenarios, seeds):
            procs = workloads.generate(sc, n=n, seed=sd)
            ms.append(Simulator(procs, algorithm="mlfq", mlfq=cfg).run().metrics())
        s = _score(ms)
        rows.append({"config": cfg.label(), "val_score": s})
        if best is None or s < best[1]:
            best = (cfg, s)
    return best[0], pd.DataFrame(rows).sort_values("val_score")


def run_experiments(scenarios=("cpu_bound", "io_intensive", "mixed", "bursty",
                               "changing", "unseen_mixed"),
                    seeds=(0, 1, 2), n=40, model=None, out_csv="results/experiments.csv",
                    traces_dir="data/workloads", tuned_cfg=None, cs_overhead=0.0,
                    include_ppo=True):
    os.makedirs(traces_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    tuned_cfg = tuned_cfg or FIXED_MLFQ
    rows = []
    for scenario, seed in itertools.product(scenarios, seeds):
        procs = workloads.generate(scenario, n=n, seed=seed)
        trace_path = os.path.join(traces_dir, f"{scenario}_seed{seed}.csv")
        workloads.save_trace(procs, trace_path)
        procs = workloads.load_trace(trace_path)  # identical input for everyone

        runs = {
            "fcfs": lambda: Simulator(procs, algorithm="fcfs", cs_overhead=cs_overhead).run(),
            "sjf": lambda: Simulator(procs, algorithm="sjf", cs_overhead=cs_overhead).run(),
            "rr": lambda: Simulator(procs, algorithm="rr", rr_quantum=4, cs_overhead=cs_overhead).run(),
            "mlfq_fixed": lambda: Simulator(procs, algorithm="mlfq", mlfq=FIXED_MLFQ, cs_overhead=cs_overhead).run(),
            "mlfq_tuned": lambda: Simulator(procs, algorithm="mlfq", mlfq=tuned_cfg, cs_overhead=cs_overhead).run(),
            "rule_adaptive": lambda: run_rule_based(procs, cs_overhead=cs_overhead),
        }
        if include_ppo:
            runs["ppo_untrained"] = lambda: ppo_schedule(procs, model=None, cs_overhead=cs_overhead)
            if model is not None:
                runs["ppo_trained"] = lambda: ppo_schedule(procs, model=model, cs_overhead=cs_overhead)
        for name, fn in runs.items():
            t0 = time.time()
            result = fn()
            m = result.metrics()
            row = {"scenario": scenario, "seed": seed, "scheduler": name,
                   "trace": trace_path, "wall_time_s": time.time() - t0}
            row.update({c: m.get(c) for c in METRIC_COLS})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def ablation(decision_intervals=(10, 20, 40), scenarios=("mixed", "changing"),
             seeds=(0, 1), n=40, model=None, out_csv="results/ablation.csv"):
    """Ablation: how the PPO decision interval affects metrics/overhead."""
    rows = []
    for di, scenario, seed in itertools.product(decision_intervals, scenarios, seeds):
        procs = workloads.generate(scenario, n=n, seed=seed)
        result = ppo_schedule(procs, model=model, decision_interval=di)
        m = result.metrics()
        row = {"decision_interval": di, "scenario": scenario, "seed": seed,
               "controller": "ppo_trained" if model is not None else "ppo_untrained"}
        row.update({c: m.get(c) for c in METRIC_COLS})
        rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby(["scenario", "scheduler"])[METRIC_COLS]
              .agg(["mean", "std"]).round(4))
