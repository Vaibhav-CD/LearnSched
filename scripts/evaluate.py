"""CLI Evaluation Tool for LearnSched.

Evaluate any or all schedulers across reproducible workload traces and print
a structured comparative performance leaderboard with the composite score.

Usage:
    python scripts/evaluate.py --scenario bursty --seed 0
    python scripts/evaluate.py --all --seed 0
    python scripts/evaluate.py --quick
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from learnsched import workloads
from learnsched.baselines import FIXED_MLFQ, TUNED_MLFQ, run_rule_based
from learnsched.simulator import Simulator
from learnsched.train import load_model, ppo_schedule

SCHEDULERS = [
    ("sjf", "Shortest Job First (Oracle)"),
    ("ppo_trained", "PPO Trained (Fairness v4)"),
    ("mlfq_tuned", "MLFQ Tuned (Validation)"),
    ("ppo_untrained", "PPO Untrained (Random)"),
    ("rule_adaptive", "Rule-Based Adaptive"),
    ("mlfq_fixed", "MLFQ Fixed (Default)"),
    ("rr", "Round Robin (RR, q=4)"),
    ("fcfs", "First-Come First-Served"),
]


def run_one(name: str, procs, model=None, cs_overhead: float = 0.0):
    t0 = time.perf_counter()
    if name == "ppo_trained":
        res = ppo_schedule(procs, model=model, cs_overhead=cs_overhead)
    elif name == "ppo_untrained":
        res = ppo_schedule(procs, model=None, cs_overhead=cs_overhead)
    elif name == "mlfq_tuned":
        res = Simulator(procs, algorithm="mlfq", cs_overhead=cs_overhead, mlfq=TUNED_MLFQ).run()
    elif name == "mlfq_fixed":
        res = Simulator(procs, algorithm="mlfq", cs_overhead=cs_overhead, mlfq=FIXED_MLFQ).run()
    elif name == "rule_adaptive":
        res = run_rule_based(procs, cs_overhead=cs_overhead)
    elif name == "rr":
        res = Simulator(procs, algorithm="rr", rr_quantum=4, cs_overhead=cs_overhead).run()
    elif name == "sjf":
        res = Simulator(procs, algorithm="sjf", cs_overhead=cs_overhead).run()
    elif name == "fcfs":
        res = Simulator(procs, algorithm="fcfs", cs_overhead=cs_overhead).run()
    else:
        raise ValueError(f"Unknown scheduler: {name}")
    wall_s = time.perf_counter() - t0
    m = res.metrics()
    m["wall_time_s"] = wall_s
    return m


def evaluate_scenario(scenario: str, seed: int = 0, n_procs: int = 40, cs_overhead: float = 0.0,
                      model=None, schedulers: list[tuple[str, str]] | None = None):
    scheds = schedulers or SCHEDULERS
    procs = workloads.generate(scenario, n=n_procs, seed=seed)
    
    results = []
    for s_key, s_label in scheds:
        # Clone process list for 100% identical arrivals
        p_copy = workloads.generate(scenario, n=n_procs, seed=seed)
        m = run_one(s_key, p_copy, model=model, cs_overhead=cs_overhead)
        results.append({
            "key": s_key,
            "label": s_label,
            "composite": m.get("composite_score", 0.0),
            "waiting": m.get("avg_waiting", 0.0),
            "response": m.get("avg_response", 0.0),
            "p95_resp": m.get("p95_response", 0.0),
            "fairness": m.get("fairness_jain", 0.0),
            "cs": m.get("context_switches", 0),
            "wall_ms": m.get("wall_time_s", 0.0) * 1000.0,
        })
    
    # Sort by composite score ascending (lower is better)
    results.sort(key=lambda r: r["composite"])
    return results


def print_table(results: list[dict], title: str):
    print("\n" + "=" * 90)
    print(f"  {title.upper()}")
    print("  Composite = avg_waiting + 0.5*avg_response - 100*fairness (Lower is Better)")
    print("=" * 90)
    header = f"{'Rank':<5} {'Scheduler':<32} {'Composite':<11} {'Waiting':<10} {'Response':<10} {'Fairness':<10} {'CS':<6}"
    print(header)
    print("-" * 90)
    for rank, r in enumerate(results, start=1):
        line = (f"{rank:<5} {r['label']:<32} "
                f"{r['composite']:<11.2f} "
                f"{r['waiting']:<10.2f} "
                f"{r['response']:<10.2f} "
                f"{r['fairness']:<10.3f} "
                f"{r['cs']:<6}")
        print(line)
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Evaluate LearnSched Schedulers")
    parser.add_argument("--scenario", default="bursty", choices=list(workloads.SCENARIOS) + ["all"],
                        help="Workload family to evaluate (default: bursty)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for trace generation (default: 0)")
    parser.add_argument("--procs", type=int, default=40, help="Number of processes (default: 40)")
    parser.add_argument("--cs-overhead", type=float, default=0.0, help="Context switch overhead in ticks (default: 0)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (15 processes)")
    parser.add_argument("--model-path", default="models/ppo_mlfq", help="Path to trained PPO model")
    args = parser.parse_args()

    n_procs = 15 if args.quick else args.procs
    model = None
    if os.path.exists(args.model_path + ".zip"):
        model = load_model(args.model_path)
    else:
        print(f"[Notice] Model {args.model_path}.zip not found; PPO trained will use random weights.")

    scenarios = workloads.SCENARIOS if args.scenario == "all" else [args.scenario]

    for scen in scenarios:
        title = f"Scenario: {scen} | Seed: {args.seed} | Processes: {n_procs}"
        res = evaluate_scenario(scen, seed=args.seed, n_procs=n_procs,
                                cs_overhead=args.cs_overhead, model=model)
        print_table(res, title)


if __name__ == "__main__":
    main()
