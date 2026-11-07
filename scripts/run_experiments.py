"""Tune fixed MLFQ on validation seeds, then run comparative experiments.

Usage: python scripts/run_experiments.py [--quick] [--model models/ppo_mlfq.zip]
"""
import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from learnsched.experiments import tune_fixed_mlfq, run_experiments, ablation, summarize, rank_schedulers

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--model", default="models/ppo_mlfq")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    tuned, val_table = tune_fixed_mlfq(seeds=(101,) if args.quick else (101, 102))
    val_table.to_csv("results/tuning_validation.csv", index=False)
    print(f"Tuned fixed MLFQ (validation only): {tuned.label()}")

    model = None
    mp = args.model if args.model.endswith(".zip") else args.model + ".zip"
    if os.path.exists(mp):
        from stable_baselines3 import PPO
        model = PPO.load(args.model)
        print(f"Loaded trained model: {mp}")
    else:
        print("No trained model found; ppo_trained will be omitted.")

    seeds = (0,) if args.quick else (0, 1, 2)
    df = run_experiments(seeds=seeds, n=args.n, model=model, tuned_cfg=tuned)
    print(df.groupby("scheduler")[["avg_waiting", "avg_response", "throughput",
                                   "fairness_jain", "context_switches"]].mean().round(3))
    ranking = rank_schedulers(df)
    print("\nComposite ranking (avg_waiting + 0.5*avg_response - 100*fairness_jain, lower is better):")
    print(ranking[["avg_waiting", "avg_response", "fairness_jain",
                   "composite_score", "oracle"]].round(3))
    best = ranking[~ranking["oracle"]].index[0]
    print(f"Best deployable scheduler: {best} (SJF is an oracle bound, not deployable)")
    ablation(model=model)
    print(f"Wrote results/experiments.csv ({len(df)} rows) and results/ablation.csv")
