"""Scheduling performance metrics. Jain fairness uses turnaround-normalised service."""
from __future__ import annotations

import numpy as np


def compute_metrics(result) -> dict:
    procs = [p for p in result.processes if p.completion is not None]
    if not procs:
        return {}
    n = len(procs)
    resp = np.array([p.response for p in procs], dtype=float)
    wait = np.array([p.waiting for p in procs], dtype=float)
    tat = np.array([p.turnaround for p in procs], dtype=float)
    makespan = max(float(result.makespan), 1.0)
    # Jain's fairness index over slowdown-normalised service: x_i = cpu_i / tat_i
    x = np.array([p.total_cpu / max(p.turnaround, 1e-9) for p in procs])
    jain = float((x.sum() ** 2) / (n * np.sum(x ** 2))) if n else 0.0
    return {
        "n_processes": n,
        "avg_response": float(resp.mean()),
        "p95_response": float(np.percentile(resp, 95)),
        "max_response": float(resp.max()),
        "avg_waiting": float(wait.mean()),
        "p95_waiting": float(np.percentile(wait, 95)),
        "avg_turnaround": float(tat.mean()),
        "throughput": float(n / makespan),
        "makespan": float(makespan),
        "cpu_utilization": float(sum(p.total_cpu for p in procs) / makespan),
        "fairness_jain": jain,
        "context_switches": int(result.context_switches),
        "overhead_time": float(result.overhead_time),
        "decision_overhead_s": float(getattr(result, "decision_overhead", 0.0)),
    }


def percentile(values, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if len(values) else 0.0
