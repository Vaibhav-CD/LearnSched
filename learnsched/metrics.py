"""Scheduling performance metrics.

Fairness definition (used everywhere: reward, tuning, ranking):
Jain's index over slowdown-normalised service, ``x_i = cpu_i / turnaround_i``
(= 1 / slowdown_i). 1.0 means every process received the same CPU share per
unit of turnaround; low values mean some processes were starved relative to
others. The single composite score below is how "best" is judged, so latency
and fairness are never traded off implicitly in separate places.
"""
from __future__ import annotations

import numpy as np

# Composite "best" score (lower is better). One full Jain point (0 -> 1) is
# worth 100 waiting ticks (~a third of typical avg waiting), so a 0.10
# fairness gain offsets 10 ticks of avg waiting. Latency part matches the
# historical validation objective (waiting + 0.5 * response) so old and new
# rankings stay comparable.
FAIRNESS_SCORE_WEIGHT = 100.0


def jain_index(values) -> float:
    """Jain's fairness index of a non-negative vector, in [0, 1]."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    if x.size == 1:
        return 1.0
    denom = x.size * float(np.sum(x ** 2))
    return float((x.sum() ** 2) / denom) if denom > 0 else 0.0


def fairness_values(procs) -> np.ndarray:
    """Completed-process fairness inputs: x_i = total_cpu_i / turnaround_i."""
    return np.array([p.total_cpu / max(p.turnaround, 1e-9)
                     for p in procs if p.completion is not None], dtype=float)


def current_jain(processes, now: float) -> float:
    """Observable Jain index so far, for the per-window reward.

    Uses only arrived processes and service already received
    (``cpu_received / elapsed_since_arrival``), so it needs no future
    information. A single arrived process is trivially fair (1.0).
    """
    vals = []
    for p in processes:
        if p.arrival > now:
            continue
        end = p.completion if p.completion is not None else now
        elapsed = end - p.arrival
        if elapsed <= 0:
            continue
        received = p.total_cpu if p.completion is not None else float(getattr(p, "cpu_received", 0.0))
        vals.append(received / elapsed)
    return jain_index(vals) if len(vals) > 1 else 1.0


def composite_score(metrics: dict) -> float:
    """Single latency+fairness objective (lower is better)."""
    return float(metrics.get("avg_waiting", 0.0)
                 + 0.5 * metrics.get("avg_response", 0.0)
                 - FAIRNESS_SCORE_WEIGHT * metrics.get("fairness_jain", 0.0))


def compute_metrics(result) -> dict:
    procs = [p for p in result.processes if p.completion is not None]
    if not procs:
        return {}
    n = len(procs)
    resp = np.array([p.response for p in procs], dtype=float)
    wait = np.array([p.waiting for p in procs], dtype=float)
    tat = np.array([p.turnaround for p in procs], dtype=float)
    makespan = max(float(result.makespan), 1.0)
    slowdown = tat / np.array([max(p.total_cpu, 1e-9) for p in procs])
    out = {
        "n_processes": n,
        "avg_response": float(resp.mean()),
        "p95_response": float(np.percentile(resp, 95)),
        "max_response": float(resp.max()),
        "avg_waiting": float(wait.mean()),
        "p95_waiting": float(np.percentile(wait, 95)),
        "avg_turnaround": float(tat.mean()),
        "avg_slowdown": float(slowdown.mean()),
        "p95_slowdown": float(np.percentile(slowdown, 95)),
        "max_slowdown": float(slowdown.max()),
        "throughput": float(n / makespan),
        "makespan": float(makespan),
        "cpu_utilization": float(sum(p.total_cpu for p in procs) / makespan),
        "fairness_jain": jain_index(fairness_values(procs)),
        "context_switches": int(result.context_switches),
        "overhead_time": float(result.overhead_time),
        "decision_overhead_s": float(getattr(result, "decision_overhead", 0.0)),
    }
    out["composite_score"] = composite_score(out)
    return out


def percentile(values, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if len(values) else 0.0
