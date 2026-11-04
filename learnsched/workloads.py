"""Reproducible workload generation.

Six scenario families. Every generator takes an explicit seed; identical
(seed, scenario, n) always yields the identical trace, so all schedulers
are compared on exactly the same work. Generators only use information a
real workload would have -- the RL agent never sees future arrivals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .process import Process

SCENARIOS = ("cpu_bound", "io_intensive", "mixed", "bursty",
             "changing", "unseen_mixed")


def _poisson_arrivals(rng, n, rate):
    return np.cumsum(rng.exponential(1.0 / rate, size=n))


def _cpu_bound(rng, n):
    arrivals = _poisson_arrivals(rng, n, rate=0.12)
    return [Process(pid=i + 1, arrival=float(a),
                    cpu_bursts=[float(rng.integers(15, 45))])
            for i, a in enumerate(arrivals)]


def _io_intensive(rng, n):
    procs = []
    for i, a in enumerate(_poisson_arrivals(rng, n, rate=0.30)):
        k = int(rng.integers(3, 7))
        cpus = [float(rng.integers(1, 4)) for _ in range(k)]
        ios = [float(rng.integers(3, 12)) for _ in range(k - 1)]
        procs.append(Process(pid=i + 1, arrival=float(a), cpu_bursts=cpus, io_bursts=ios))
    return procs


def _mixed(rng, n, cpu_frac=0.5):
    procs = []
    for i, a in enumerate(_poisson_arrivals(rng, n, rate=0.20)):
        if rng.random() < cpu_frac:
            procs.append(Process(pid=i + 1, arrival=float(a),
                                 cpu_bursts=[float(rng.integers(10, 35))]))
        else:
            k = int(rng.integers(2, 6))
            procs.append(Process(pid=i + 1, arrival=float(a),
                                 cpu_bursts=[float(rng.integers(1, 5)) for _ in range(k)],
                                 io_bursts=[float(rng.integers(2, 10)) for _ in range(k - 1)]))
    return procs


def _bursty(rng, n):
    arrivals, t = [], 0.0
    while len(arrivals) < n:
        burst_n = int(rng.integers(5, 12))
        arrivals.extend(t + rng.uniform(0, 2, size=burst_n))
        t += float(rng.uniform(25, 60))  # idle gap between bursts
    arrivals = sorted(arrivals[:n])
    return [Process(pid=i + 1, arrival=float(a),
                    cpu_bursts=[float(rng.integers(4, 20))])
            for i, a in enumerate(arrivals)]


def _changing(rng, n):
    """Pattern switches mid-trace: CPU-bound phase -> I/O phase -> mixed."""
    third = max(1, n // 3)
    parts = [_cpu_bound(rng, third), _io_intensive(rng, third),
             _mixed(rng, n - 2 * third)]
    # shift each phase so phases are sequential in time, reassign pids
    out, offset = [], 0.0
    for part in parts:
        shift = offset - (part[0].arrival if part else 0.0)
        for p in part:
            p.arrival = float(p.arrival + shift)
            p.pid = len(out) + 1
            out.append(p)
        offset = max(p.arrival for p in part) + 20.0
    return out


_GENERATORS = {"cpu_bound": _cpu_bound, "io_intensive": _io_intensive,
               "mixed": _mixed, "bursty": _bursty, "changing": _changing}


def generate(scenario: str, n: int = 40, seed: int = 0) -> list[Process]:
    if scenario == "unseen_mixed":
        # held-out family: different distributions / arrival rate than training
        rng = np.random.default_rng(seed + 10_000)
        procs = []
        for i, a in enumerate(_poisson_arrivals(rng, n, rate=0.16)):
            r = rng.random()
            if r < 0.35:
                procs.append(Process(pid=i + 1, arrival=float(a),
                                     cpu_bursts=[float(rng.integers(20, 60))]))
            elif r < 0.75:
                k = int(rng.integers(4, 9))
                procs.append(Process(pid=i + 1, arrival=float(a),
                                     cpu_bursts=[float(rng.integers(1, 3)) for _ in range(k)],
                                     io_bursts=[float(rng.integers(5, 18)) for _ in range(k - 1)]))
            else:
                procs.append(Process(pid=i + 1, arrival=float(a),
                                     cpu_bursts=[float(rng.integers(5, 15)),
                                                 float(rng.integers(5, 15))],
                                     io_bursts=[float(rng.integers(4, 10))]))
        return procs
    if scenario not in _GENERATORS:
        raise ValueError(f"unknown scenario {scenario!r}, expected {SCENARIOS}")
    rng = np.random.default_rng(seed)
    procs = _GENERATORS[scenario](rng, n)
    return sorted(procs, key=lambda p: (p.arrival, p.pid))


def save_trace(processes: list[Process], path: str) -> str:
    pd.DataFrame([p.to_dict() for p in processes]).to_csv(path, index=False)
    return path


def load_trace(path: str) -> list[Process]:
    df = pd.read_csv(path, dtype=str).fillna("")
    return sorted((Process.from_dict(r) for _, r in df.iterrows()),
                  key=lambda p: (p.arrival, p.pid))
