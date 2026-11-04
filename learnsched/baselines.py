"""Baseline schedulers compared on identical traces.

Includes a rule-based adaptive MLFQ (no learning): it inspects only
currently observable state (ready-queue length, recent burst behaviour)
and switches quantum/boost by fixed thresholds. This is the honest
non-RL adaptive competitor for the PPO controller.
"""
from __future__ import annotations

from .env import ACTION_CONFIGS
from .simulator import MLFQConfig, Simulator

# Fixed MLFQ used everywhere as the reference configuration.
FIXED_MLFQ = MLFQConfig((4, 8, 16, float("inf")), 50)
# "Tuned" MLFQ: selected on validation traces (see experiments.tune_fixed_mlfq),
# never on test traces. Default below is replaced by the tuned winner when
# experiments run; the value here is a documented starting point.
TUNED_MLFQ = MLFQConfig((2, 4, 8, float("inf")), 50)


def run_scheduler(processes, algorithm, **kw):
    return Simulator(processes, algorithm=algorithm, **kw).run()


def run_rule_based(processes, decision_interval=20, cs_overhead=0.0):
    """Adaptive by rule: short quanta when many short jobs wait, long
    quanta when the queue is dominated by long jobs, frequent boosts
    when the queue grows (starvation risk)."""
    sim = Simulator(processes, algorithm="mlfq", mlfq=FIXED_MLFQ, cs_overhead=cs_overhead)
    while not sim.done:
        ready = sim.ready_count()
        if ready >= 10:
            cfg = ACTION_CONFIGS[6]   # short quanta, frequent boost
        elif ready >= 5:
            cfg = ACTION_CONFIGS[0]   # medium quanta, frequent boost
        elif ready >= 2:
            cfg = ACTION_CONFIGS[3]   # default quanta
        else:
            cfg = ACTION_CONFIGS[5]   # long quanta, low overhead
        sim.set_mlfq_config(cfg)
        sim.step_ticks(decision_interval)
    from .simulator import SimResult
    result = SimResult(processes=sim.procs, timeline=sim.timeline,
                       context_switches=sim.context_switches, makespan=sim.time,
                       overhead_time=sim.overhead_time,
                       config_history=list(sim.config_history))
    return result
