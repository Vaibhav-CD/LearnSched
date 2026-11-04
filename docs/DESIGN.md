# LearnSched — Design

**Research question:** Can a PPO agent dynamically tune MLFQ parameters to
improve CPU scheduling under changing workloads vs traditional schedulers?

## Workflow

```
Workload → CPU Simulator → PPO Controller → MLFQ Parameter Adjustment
        → Scheduling Performance → Reward → PPO Learning
```

Every `decision_interval` ticks (default 20), the controller observes the
simulator, picks one valid MLFQ configuration, and the simulator runs with
it for the next interval. The agent never dispatches individual processes.

## Modules

| Module | Responsibility |
|---|---|
| [learnsched/process.py](/learnsched/process.py) | Process model: arrival, CPU/I/O bursts, priority, timing fields |
| [learnsched/simulator.py](/learnsched/simulator.py) | Tick simulator: FCFS, SJF, RR, MLFQ; CS overhead; stepwise control |
| [learnsched/workloads.py](/learnsched/workloads.py) | 6 reproducible scenario generators; trace save/load |
| [learnsched/env.py](/learnsched/env.py) | Gymnasium MDP environment (see MDP.md) |
| [learnsched/train.py](/learnsched/train.py) | SB3 PPO training, checkpoints, reward logging |
| [learnsched/baselines.py](/learnsched/baselines.py) | Fixed/tuned MLFQ, rule-based adaptive MLFQ |
| [learnsched/experiments.py](/learnsched/experiments.py) | Identical-trace comparisons, validation-only tuning, ablation |
| [learnsched/metrics.py](/learnsched/metrics.py) | Response/waiting/turnaround/throughput/fairness/CS metrics |
| [dashboard/app.py](/dashboard/app.py) | Streamlit research dashboard |

## Assumptions

- Single core, single CPU, tick = 1 time unit; all bursts are integers.
- Context-switch overhead is a fixed number of ticks per dispatch to a
  different process, identical for every scheduler.
- SJF is non-preemptive and uses the true next CPU burst (standard textbook
  assumption; flagged as a limitation — real schedulers cannot know bursts).
- I/O processes re-enter MLFQ at their previous queue level (they yielded
  voluntarily); CPU-bound processes that exhaust a quantum are demoted.
- Priority boost periodically returns all ready processes to the top queue;
  together with demotion this is the starvation-prevention mechanism.
- This is a simulator, **not** Linux CFS/EEVDF; no claim is made about the
  real kernel scheduler.
- The PPO controller sees only observable state (queue length, utilisation,
  completed-process statistics, its own current parameters) — never future
  arrivals or unobserved burst lengths of running processes beyond the
  current burst statistics of incomplete processes.
