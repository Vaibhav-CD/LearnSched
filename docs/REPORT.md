# LearnSched — Research Report

## 1. Abstract
We study whether a standard PPO agent can tune MLFQ parameters (time
quanta, priority-boost interval) online to improve scheduling under
changing workloads. A validated tick simulator, six reproducible workload
families, and an identical-trace comparison against FCFS, SJF, RR, fixed /
validation-tuned / rule-based MLFQ and an untrained PPO controller are
implemented. PPO's training reward improved, but the trained controller
did not outperform the best classical baselines on held-out traces.

## 2. Introduction
Fixed scheduling parameters are a compromise: short quanta favour
interactive response, long quanta favour throughput and low overhead.
Workloads change, so a controller that adapts parameters is attractive.

## 3. Problem statement
Learn a mapping from observable scheduler state to MLFQ configuration
that minimises latency and unfairness without starvation, generalising to
unseen workloads, at negligible decision overhead.

## 4. Existing methods
FCFS (simple, convoy effect), SJF (optimal mean waiting when bursts are
known, impractical and starvation-prone), RR (responsive, overhead scales
with 1/quantum), MLFQ (approximates SJF behaviour without burst knowledge,
needs parameter choices and boost/aging against starvation). Production
kernels (Linux CFS/EEVDF) are far richer; this project does not model them.

## 5. Why reinforcement learning?
The parameter-to-performance mapping is workload-dependent and hard to
derive analytically, while good/bad parameter choices are directly
measurable. RL fits sequential decisions under uncertainty. RL-based
scheduling is an existing research area; no novelty is claimed for it.

## 6. Proposed architecture
Workload → Simulator → PPO controller (every 20 ticks) → MLFQ config →
performance → reward → PPO update. See DESIGN.md.

## 7. Mathematical formulation
MDP $(S,A,P,R,\gamma)$ in MDP.md; $J(\pi)$, policy gradient and PPO
clipping in PPO.md.

## 8. PPO algorithm
Standard SB3 PPO, MLP [64,64], clip 0.2, GAE 0.95, γ=0.99, lr 3e-4.

## 9. Implementation
Python, NumPy, Pandas, Gymnasium, Stable-Baselines3/PyTorch, Streamlit.
Modules and assumptions: DESIGN.md.

## 10. Experimental methodology
Six scenarios (cpu_bound, io_intensive, mixed, bursty, changing,
unseen_mixed), 40 processes, seeds 0–2, identical saved trace CSV per
(scenario, seed) for all 8 schedulers (144 runs). Tuned MLFQ selected by
grid search on validation seeds 101–102 only. Ablation varies the PPO
decision interval (10/20/40). Metrics: response (avg, p95), waiting,
turnaround, throughput, Jain fairness, context switches, overheads.

## 11. Results (measured; results/experiments.csv, mean over scenarios×seeds)

Reward design was iterated after a diagnostic (see "Reward iteration"
below). Final model: reward v3, 10-configuration action space, 120k steps.

| Scheduler | Avg response | Avg waiting | Fairness | Context switches |
|---|---|---|---|---|
| SJF | 147.5 | 166.2 | 0.54 | 97 |
| MLFQ tuned (validation) | 4.0 | 286.3 | 0.59 | 186 |
| PPO trained (final) | 4.2 | 286.5 | 0.51 | 180 |
| PPO untrained | 8.5 | 306.6 | 0.73 | 189 |
| Rule-based adaptive | 11.2 | 343.5 | 0.69 | 598 |
| MLFQ fixed | 40.1 | 356.7 | 0.68 | 217 |
| RR | 52.3 | 360.5 | 0.65 | 222 |
| FCFS | 196.4 | 297.6 | 0.37 | 97 |

Per-scenario, trained PPO matches tuned MLFQ waiting within ±5 ticks in
all six scenarios (slightly better on bursty and io_intensive, slightly
worse on changing/mixed), with comparable response and p95 response
(14.3 vs 14.5) and fewer context switches (180 vs 186). Its fairness is
worse (0.51 vs 0.59), driven by bursty workloads (0.55 vs 0.91). SJF still
has the lowest waiting but catastrophic response/p95 (starvation).

**Reward iteration (the methodological finding):** v1/v2 rewards ranked
long quanta best *even on latency* when every fixed action was scored
directly — the agent was correctly optimising a misaligned reward and
collapsed to q=8/16/32. The final reward is built from accrued waiting
time itself (episodic return ≈ −(0.10·avg waiting + 0.25·avg response)),
verified to rank fixed configurations consistently with the reported
metrics before retraining. The tuned winner's configuration family
(short quanta, boost off) also had to be present in the action space.

## 12. Performance trade-offs
Short quanta: better response, more switches/overhead. Long quanta: the
reverse. SJF minimises waiting but needs unknowable burst lengths and has
the worst p95 response (starvation of long jobs). No scheduler dominates.

## 13. Limitations
Single core, integer ticks, no cache/affinity/multi-level memory effects;
SJF oracle bursts; hand-set reward weights; 50k training steps, one
training seed; discrete 8-configuration action space; Python simulator
speed makes millions of steps impractical; decision overhead measured as
wall-clock on the development laptop only.

## 14. Conclusions
With a reward aligned to the latency metrics and the tuned configuration
family inside its action space, PPO learns a policy that matches the
validation-tuned fixed MLFQ on waiting/response in every scenario —
without per-scenario hand tuning — at a fairness cost. It does not beat
SJF on waiting (SJF uses oracle burst knowledge) and does not dominate
the tuned MLFQ. The strongest finding is methodological: reward
alignment, checked by scoring fixed actions before training, determined
success or failure more than training length did.

## 15. Future work
Reward rebalance (normalised response-time term, per-phase credit),
longer/multi-seed training, finer action space, preemption-aware SJF
baseline (SRTF), and comparison under context-switch overhead > 0.
