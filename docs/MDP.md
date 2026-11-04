# MDP Formulation

$$\mathcal M = (S, A, P, R, \gamma)$$

## State $S$ (10 features, all in $[0,1]$, no future information)

| # | Feature | Meaning |
|---|---|---|
| 0 | ready length / 20 | how congested the ready queues are |
| 1 | recent CPU utilisation | busy ticks / ticks in the last window |
| 2 | mean waiting of completed / 100 | congestion history |
| 3 | completion progress | fraction of processes finished |
| 4 | mean current burst / 30 | observed burst scale of unfinished work |
| 5 | std of current burst / 30 | burst heterogeneity |
| 6 | current Q0 quantum / 8 | controller's own current setting |
| 7 | current boost interval / 100 | controller's own current setting |
| 8 | elapsed time / max ticks | episode progress |
| 9 | I/O share | fraction of processes that ever block on I/O |

## Action $A$

`Discrete(8)`: an index into 8 valid MLFQ configurations
(`learnsched.env.ACTION_CONFIGS`) spanning short/long quanta
(1–32 ticks across queue levels) and boost intervals (25/50/100/off).
The agent tunes *parameters*; per-tick dispatch stays inside MLFQ.

## Transition $P$

The simulator advances `decision_interval` ticks under the chosen
configuration. Arrivals and I/O completions during the window are
stochastic from the agent's viewpoint (workload-generated).

## Reward $R$

Per decision window:

$$r = 1.0 \cdot (\text{completions}) - 0.02 \cdot (\text{mean ready length})
      - 0.05 \cdot (\text{context switches}) - 0.10 \cdot (\text{waiting spread})$$

- Completions reward throughput; ready-length penalises latency build-up.
- Context switches penalise configurations that thrash (tiny quanta).
- Waiting spread (std/mean of completed waiting times, clipped) penalises
  unfair configurations.
- There is **no** bonus for the episode ending and no way to gain reward by
  dropping work: unfinished processes keep accruing ready-length penalty,
  so starvation is punished, not rewarded.

## Discount $\gamma$

0.99 (PPO default in `learnsched.train.DEFAULT_HPARAMS`): the controller
values near-term latency most but still accounts for knock-on congestion.
