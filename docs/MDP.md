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

`Discrete(10)`: an index into 10 valid MLFQ configurations
(`learnsched.env.ACTION_CONFIGS`) spanning short/long quanta
(1–32 ticks across queue levels) and boost intervals (25/50/100/off).
The agent tunes *parameters*; per-tick dispatch stays inside MLFQ.

## Transition $P$

The simulator advances `decision_interval` ticks under the chosen
configuration. Arrivals and I/O completions during the window are
stochastic from the agent's viewpoint (workload-generated).

## Reward $R$ (v4, fairness-aware)

Per decision window (`learnsched.env`):

$$r = -0.10 \cdot \frac{\text{wait ticks}}{n}
     - 0.25 \cdot \frac{\text{response-wait ticks}}{n}
     - 0.01 \cdot (\text{context switches})
     + 0.10 \cdot (\text{completions})
     - 0.20 \cdot (1 - J_{\text{window}})$$

- The waiting terms are accrued waiting time itself, so the episodic return
  tracks the reported latency metrics (sum of wait ticks / $n$ = avg waiting).
- $J_{\text{window}}$ is Jain's fairness index over
  `cpu_received / elapsed_since_arrival` for arrived processes
  (`metrics.current_jain`) — the same cpu/turnaround definition as the
  episodic `fairness_jain`, from observable state only. It is a bounded
  penalty, never a bonus, so prolonging an episode cannot farm fairness.
- `W_FAIR = 0.20` is calibrated to the ranking score
  (`metrics.composite_score`: waiting + 0.5·response − 100·Jain): episodic
  fairness pressure ≈ 0.20 × ~45 windows against 0.10 per waiting tick,
  a ratio of ~90 ≈ 100. At 0.50 (ratio ~225) the policy bought +0.10 Jain
  for +51 ticks of waiting and scored worse overall.
- There is **no** bonus for the episode ending and no way to gain reward by
  dropping work: unfinished processes keep accruing waiting penalty,
  so starvation is punished, not rewarded.

## Discount $\gamma$

0.99 (PPO default in `learnsched.train.DEFAULT_HPARAMS`): the controller
values near-term latency most but still accounts for knock-on congestion.
