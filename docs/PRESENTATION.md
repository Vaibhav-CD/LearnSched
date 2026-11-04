# LearnSched: Workload-Adaptive CPU Scheduling Using PPO
## Research Presentation & Slide Deck

---

### Slide 1: Title & Overview
**LearnSched: Workload-Adaptive CPU Scheduling Using PPO**
*Can an RL agent dynamically tune MLFQ parameters to improve scheduling under changing workloads?*

- **Domain:** Operating Systems (CPU Scheduling) + Reinforcement Learning (PPO)
- **Framework:** Gymnasium-compatible tick simulator + Stable-Baselines3
- **Evaluation:** 144 identical-trace benchmark runs across 6 distinct workload patterns
- **Key finding:** Through direct latency-aligned reward formulation (v3), PPO matches validation-tuned MLFQ across all workloads without per-scenario hand-tuning.

---

### Slide 2: Operating System Scheduling Foundations
**Why CPU Scheduling Matters & The Classical Dilemma**

1. **Process Lifecycle & States:**
   - **Ready:** Waiting in ready queues for CPU allocation.
   - **Running:** Actively executing on the core.
   - **Blocked:** Suspended awaiting I/O completion.
2. **Key Evaluation Metrics:**
   - **Response Time ($T_{\text{first\_run}} - T_{\text{arrival}}$):** Latency to start interactive jobs.
   - **Waiting Time ($T_{\text{completion}} - T_{\text{arrival}} - T_{\text{burst}}$):** Total queue delay.
   - **Turnaround Time ($T_{\text{completion}} - T_{\text{arrival}}$):** Total completion span.
   - **Context Switch Overhead:** Cost of saving/restoring register states and cache invalidation.
3. **Traditional Baseline Schedulers:**
   - **FCFS (First-Come, First-Served):** Non-preemptive, suffers from the *convoy effect*.
   - **SJF (Shortest Job First):** Provably optimal mean waiting time, but requires impossible future burst knowledge and starves long jobs.
   - **Round Robin (RR):** Fixed time quantum. Small quantum = high context-switch cost; large quantum = degenerates to FCFS.
   - **Multi-Level Feedback Queue (MLFQ):** Priority levels $Q_0, Q_1, \dots$ with demotion on quantum expiration and periodic priority boost to prevent starvation. Approximates SJF without prior knowledge.
4. **The Dilemma:**
   - Static parameters (quanta per queue, boost interval) cannot simultaneously optimize for bursty I/O jobs and long compute-bound jobs when workloads drift.

---

### Slide 3: Reinforcement Learning Architecture
**Formulating CPU Scheduling as a Markov Decision Process (MDP)**

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$$

- **Hierarchical Design:**
  - *Fast path (per-tick):* MLFQ performs standard low-latency dispatching.
  - *Slow path (every 20 ticks):* PPO agent observes macro-state and updates MLFQ configuration parameters.
- **State Space $\mathcal{S} \in \mathbb{R}^{10}$ (normalized $[0, 1]$):**
  1. Ready queue length $/ 20$
  2. Recent CPU utilization ($\text{busy ticks} / \text{window ticks}$)
  3. Mean waiting time of completed jobs $/ 100$
  4. Overall completion progress
  5. Mean current burst $/ 30$
  6. Standard deviation of current burst $/ 30$
  7. Current $Q_0$ quantum $/ 8$
  8. Current boost interval $/ 100$
  9. Elapsed time / max ticks
  10. I/O share (fraction of processes with I/O phases)
- **Action Space $\mathcal{A}$:** Discrete set of MLFQ configuration tuples `(quanta, boost_interval)`.
- **Policy Network:**
  - SB3 PPO with 2-layer MLP $[64, 64]$
  - Discount $\gamma = 0.99$, GAE parameter $\lambda = 0.95$, clipping parameter $\epsilon = 0.20$.

---

### Slide 4: Reward Iteration & The Methodological Breakthrough
**The Pitfall of Misaligned Rewards in Systems RL**

- **Reward v1 / v2 Failure Mode:**
  - Formula: $r = 1.0 \cdot \text{comp} - 0.02 \cdot \text{wait} - 0.05 \cdot \text{cs} - 0.10 \cdot \text{fairness\_pen}$
  - **Result:** The agent learned to fear context switches and exploit completions of long jobs, collapsing to maximum quanta ($q=8/16/32$, boost 50).
  - Average response time degraded to **74.8 ticks**!
- **Diagnostic Method:**
  - Scored every fixed action directly with the reward function. Discovered that the reward function itself ranked bad-latency configurations highest!
- **Reward v3 (Ground-Truth Latency Tracking):**
  - Built directly from accrued waiting ticks and first-response waiting ticks:
  $$r = -0.10 \cdot \left(\frac{\text{wait\_window}}{n}\right) - 0.25 \cdot \left(\frac{\text{response\_wait\_window}}{n}\right) - 0.01 \cdot \text{cs} + 0.10 \cdot \text{completed}$$
  - Expanded action space to include short-quantum, zero-boost configurations.
  - **Result:** Average response time plummeted from **74.8 &rarr; 4.2 ticks**!

---

### Slide 5: Empirical Benchmark Comparisons
**Mean Results Across 6 Scenarios $\times$ 3 Seeds (144 Identical-Trace Runs)**

| Scheduler | Avg Response $\downarrow$ | Avg Waiting $\downarrow$ | Jain Fairness $\uparrow$ | Context Switches $\downarrow$ |
|---|:---:|:---:|:---:|:---:|
| **PPO Trained (Final v3)** | **4.2** | **286.5** | 0.51 | **180** |
| **MLFQ Tuned (Grid Search)** | **4.0** | **286.3** | 0.59 | 186 |
| **SJF (Theoretical Oracle)** | 147.5 | **166.2** | 0.54 | 97 |
| **PPO Untrained** | 8.5 | 306.6 | 0.73 | 189 |
| **Rule-Based Adaptive** | 11.2 | 343.5 | 0.69 | 598 *(thrashing)* |
| **MLFQ Fixed Default** | 40.1 | 356.7 | 0.68 | 217 |
| **Round Robin (RR)** | 52.3 | 360.5 | 0.65 | 222 |
| **FCFS** | 196.4 | 297.6 | 0.37 | 97 |

- PPO matches the validation-tuned MLFQ within $\pm 0.2$ ticks on response and waiting.
- Achieves 6 fewer context switches (180 vs 186) than tuned MLFQ and 418 fewer than rule-based MLFQ.

---

### Slide 6: Systems & Machine Learning Trade-Offs

1. **Short vs. Long Quanta Trade-Off:**
   - Short quanta ($q=1, 2, 4$) provide near-instantaneous response times (4.0–4.2 ticks) for interactive jobs, but increase total context switch overhead.
   - Long quanta minimize switching overhead but delay newly arrived processes.
2. **Fairness vs. Response Time Trade-Off:**
   - PPO achieved lower wait and response times by actively prioritizing short bursts, causing lower Jain fairness on bursty workloads (0.51 vs 0.59).
3. **Automated Adaptation vs. Manual Tuning:**
   - Tuned MLFQ required an offline validation grid search over 16 parameter pairs across multiple seeds.
   - PPO converged to this optimal frontier automatically without workload-specific tuning.
4. **Decision Interval vs. Inference Overhead:**
   - Decision interval of 20 ticks incurs negligible runtime overhead ($\approx 7.2$ ms across an entire simulation) while preventing policy instability.

---

### Slide 7: Iterative Improvements Summary

- **~18x Response Time Speedup:** Dropped from 74.8 ticks (v1) to 4.2 ticks (v3).
- **Waiting Time Reduction:** Accrued wait dropped from 370.0 to 286.5 ticks (beating default fixed MLFQ by 70 ticks and RR by 74 ticks).
- **Stabilized Switching:** Prevented oscillatory switching thrashing (180 vs 598 for rule-based).
- **Tooling & Visualization:** Refactored Streamlit Gantt chart to discrete numeric-tick ranges, eliminating timeline rendering artifacts.

---

### Slide 8: Conclusions & Research Insights

1. **Feasibility:** PPO successfully learns an online parameter-adaptation policy that matches offline grid-tuned MLFQ on latency.
2. **Core Research Takeaway:** Reward alignment matters far more than policy complexity or training duration. The diagnostic of scoring fixed actions against candidate rewards is an essential practice for systems RL.
3. **Honesty in Systems ML:** RL does not surpass the theoretical burst-oracle minimum of SJF; rather, it automates finding the optimal operational trade-off point between response time, throughput, and switching overhead under uncertain workloads.
