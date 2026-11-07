# LearnSched: Workload-Adaptive CPU Scheduling Using PPO

**Research question:** Can a PPO-based RL agent dynamically tune MLFQ
scheduling parameters to improve CPU scheduling performance under changing
workloads compared with traditional scheduling approaches?

LearnSched is a single-core CPU scheduling simulator (FCFS, SJF, Round
Robin, MLFQ) plus a Gymnasium environment in which a standard
Stable-Baselines3 PPO agent selects MLFQ configurations (quanta, boost
interval) every 20 ticks. It does **not** invent a new RL algorithm, does
**not** model Linux CFS/EEVDF, and reports negative results honestly.

## Install

```powershell
python -m venv --system-site-packages .venv   # reuses a system torch/SB3 if present
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# fresh machine without system packages: python -m venv .venv ; then pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q          # 29 unit tests
.\.venv\Scripts\python.exe scripts/check_env.py          # Gymnasium env validation
.\.venv\Scripts\python.exe scripts/train_ppo.py --smoke  # 4k steps, ~10 s
.\.venv\Scripts\python.exe scripts/train_ppo.py --timesteps 50000
.\.venv\Scripts\python.exe scripts/run_experiments.py    # tuning + 144 comparison runs
.\.venv\Scripts\python.exe scripts/make_plots.py
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py   # http://localhost:8501
```

## Layout

- [learnsched/simulator.py](/learnsched/simulator.py) — tick simulator, 4 algorithms, CS overhead
- [learnsched/workloads.py](/learnsched/workloads.py) — 6 reproducible scenarios, trace CSVs
- [learnsched/env.py](/learnsched/env.py) — MDP environment (state/action/reward)
- [learnsched/train.py](/learnsched/train.py) — SB3 PPO training, checkpoints, reward log
- [learnsched/experiments.py](/learnsched/experiments.py) — identical-trace comparison, validation-only tuning, ablation
- [dashboard/app.py](/dashboard/app.py) — Streamlit research dashboard
- [docs/DESIGN.md](/docs/DESIGN.md), [docs/MDP.md](/docs/MDP.md), [docs/PPO.md](/docs/PPO.md), [docs/REPORT.md](/docs/REPORT.md)

## Measured findings (fairness-aware model, seeds 0–2, 6 scenarios)

See [docs/REPORT.md](/docs/REPORT.md) for the full table. Fairness (Jain
index over cpu/turnaround) is part of the objective: one composite score
(`avg_waiting + 0.5*avg_response - 100*fairness_jain`) drives the PPO
reward, validation tuning, and the final ranking. The fairness-aware PPO
(120k steps, validation-selected seed) is the best deployable scheduler:
waiting 285.0 (vs 286.3 tuned MLFQ), fairness 0.62 (vs 0.51 for the
latency-only PPO at the same waiting). SJF scores lower only as an
oracle bound using true burst lengths. Earlier negative results are
preserved in `results/experiments_v3_latency_only.csv`.

## Limitations

Single core; integer ticks; SJF uses true burst lengths; simulator is not
the Linux kernel scheduler; PPO trained for 50k steps on one seed in the
shipped run; reward weights are hand-chosen. Full list in the report.
