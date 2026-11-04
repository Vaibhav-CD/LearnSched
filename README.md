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
.\.venv\Scripts\python.exe -m pytest tests/ -q          # 23 unit tests
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

## Measured findings (50k-step model, seeds 0–2, 6 scenarios)

See [docs/REPORT.md](/docs/REPORT.md) for the full table. In short: PPO's
episodic reward improved during training, but the trained controller did
**not** beat the validation-tuned fixed MLFQ or SJF on average waiting
time. The learned policy collapsed toward long quanta (low context-switch
overhead, poor interactive response). This is reported, not hidden.

## Limitations

Single core; integer ticks; SJF uses true burst lengths; simulator is not
the Linux kernel scheduler; PPO trained for 50k steps on one seed in the
shipped run; reward weights are hand-chosen. Full list in the report.
