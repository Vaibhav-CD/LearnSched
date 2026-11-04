"""LearnSched Research Dashboard.

Run from the project root:  streamlit run dashboard/app.py
All numbers come from real Simulator runs and the CSVs in results/.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from learnsched import workloads
from learnsched.baselines import FIXED_MLFQ, run_rule_based
from learnsched.env import ACTION_CONFIGS, MLFQTuningEnv
from learnsched.experiments import METRIC_COLS
from learnsched.simulator import MLFQConfig, Simulator

st.set_page_config(page_title="LearnSched Research Dashboard", layout="wide")
st.title("LearnSched Research Dashboard")
st.caption("PPO-based adaptive tuning of MLFQ CPU scheduling parameters. "
           "Simulation and experiment outputs are measured, not fabricated.")

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ppo_mlfq")


@st.cache_data
def load_csv(name):
    path = os.path.join(RESULTS, name)
    return pd.read_csv(path) if os.path.exists(path) else None


# ---------------------------------------------------------------- sidebar
scenario = st.sidebar.selectbox("Workload scenario", workloads.SCENARIOS)
n_processes = st.sidebar.slider("Processes", 10, 60, 30)
seed = st.sidebar.number_input("Seed", value=0, step=1)
algo = st.sidebar.selectbox("Algorithm", ["fcfs", "sjf", "rr", "mlfq", "rule_adaptive"])
cs_overhead = st.sidebar.slider("Context-switch overhead (ticks)", 0, 3, 0)
procs = workloads.generate(scenario, n=n_processes, seed=int(seed))

# ---------------------------------------------------------------- Gantt
st.header("Schedule")
if algo == "rule_adaptive":
    result = run_rule_based(procs, cs_overhead=cs_overhead)
elif algo == "mlfq":
    q = st.sidebar.selectbox("MLFQ Q0 quantum", [1, 2, 4, 8])
    result = Simulator(procs, algorithm="mlfq", cs_overhead=cs_overhead,
                       mlfq=MLFQConfig((q, q * 2, q * 4, float("inf")), 50)).run()
else:
    result = Simulator(procs, algorithm=algo, rr_quantum=4, cs_overhead=cs_overhead).run()

rows = [{"Process": f"P{s.pid}" if s.pid and s.pid > 0 else ("CS overhead" if s.pid == -1 else "Idle"),
         "Start": s.start, "Finish": s.end} for s in result.timeline]
fig = px.timeline(pd.DataFrame(rows), x_start="Start", x_end="Finish", y="Process",
                  color="Process", title=f"Gantt chart - {algo} on {scenario} (seed {seed})")
fig.update_yaxes(autorange="reversed")
st.plotly_chart(fig, use_container_width=True)

step = st.slider("Playback: show schedule up to time", 0.0, float(result.makespan),
                 float(result.makespan))
visible = [r for r in rows if r["Start"] < step]
if visible:
    fig2 = px.timeline(pd.DataFrame(visible), x_start="Start", x_end="Finish",
                       y="Process", color="Process", title=f"Playback t <= {step:.0f}")
    fig2.update_yaxes(autorange="reversed")
    st.plotly_chart(fig2, use_container_width=True)

m = result.metrics()
cols = st.columns(4)
for col, key in zip(cols, ["avg_response", "avg_waiting", "throughput", "fairness_jain"]):
    col.metric(key, f"{m.get(key, 0):.3f}")
cols2 = st.columns(4)
for col, key in zip(cols2, ["p95_response", "avg_turnaround", "context_switches", "makespan"]):
    col.metric(key, f"{m.get(key, 0):.2f}" if isinstance(m.get(key), float) else m.get(key, 0))

if result.config_history:
    st.subheader("Parameter decisions over time")
    st.dataframe(pd.DataFrame(result.config_history, columns=["time", "MLFQ config"]))

# ------------------------------------------------------------- PPO decisions
st.header("PPO parameter decisions over time")
if os.path.exists(MODEL_PATH + ".zip"):
    from stable_baselines3 import PPO
    from learnsched.train import ppo_schedule
    if st.button("Run trained PPO on this workload"):
        model = PPO.load(MODEL_PATH)
        pres = ppo_schedule(procs, model=model, cs_overhead=cs_overhead)
        st.dataframe(pd.DataFrame(pres.config_history, columns=["time", "MLFQ config"]))
        st.write(pres.metrics())
else:
    st.info("No trained model yet. Run: python scripts/train_ppo.py --smoke")

# ------------------------------------------------------------- experiments
st.header("Experiment results")
exp = load_csv("experiments.csv")
train_log = load_csv("training_rewards.csv")
if train_log is not None and len(train_log):
    st.subheader("PPO training reward curve")
    t = train_log.copy()
    t["reward_smooth"] = t["episodic_reward"].rolling(10, min_periods=1).mean()
    st.plotly_chart(px.line(t, x="timesteps", y=["episodic_reward", "reward_smooth"],
                            title="Episodic reward during PPO training (measured)"),
                    use_container_width=True)

if exp is not None:
    for title, col in [("Latency comparison (avg response)", "avg_response"),
                       ("Throughput comparison", "throughput"),
                       ("Fairness comparison (Jain index)", "fairness_jain"),
                       ("Context-switch comparison", "context_switches")]:
        st.subheader(title)
        agg = exp.groupby("scheduler")[col].mean().reset_index()
        st.plotly_chart(px.bar(agg, x="scheduler", y=col, title=title), use_container_width=True)
    st.subheader("Trained vs untrained PPO")
    ppo = exp[exp.scheduler.isin(["ppo_trained", "ppo_untrained"])]
    if len(ppo):
        st.plotly_chart(px.bar(ppo.groupby(["scenario", "scheduler"])["avg_waiting"]
                               .mean().reset_index(),
                               x="scenario", y="avg_waiting", color="scheduler", barmode="group"),
                        use_container_width=True)
    st.subheader("Workload adaptation (changing / unseen workloads)")
    adapt = exp[exp.scenario.isin(["changing", "unseen_mixed"])]
    if len(adapt):
        st.plotly_chart(px.bar(adapt.groupby(["scenario", "scheduler"])["avg_response"]
                               .mean().reset_index(),
                               x="scenario", y="avg_response", color="scheduler", barmode="group"),
                        use_container_width=True)
    st.subheader("Experiment results summary")
    st.dataframe(exp.groupby(["scenario", "scheduler"])[METRIC_COLS].mean().round(3))
else:
    st.info("No results/experiments.csv yet. Run: python scripts/run_experiments.py --quick")

abl = load_csv("ablation.csv")
if abl is not None:
    st.subheader("Ablation: PPO decision interval")
    st.plotly_chart(px.bar(abl.groupby("decision_interval")["avg_waiting"].mean().reset_index(),
                           x="decision_interval", y="avg_waiting"), use_container_width=True)
