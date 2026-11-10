"""LearnSched Research Dashboard.

Run from the project root:  streamlit run dashboard/app.py
All numbers come from real Simulator runs and the CSVs in results/.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from learnsched import workloads
from learnsched.baselines import FIXED_MLFQ, TUNED_MLFQ, run_rule_based
from learnsched.env import ACTION_CONFIGS, MLFQTuningEnv
from learnsched.experiments import METRIC_COLS
from learnsched.simulator import MLFQConfig, Simulator
from learnsched.train import load_model, ppo_schedule

st.set_page_config(
    page_title="LearnSched Research Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("⚡ LearnSched: RL-Adaptive CPU Scheduling Dashboard")
st.caption(
    "Workload-adaptive tuning of MLFQ parameters using Proximal Policy Optimization (PPO). "
    "All simulations, traces, and benchmark numbers are empirical and measured."
)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "ppo_mlfq")


@st.cache_data
def load_csv(name: str):
    path = os.path.join(RESULTS, name)
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_resource
def get_trained_ppo_model():
    if os.path.exists(MODEL_PATH + ".zip"):
        return load_model(MODEL_PATH)
    return None


ALGO_CHOICES = {
    "PPO Trained (Fairness-Aware v4)": "ppo_trained",
    "MLFQ Tuned (Validation Oracle)": "mlfq_tuned",
    "MLFQ Fixed (Default q=4/8/16)": "mlfq_fixed",
    "Rule-Based Adaptive Heuristic": "rule_adaptive",
    "Round Robin (RR, q=4)": "rr",
    "Shortest Job First (SJF Oracle)": "sjf",
    "First-Come First-Served (FCFS)": "fcfs",
    "PPO Untrained (Random Weights)": "ppo_untrained",
}


def run_scheduler_algorithm(algo_key: str, processes, cs_overhead: float = 0.0):
    """Run any selected scheduler on a set of processes."""
    if algo_key == "ppo_trained":
        model = get_trained_ppo_model()
        if model is not None:
            return ppo_schedule(processes, model=model, cs_overhead=cs_overhead)
        st.warning("Trained model checkpoint not found; running default MLFQ.")
        return Simulator(processes, algorithm="mlfq", cs_overhead=cs_overhead, mlfq=FIXED_MLFQ).run()

    elif algo_key == "ppo_untrained":
        return ppo_schedule(processes, model=None, cs_overhead=cs_overhead)

    elif algo_key == "mlfq_tuned":
        return Simulator(processes, algorithm="mlfq", cs_overhead=cs_overhead, mlfq=TUNED_MLFQ).run()

    elif algo_key == "mlfq_fixed":
        return Simulator(processes, algorithm="mlfq", cs_overhead=cs_overhead, mlfq=FIXED_MLFQ).run()

    elif algo_key == "rule_adaptive":
        return run_rule_based(processes, cs_overhead=cs_overhead)

    elif algo_key == "rr":
        return Simulator(processes, algorithm="rr", rr_quantum=4, cs_overhead=cs_overhead).run()

    elif algo_key == "sjf":
        return Simulator(processes, algorithm="sjf", cs_overhead=cs_overhead).run()

    elif algo_key == "fcfs":
        return Simulator(processes, algorithm="fcfs", cs_overhead=cs_overhead).run()

    else:
        return Simulator(processes, algorithm="mlfq", cs_overhead=cs_overhead, mlfq=FIXED_MLFQ).run()


def gantt_fig(segments, title: str, max_x: float | None = None):
    """Numeric-tick Gantt chart with consistent coloring."""
    labels = {}
    for s in segments:
        name = f"P{s.pid}" if s.pid and s.pid > 0 else ("CS overhead" if s.pid == -1 else "Idle")
        labels.setdefault(name, []).append(s)

    fig = go.Figure()
    for name, segs in labels.items():
        is_special = name in ("Idle", "CS overhead")
        fig.add_trace(
            go.Bar(
                name=name,
                orientation="h",
                y=[name] * len(segs),
                x=[s.end - s.start for s in segs],
                base=[s.start for s in segs],
                marker_color="#64748b" if name == "CS overhead" else ("#334155" if name == "Idle" else None),
                hovertemplate=f"<b>{name}</b>: [%{{base}}, %{{x}} ticks]<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        barmode="overlay",
        xaxis_title="Time (ticks)",
        yaxis_title="Process",
        height=max(260, 24 * len(labels)),
        margin=dict(l=20, r=20, t=40, b=20),
    )
    if max_x:
        fig.update_xaxes(range=[0, max_x])
    fig.update_yaxes(autorange="reversed")
    return fig


# ---------------------------------------------------------------- Sidebar
st.sidebar.header("🕹️ Simulation Controls")
view_mode = st.sidebar.radio("Inspection Mode", ["Single Algorithm", "Side-by-Side Comparison"])

scenario = st.sidebar.selectbox("Workload scenario", workloads.SCENARIOS, index=0)
n_processes = st.sidebar.slider("Number of processes", 10, 60, 30)
seed = st.sidebar.number_input("Random trace seed", value=0, step=1)
cs_overhead = st.sidebar.slider("Context-switch overhead (ticks)", 0, 3, 0)

algo_name = st.sidebar.selectbox("Algorithm", list(ALGO_CHOICES.keys()), index=0)
algo_key = ALGO_CHOICES[algo_name]

if view_mode == "Side-by-Side Comparison":
    comp_name = st.sidebar.selectbox(
        "Compare Against",
        [k for k in ALGO_CHOICES.keys() if k != algo_name],
        index=0,
    )
    comp_key = ALGO_CHOICES[comp_name]
else:
    comp_name, comp_key = None, None

# Generate workload trace
procs = workloads.generate(scenario, n=n_processes, seed=int(seed))

# ---------------------------------------------------------------- Simulation Execution
res_primary = run_scheduler_algorithm(algo_key, procs, cs_overhead=cs_overhead)
m_primary = res_primary.metrics()

if comp_key:
    # Run comparison on identical trace
    procs_comp = workloads.generate(scenario, n=n_processes, seed=int(seed))
    res_comp = run_scheduler_algorithm(comp_key, procs_comp, cs_overhead=cs_overhead)
    m_comp = res_comp.metrics()
else:
    res_comp, m_comp = None, None

# ---------------------------------------------------------------- Header Metrics
st.header("📊 Performance Metrics")

if view_mode == "Side-by-Side Comparison" and m_comp:
    col1, col2, col3, col4, col5 = st.columns(5)
    
    comp_diff = m_primary.get("composite_score", 0) - m_comp.get("composite_score", 0)
    wait_diff = m_primary.get("avg_waiting", 0) - m_comp.get("avg_waiting", 0)
    resp_diff = m_primary.get("avg_response", 0) - m_comp.get("avg_response", 0)
    fair_diff = m_primary.get("fairness_jain", 0) - m_comp.get("fairness_jain", 0)
    cs_diff = m_primary.get("context_switches", 0) - m_comp.get("context_switches", 0)

    col1.metric("Composite Score ↓", f"{m_primary.get('composite_score', 0):.2f}",
                delta=f"{comp_diff:+.2f} vs baseline", delta_color="inverse")
    col2.metric("Avg Waiting (ticks) ↓", f"{m_primary.get('avg_waiting', 0):.2f}",
                delta=f"{wait_diff:+.2f}", delta_color="inverse")
    col3.metric("Avg Response (ticks) ↓", f"{m_primary.get('avg_response', 0):.2f}",
                delta=f"{resp_diff:+.2f}", delta_color="inverse")
    col4.metric("Jain Fairness ↑", f"{m_primary.get('fairness_jain', 0):.3f}",
                delta=f"{fair_diff:+.3f}", delta_color="normal")
    col5.metric("Context Switches ↓", f"{m_primary.get('context_switches', 0)}",
                delta=f"{cs_diff:+d}", delta_color="inverse")

    # Detailed Comparison Table
    st.subheader("Side-by-Side Comparison Table")
    cmp_df = pd.DataFrame([
        {
            "Metric": "Composite Score (Lower = Better)",
            algo_name: round(m_primary.get("composite_score", 0), 2),
            comp_name: round(m_comp.get("composite_score", 0), 2),
            "Delta (Primary - Baseline)": round(comp_diff, 2),
        },
        {
            "Metric": "Average Waiting Time (ticks)",
            algo_name: round(m_primary.get("avg_waiting", 0), 2),
            comp_name: round(m_comp.get("avg_waiting", 0), 2),
            "Delta (Primary - Baseline)": round(wait_diff, 2),
        },
        {
            "Metric": "Average Response Time (ticks)",
            algo_name: round(m_primary.get("avg_response", 0), 2),
            comp_name: round(m_comp.get("avg_response", 0), 2),
            "Delta (Primary - Baseline)": round(resp_diff, 2),
        },
        {
            "Metric": "p95 Response Time (ticks)",
            algo_name: round(m_primary.get("p95_response", 0), 2),
            comp_name: round(m_comp.get("p95_response", 0), 2),
            "Delta (Primary - Baseline)": round(m_primary.get("p95_response", 0) - m_comp.get("p95_response", 0), 2),
        },
        {
            "Metric": "Jain Fairness Index (Higher = Fairer)",
            algo_name: round(m_primary.get("fairness_jain", 0), 3),
            comp_name: round(m_comp.get("fairness_jain", 0), 3),
            "Delta (Primary - Baseline)": round(fair_diff, 3),
        },
        {
            "Metric": "Context Switches (Overhead)",
            algo_name: int(m_primary.get("context_switches", 0)),
            comp_name: int(m_comp.get("context_switches", 0)),
            "Delta (Primary - Baseline)": int(cs_diff),
        },
        {
            "Metric": "Makespan (Total completion ticks)",
            algo_name: round(m_primary.get("makespan", 0), 1),
            comp_name: round(m_comp.get("makespan", 0), 1),
            "Delta (Primary - Baseline)": round(m_primary.get("makespan", 0) - m_comp.get("makespan", 0), 1),
        },
    ])
    st.dataframe(cmp_df, use_container_width=True)

else:
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Composite Score ↓", f"{m_primary.get('composite_score', 0):.2f}",
                help="waiting + 0.5*response - 100*fairness")
    col2.metric("Avg Waiting ↓", f"{m_primary.get('avg_waiting', 0):.2f}")
    col3.metric("Avg Response ↓", f"{m_primary.get('avg_response', 0):.2f}")
    col4.metric("Jain Fairness ↑", f"{m_primary.get('fairness_jain', 0):.3f}")
    col5.metric("Context Switches ↓", f"{m_primary.get('context_switches', 0)}")

    sec1, sec2, sec3, sec4 = st.columns(4)
    sec1.metric("p95 Response", f"{m_primary.get('p95_response', 0):.2f}")
    sec2.metric("Avg Turnaround", f"{m_primary.get('avg_turnaround', 0):.2f}")
    sec3.metric("Throughput (proc/tick)", f"{m_primary.get('throughput', 0):.4f}")
    sec4.metric("Makespan", f"{m_primary.get('makespan', 0):.1f}")

# ---------------------------------------------------------------- Timeline & Playback
st.header("📈 Schedule Timeline")

max_makespan = max(float(res_primary.makespan), float(res_comp.makespan) if res_comp else 0.0)

if view_mode == "Side-by-Side Comparison" and res_comp:
    st.subheader(f"1. {algo_name}")
    st.plotly_chart(gantt_fig(res_primary.timeline, f"Gantt: {algo_name}", max_x=max_makespan), use_container_width=True)

    st.subheader(f"2. {comp_name}")
    st.plotly_chart(gantt_fig(res_comp.timeline, f"Gantt: {comp_name}", max_x=max_makespan), use_container_width=True)

else:
    st.plotly_chart(gantt_fig(res_primary.timeline, f"Gantt Chart — {algo_name} on {scenario} (seed {seed})"), use_container_width=True)

    step = st.slider("Playback: show schedule up to time", 0.0, float(res_primary.makespan), float(res_primary.makespan))
    visible = [s for s in res_primary.timeline if s.start < step]
    if visible:
        st.plotly_chart(gantt_fig(visible, f"Playback t <= {step:.0f}"), use_container_width=True)

# ---------------------------------------------------------------- Parameter Adaptation Tracking
if res_primary.config_history:
    st.header("🎛️ Online Parameter Adaptations Over Time")
    st.caption("Dynamic MLFQ adjustments selected by the controller at 20-tick intervals.")
    
    cfg_rows = []
    for t_step, cfg in res_primary.config_history:
        q0 = cfg.quanta[0] if hasattr(cfg, "quanta") else None
        boost = cfg.boost_interval if hasattr(cfg, "boost_interval") else None
        cfg_rows.append({"Time (ticks)": t_step, "Q0 Quantum": q0, "Boost Interval": boost, "Config Summary": str(cfg)})
    
    cfg_df = pd.DataFrame(cfg_rows)
    
    c1, c2 = st.columns([2, 1])
    with c1:
        fig_cfg = px.step(
            cfg_df,
            x="Time (ticks)",
            y="Q0 Quantum",
            title=f"Q0 Quantum Adaptation Over Time ({algo_name})",
            markers=True,
        )
        fig_cfg.update_layout(yaxis_title="Q0 Quantum (ticks)")
        st.plotly_chart(fig_cfg, use_container_width=True)
    with c2:
        st.dataframe(cfg_df[["Time (ticks)", "Q0 Quantum", "Boost Interval"]], use_container_width=True, height=300)

# ---------------------------------------------------------------- Workload Trace Inspector
with st.expander("🔍 Inspect Generated Workload Trace (Process Details)"):
    trace_data = []
    for p in procs:
        trace_data.append({
            "PID": p.pid,
            "Arrival": p.arrival,
            "Total CPU": p.total_cpu,
            "Total I/O": p.total_io,
            "CPU Bursts": len(p.cpu_bursts),
            "First Burst": p.cpu_bursts[0] if p.cpu_bursts else 0,
        })
    st.dataframe(pd.DataFrame(trace_data), use_container_width=True)

# ---------------------------------------------------------------- Empirical Benchmark Results
st.header("🧪 Empirical Benchmark Results (144 Runs)")
exp = load_csv("experiments.csv")
train_log = load_csv("training_rewards.csv")

if exp is not None:
    tab1, tab2, tab3 = st.tabs(["Overall Comparison", "Workload Breakdown", "Training Convergence"])

    with tab1:
        st.subheader("Leaderboard: Mean Performance Across All Scenarios & Seeds")
        if "composite_score" in exp.columns:
            leaderboard = exp.groupby("scheduler")[
                ["composite_score", "avg_waiting", "avg_response", "fairness_jain", "context_switches", "p95_response"]
            ].mean().sort_values("composite_score").reset_index()
            st.dataframe(leaderboard.round(2), use_container_width=True)

            fig_comp = px.bar(
                leaderboard,
                x="scheduler",
                y="composite_score",
                color="scheduler",
                title="Composite Score by Scheduler ↓ (Lower is Better: waiting + 0.5*response - 100*fairness)",
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        for title, col in [
            ("Average Waiting Time (ticks) ↓", "avg_waiting"),
            ("Average Response Time (ticks) ↓", "avg_response"),
            ("Jain Fairness Index ↑", "fairness_jain"),
            ("Context Switches (Overhead) ↓", "context_switches"),
        ]:
            if col in exp.columns:
                agg = exp.groupby("scheduler")[col].mean().reset_index()
                st.plotly_chart(px.bar(agg, x="scheduler", y=col, title=title), use_container_width=True)

    with tab2:
        st.subheader("Workload Family Breakdown")
        scen_agg = exp.groupby(["scenario", "scheduler"])[["avg_waiting", "avg_response", "fairness_jain", "composite_score"]].mean().reset_index()
        
        target_metric = st.selectbox("Select metric for breakdown", ["composite_score", "avg_waiting", "avg_response", "fairness_jain"])
        st.plotly_chart(
            px.bar(scen_agg, x="scenario", y=target_metric, color="scheduler", barmode="group",
                   title=f"{target_metric} Across Workload Families"),
            use_container_width=True,
        )

    with tab3:
        if train_log is not None and len(train_log):
            st.subheader("PPO Training Return (Reward v4 with Jain Fairness Penalty)")
            t = train_log.copy()
            t["reward_smooth"] = t["episodic_reward"].rolling(50, min_periods=5).mean()
            st.plotly_chart(
                px.line(t, x="timesteps", y=["episodic_reward", "reward_smooth"],
                        title="Episodic Return Over 120,000 Environment Timesteps (Measured)"),
                use_container_width=True,
            )
else:
    st.info("No results/experiments.csv found. Run: python scripts/run_experiments.py")
