"""
Generate high-resolution visual assets for the IIT Guwahati LearnSched PPT presentation.
"""
import os
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs('presentation_assets', exist_ok=True)

# Style configuration: Deep navy & electric blue theme matching presentation
NAVY = '#0b132b'
DARK_CARD = '#1c2541'
ELECTRIC_BLUE = '#00b4d8'
CYAN = '#48cae4'
EMERALD = '#06d6a0'
AMBER = '#ffd166'
CORAL = '#ef476f'
TEXT_WHITE = '#f8f9fa'
TEXT_MUTED = '#a0aec0'
GRID_COLOR = '#2d3748'

plt.rcParams.update({
    'figure.facecolor': DARK_CARD,
    'axes.facecolor': DARK_CARD,
    'axes.edgecolor': GRID_COLOR,
    'axes.labelcolor': TEXT_WHITE,
    'xtick.color': TEXT_WHITE,
    'ytick.color': TEXT_WHITE,
    'text.color': TEXT_WHITE,
    'font.size': 11,
    'grid.color': GRID_COLOR,
    'grid.linestyle': '--',
    'grid.alpha': 0.6,
})

def make_static_vs_adaptive():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)
    
    # Static Scheduler
    time = np.arange(0, 100, 1)
    workload = np.sin(time / 10) * 10 + 15  # varying backlog
    static_quantum = np.full_like(time, 8.0)
    
    ax1.plot(time, workload, color=AMBER, label='Incoming Workload Burstiness', linewidth=2)
    ax1.plot(time, static_quantum, color=CORAL, linestyle='--', linewidth=2.5, label='Fixed MLFQ Quantum (Static = 8 ticks)')
    ax1.set_title('Static Scheduling: Rigid Parameters', color=TEXT_WHITE, fontweight='bold', pad=12)
    ax1.set_xlabel('Simulation Ticks (Time)', color=TEXT_WHITE)
    ax1.set_ylabel('Quantum / Queue Scale', color=TEXT_WHITE)
    ax1.legend(loc='upper right', framealpha=0.8, facecolor=NAVY, edgecolor=GRID_COLOR, fontsize=9)
    ax1.grid(True)
    ax1.text(50, 4, 'Mismatch: Suboptimal for\nbursty I/O & high contention', ha='center', color=CORAL, fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor=NAVY, edgecolor=CORAL, alpha=0.9))

    # Adaptive PPO Scheduler
    adaptive_quantum = np.where(workload > 16, 2.0, np.where(workload < 10, 8.0, 4.0))
    ax2.plot(time, workload, color=AMBER, label='Incoming Workload Burstiness', linewidth=2)
    ax2.step(time, adaptive_quantum, color=ELECTRIC_BLUE, where='post', linewidth=2.5, label='PPO Dynamic Quantum Choice')
    ax2.set_title('Adaptive Scheduling: Workload-Driven', color=TEXT_WHITE, fontweight='bold', pad=12)
    ax2.set_xlabel('Simulation Ticks (Time)', color=TEXT_WHITE)
    ax2.set_ylabel('Quantum / Queue Scale', color=TEXT_WHITE)
    ax2.legend(loc='upper right', framealpha=0.8, facecolor=NAVY, edgecolor=GRID_COLOR, fontsize=9)
    ax2.grid(True)
    ax2.text(50, 4, 'PPO contracts quanta on burst\n& expands on CPU-bound phase', ha='center', color=CYAN, fontsize=9,
             bbox=dict(boxstyle='round,pad=0.5', facecolor=NAVY, edgecolor=ELECTRIC_BLUE, alpha=0.9))

    plt.tight_layout()
    fig.savefig('presentation_assets/static_vs_adaptive.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved static_vs_adaptive.png")

def make_ppo_clipping():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)
    eps = 0.2
    r = np.linspace(0.4, 1.6, 200)
    
    # Positive Advantage
    A_pos = 1.0
    obj_unclipped_pos = r * A_pos
    obj_clipped_pos = np.clip(r, 1 - eps, 1 + eps) * A_pos
    ppo_obj_pos = np.minimum(obj_unclipped_pos, obj_clipped_pos)

    ax1.plot(r, obj_unclipped_pos, '--', color=TEXT_MUTED, label=r'$r_t(\theta)\hat{A}_t$', linewidth=1.8)
    ax1.plot(r, ppo_obj_pos, color=ELECTRIC_BLUE, label=r'$L^{CLIP}(\theta)$ Objective', linewidth=2.5)
    ax1.axvline(1 - eps, color=CORAL, linestyle=':', alpha=0.8)
    ax1.axvline(1 + eps, color=CORAL, linestyle=':', alpha=0.8, label=r'$1 \pm \epsilon$ Clipping Bounds')
    ax1.fill_between(r, ppo_obj_pos, 0, color=ELECTRIC_BLUE, alpha=0.15)
    ax1.set_title(r'Positive Advantage ($\hat{A}_t > 0$): Action Better than Avg', color=TEXT_WHITE, fontweight='bold', pad=12)
    ax1.set_xlabel(r'Probability Ratio $r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)}$', color=TEXT_WHITE)
    ax1.set_ylabel('Surrogate Objective Value', color=TEXT_WHITE)
    ax1.legend(loc='lower right', framealpha=0.8, facecolor=NAVY, edgecolor=GRID_COLOR, fontsize=9)
    ax1.grid(True)
    ax1.annotate('Clipping prevents\nexcessive policy updates', xy=(1.35, 1.2), xytext=(1.1, 0.7),
                 arrowprops=dict(arrowstyle='->', color=CYAN), color=CYAN, fontsize=9)

    # Negative Advantage
    A_neg = -1.0
    obj_unclipped_neg = r * A_neg
    obj_clipped_neg = np.clip(r, 1 - eps, 1 + eps) * A_neg
    ppo_obj_neg = np.minimum(obj_unclipped_neg, obj_clipped_neg)

    ax2.plot(r, obj_unclipped_neg, '--', color=TEXT_MUTED, label=r'$r_t(\theta)\hat{A}_t$', linewidth=1.8)
    ax2.plot(r, ppo_obj_neg, color=AMBER, label=r'$L^{CLIP}(\theta)$ Objective', linewidth=2.5)
    ax2.axvline(1 - eps, color=CORAL, linestyle=':', alpha=0.8)
    ax2.axvline(1 + eps, color=CORAL, linestyle=':', alpha=0.8, label=r'$1 \pm \epsilon$ Clipping Bounds')
    ax2.fill_between(r, ppo_obj_neg, -2, color=AMBER, alpha=0.15)
    ax2.set_title(r'Negative Advantage ($\hat{A}_t < 0$): Action Worse than Avg', color=TEXT_WHITE, fontweight='bold', pad=12)
    ax2.set_xlabel(r'Probability Ratio $r_t(\theta)$', color=TEXT_WHITE)
    ax2.set_ylabel('Surrogate Objective Value', color=TEXT_WHITE)
    ax2.legend(loc='lower left', framealpha=0.8, facecolor=NAVY, edgecolor=GRID_COLOR, fontsize=9)
    ax2.grid(True)

    plt.tight_layout()
    fig.savefig('presentation_assets/ppo_clipping.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved ppo_clipping.png")

def make_training_curve():
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=200)
    df = pd.read_csv('results/training_rewards.csv')
    x = df['timesteps']
    y = df['episodic_reward']
    
    # Calculate rolling window
    rolling_mean = y.rolling(window=100, min_periods=10).mean()
    rolling_std = y.rolling(window=100, min_periods=10).std().fillna(0)
    
    ax.scatter(x, y, color=CYAN, alpha=0.15, s=8, label='Raw Episode Returns')
    ax.plot(x, rolling_mean, color=ELECTRIC_BLUE, linewidth=2.5, label='100-Episode Moving Average')
    ax.fill_between(x, rolling_mean - rolling_std, rolling_mean + rolling_std, color=ELECTRIC_BLUE, alpha=0.2, label=r'$\pm 1\sigma$ Return Variance')
    
    ax.set_title('PPO Training Progress over 120,000 Timesteps (Reward v3 Formulation)', color=TEXT_WHITE, fontweight='bold', pad=12)
    ax.set_xlabel('Environment Timesteps', color=TEXT_WHITE)
    ax.set_ylabel('Episodic Return', color=TEXT_WHITE)
    ax.legend(loc='lower right', framealpha=0.8, facecolor=NAVY, edgecolor=GRID_COLOR, fontsize=9)
    ax.grid(True)
    
    # Annotations
    ax.annotate('Initial exploration\n(high variance)', xy=(10000, rolling_mean.iloc[100]), xytext=(20000, -80),
                arrowprops=dict(arrowstyle='->', color=AMBER), color=AMBER, fontsize=9)
    ax.annotate('Stabilized policy\nconvergence', xy=(100000, rolling_mean.iloc[-50]), xytext=(75000, -25),
                arrowprops=dict(arrowstyle='->', color=EMERALD), color=EMERALD, fontsize=9)

    plt.tight_layout()
    fig.savefig('presentation_assets/training_curve.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved training_curve.png")

def make_overall_comparison():
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 6.5), dpi=200)
    
    df = pd.read_csv('results/experiments.csv')
    order = ['sjf', 'mlfq_tuned', 'ppo_trained', 'ppo_untrained', 'rule_adaptive', 'mlfq_fixed', 'rr', 'fcfs']
    labels = ['SJF (Oracle)', 'MLFQ Tuned', 'PPO Trained', 'PPO Untrained', 'Rule Adaptive', 'MLFQ Fixed', 'Round Robin', 'FCFS']
    colors = [EMERALD, CYAN, ELECTRIC_BLUE, TEXT_MUTED, CORAL, TEXT_MUTED, TEXT_MUTED, TEXT_MUTED]
    
    summary = df.groupby('scheduler').mean(numeric_only=True).reindex(order)
    
    # 1. Average Waiting Time
    ax1.barh(labels[::-1], summary['avg_waiting'].values[::-1], color=colors[::-1], height=0.6)
    ax1.set_title('Average Waiting Time (ticks) ↓ (Lower is Better)', color=TEXT_WHITE, fontweight='bold', fontsize=10)
    ax1.grid(True, axis='x')
    for i, v in enumerate(summary['avg_waiting'].values[::-1]):
        ax1.text(v + 5, i, f"{v:.1f}", va='center', color=TEXT_WHITE, fontsize=8)

    # 2. Average Response Time
    ax2.barh(labels[::-1], summary['avg_response'].values[::-1], color=colors[::-1], height=0.6)
    ax2.set_title('Average Response Time (ticks) ↓ (Lower is Better)', color=TEXT_WHITE, fontweight='bold', fontsize=10)
    ax2.grid(True, axis='x')
    for i, v in enumerate(summary['avg_response'].values[::-1]):
        ax2.text(v + 3, i, f"{v:.1f}", va='center', color=TEXT_WHITE, fontsize=8)

    # 3. Context Switches
    ax3.barh(labels[::-1], summary['context_switches'].values[::-1], color=colors[::-1], height=0.6)
    ax3.set_title('Context Switches ↓ (Lower Overhead)', color=TEXT_WHITE, fontweight='bold', fontsize=10)
    ax3.grid(True, axis='x')
    for i, v in enumerate(summary['context_switches'].values[::-1]):
        ax3.text(v + 8, i, f"{int(v)}", va='center', color=TEXT_WHITE, fontsize=8)

    # 4. Jain Fairness Index
    ax4.barh(labels[::-1], summary['fairness_jain'].values[::-1], color=colors[::-1], height=0.6)
    ax4.set_title('Jain Fairness Index ↑ (Higher is Fairer)', color=TEXT_WHITE, fontweight='bold', fontsize=10)
    ax4.grid(True, axis='x')
    ax4.set_xlim(0, 1.0)
    for i, v in enumerate(summary['fairness_jain'].values[::-1]):
        ax4.text(v + 0.02, i, f"{v:.2f}", va='center', color=TEXT_WHITE, fontsize=8)

    plt.tight_layout()
    fig.savefig('presentation_assets/overall_comparison.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved overall_comparison.png")

def make_workload_breakdown():
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=200)
    
    scenarios = ['io_intensive', 'bursty', 'changing', 'mixed', 'unseen_mixed', 'cpu_bound']
    display_names = ['I/O Intensive', 'Bursty Arrivals', 'Changing Workload', 'Mixed Traffic', 'Unseen Mixed (Held-out)', 'CPU-bound']
    
    # Exact measured numbers from experiments.csv:
    # io: ppo 176.3, tuned 182.0
    # bursty: ppo 236.3, tuned 240.7
    # changing: ppo 238.2, tuned 231.4
    # mixed: ppo 217.6, tuned 213.9
    # unseen: ppo 333.1, tuned 333.1
    # cpu: ppo 517.7, tuned 516.4
    ppo_vals = [176.3, 236.3, 238.2, 217.6, 333.1, 517.7]
    tuned_vals = [182.0, 240.7, 231.4, 213.9, 333.1, 516.4]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, ppo_vals, width, label='PPO Trained (Final)', color=ELECTRIC_BLUE)
    rects2 = ax.bar(x + width/2, tuned_vals, width, label='MLFQ Tuned (Validation Oracle)', color=AMBER)
    
    ax.set_title('Average Waiting Time by Workload Family (PPO vs Tuned MLFQ)', color=TEXT_WHITE, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Waiting Time (ticks) ↓', color=TEXT_WHITE)
    ax.legend(framealpha=0.8, facecolor=NAVY, edgecolor=GRID_COLOR, fontsize=9)
    ax.grid(True, axis='y')
    
    for r in rects1:
        h = r.get_height()
        ax.annotate(f'{h:.1f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8, color=CYAN)
    for r in rects2:
        h = r.get_height()
        ax.annotate(f'{h:.1f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8, color=AMBER)

    plt.tight_layout()
    fig.savefig('presentation_assets/workload_breakdown.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved workload_breakdown.png")

def make_tradeoff_pareto():
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=200)
    
    scheds = {
        'PPO Trained': (286.5, 0.51, ELECTRIC_BLUE, 'o', 140),
        'MLFQ Tuned': (286.3, 0.59, CYAN, 's', 140),
        'SJF (Oracle)': (166.2, 0.54, EMERALD, '^', 150),
        'PPO Untrained': (306.6, 0.73, TEXT_MUTED, 'o', 100),
        'Rule Adaptive': (343.5, 0.69, CORAL, 'd', 120),
        'MLFQ Fixed': (356.7, 0.68, TEXT_MUTED, 's', 100),
        'Round Robin': (360.5, 0.65, TEXT_MUTED, 'v', 100),
        'FCFS': (297.6, 0.37, TEXT_MUTED, 'x', 120),
    }
    
    for name, (wait, fair, color, marker, size) in scheds.items():
        ax.scatter(wait, fair, color=color, marker=marker, s=size, label=name, edgecolors=TEXT_WHITE if color in (ELECTRIC_BLUE, CYAN, EMERALD) else None, linewidth=1.5)
        offset_x = 7
        offset_y = 0.015
        if name == 'MLFQ Tuned':
            offset_y = 0.025
        elif name == 'PPO Trained':
            offset_y = -0.035
        elif name == 'SJF (Oracle)':
            offset_x = -15
            offset_y = 0.02
        ax.annotate(name, (wait, fair), xytext=(wait + offset_x, fair + offset_y),
                    color=color, fontweight='bold' if color in (ELECTRIC_BLUE, CYAN, EMERALD) else 'normal', fontsize=9)
        
    ax.set_title('Pareto Trade-off: Average Waiting Time vs Jain Fairness', color=TEXT_WHITE, fontweight='bold', pad=12)
    ax.set_xlabel('Average Waiting Time (ticks) ↓ (Lower is Better)', color=TEXT_WHITE)
    ax.set_ylabel('Jain Fairness Index ↑ (Higher is Better)', color=TEXT_WHITE)
    ax.grid(True)
    ax.set_ylim(0.3, 0.85)
    ax.set_xlim(140, 400)
    
    # Pareto frontier sketch
    ax.annotate('Pareto Frontier Region\n(Latency vs Fairness Trade-off)', xy=(286.5, 0.51), xytext=(220, 0.40),
                arrowprops=dict(arrowstyle='->', color=AMBER, lw=1.5), color=AMBER, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor=NAVY, edgecolor=AMBER, alpha=0.9))

    plt.tight_layout()
    fig.savefig('presentation_assets/tradeoff_pareto.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved tradeoff_pareto.png")

if __name__ == '__main__':
    make_static_vs_adaptive()
    make_ppo_clipping()
    make_training_curve()
    make_overall_comparison()
    make_workload_breakdown()
    make_tradeoff_pareto()
    print("All presentation assets successfully generated!")
