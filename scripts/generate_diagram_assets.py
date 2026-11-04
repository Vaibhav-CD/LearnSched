"""
Generate crisp architecture and MDP flow diagrams for slides 4 and 6.
"""
import os
import tempfile
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

import matplotlib.pyplot as plt
import matplotlib.patches as patches

NAVY = '#0b132b'
DARK_CARD = '#1c2541'
BOX_BG = '#202c4f'
BORDER = '#3a506b'
ELECTRIC_BLUE = '#00b4d8'
CYAN = '#48cae4'
EMERALD = '#06d6a0'
AMBER = '#ffd166'
CORAL = '#ef476f'
TEXT_WHITE = '#f8f9fa'
TEXT_MUTED = '#a0aec0'

def make_mdp_loop():
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=200, facecolor=DARK_CARD)
    ax.set_facecolor(DARK_CARD)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Agent Box
    agent_box = patches.FancyBboxPatch((1.0, 2.7), 3.5, 1.8, boxstyle="round,pad=0.2",
                                       fc=BOX_BG, ec=ELECTRIC_BLUE, lw=2.5)
    ax.add_patch(agent_box)
    ax.text(2.75, 4.0, "PPO Agent (Policy π_θ)", ha='center', va='center',
            color=TEXT_WHITE, fontsize=12, fontweight='bold')
    ax.text(2.75, 3.4, "Actor-Critic Network [64, 64]\nMeta-Tuning Policy", ha='center', va='center',
            color=CYAN, fontsize=9.5)

    # Environment Box
    env_box = patches.FancyBboxPatch((5.5, 2.7), 3.5, 1.8, boxstyle="round,pad=0.2",
                                     fc=BOX_BG, ec=EMERALD, lw=2.5)
    ax.add_patch(env_box)
    ax.text(7.25, 4.0, "CPU Simulator Environment", ha='center', va='center',
            color=TEXT_WHITE, fontsize=12, fontweight='bold')
    ax.text(7.25, 3.4, "Multi-Level Feedback Queue (MLFQ)\nDispatches processes every tick", ha='center', va='center',
            color=EMERALD, fontsize=9.5)

    # Action Arrow: Top (Agent -> Env)
    ax.annotate("", xy=(5.4, 3.8), xytext=(4.6, 3.8),
                arrowprops=dict(arrowstyle="->", color=ELECTRIC_BLUE, lw=2.5))
    ax.text(5.0, 4.25, "Action A_t\n(MLFQ Configuration)", ha='center', va='center',
            color=ELECTRIC_BLUE, fontsize=9, fontweight='bold')

    # State & Reward Arrow: Bottom (Env -> Agent)
    ax.annotate("", xy=(2.75, 2.6), xytext=(2.75, 1.2),
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=2.5))
    ax.plot([7.25, 7.25], [2.6, 1.2], color=AMBER, lw=2.5)
    ax.plot([7.25, 2.75], [1.2, 1.2], color=AMBER, lw=2.5)

    ax.text(5.0, 1.6, "State S_t (10D Normalized Vector)\nReward R_t (Ground-Truth Latency Signal)",
            ha='center', va='center', color=AMBER, fontsize=9.5, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.4", fc=NAVY, ec=AMBER, lw=1.5))

    ax.text(5.0, 0.4, "Objective: max_π E_π [ Σ γ^t r_t ]  (γ = 0.99, Meta-Interval = 20 ticks)",
            ha='center', va='center', color=TEXT_WHITE, fontsize=10, style='italic')

    plt.tight_layout()
    fig.savefig('presentation_assets/mdp_loop.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved mdp_loop.png")

def make_system_architecture():
    fig, ax = plt.subplots(figsize=(10.5, 4.2), dpi=200, facecolor=DARK_CARD)
    ax.set_facecolor(DARK_CARD)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    boxes = [
        ("Workload Traces\n6 Scenarios / 3 Seeds", 1.2, 2.5, AMBER),
        ("CPU Simulator\nTick-based Engine", 3.6, 2.5, EMERALD),
        ("State Observation\n10D Metric Vector", 6.0, 2.5, CYAN),
        ("PPO Policy Network\nMLP [64, 64] Actor", 8.4, 2.5, ELECTRIC_BLUE),
        ("MLFQ Scheduler\nQuanta & Boost Update", 10.8, 2.5, CORAL),
    ]

    for text, cx, cy, col in boxes:
        box = patches.FancyBboxPatch((cx - 1.0, cy - 0.9), 2.0, 1.8, boxstyle="round,pad=0.2",
                                     fc=BOX_BG, ec=col, lw=2)
        ax.add_patch(box)
        ax.text(cx, cy, text, ha='center', va='center', color=TEXT_WHITE, fontsize=8.5, fontweight='bold')

    for i in range(len(boxes) - 1):
        x1 = boxes[i][1] + 1.1
        x2 = boxes[i+1][1] - 1.1
        ax.annotate("", xy=(x2, 2.5), xytext=(x1, 2.5),
                    arrowprops=dict(arrowstyle="->", color=TEXT_WHITE, lw=2))

    # Feedback loop: from MLFQ back to Simulator every 20 ticks
    ax.plot([10.8, 10.8], [1.5, 0.7], color=CORAL, lw=2)
    ax.plot([10.8, 3.6], [0.7, 0.7], color=CORAL, lw=2)
    ax.annotate("", xy=(3.6, 1.5), xytext=(3.6, 0.7),
                arrowprops=dict(arrowstyle="->", color=CORAL, lw=2))
    ax.text(7.2, 0.95, "Feedback Cycle: Configuration applied every 20 ticks | Dispatcher runs every tick",
            ha='center', va='center', color=CORAL, fontsize=8.5, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc=NAVY, ec=CORAL, lw=1))

    plt.tight_layout()
    fig.savefig('presentation_assets/system_architecture.png', bbox_inches='tight')
    plt.close(fig)
    print("Saved system_architecture.png")

if __name__ == '__main__':
    make_mdp_loop()
    make_system_architecture()
