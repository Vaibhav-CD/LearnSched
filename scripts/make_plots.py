"""Generate Matplotlib figures from measured results CSVs."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("results/plots", exist_ok=True)
exp = pd.read_csv("results/experiments.csv")
for col in ["avg_waiting", "avg_response", "fairness_jain", "composite_score", "context_switches"]:
    agg = exp.groupby("scheduler")[col].mean()
    ax = agg.plot(kind="bar", figsize=(9, 4), title=f"Mean {col} by scheduler (all scenarios/seeds)")
    ax.set_ylabel(col)
    plt.tight_layout()
    plt.savefig(f"results/plots/{col}.png", dpi=130)
    plt.close()

tr = pd.read_csv("results/training_rewards.csv")
plt.figure(figsize=(9, 4))
plt.plot(tr["timesteps"], tr["episodic_reward"], alpha=0.35, label="episodic reward")
plt.plot(tr["timesteps"], tr["episodic_reward"].rolling(50, min_periods=1).mean(),
         label="rolling mean (50)")
plt.xlabel("timesteps"); plt.ylabel("reward"); plt.legend()
plt.title("PPO training reward (measured)")
plt.tight_layout(); plt.savefig("results/plots/training_reward.png", dpi=130); plt.close()
print("plots written to results/plots/")
