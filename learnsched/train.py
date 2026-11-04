"""PPO training with Stable-Baselines3 (standard PPO, MLP policy).

No new RL algorithm is invented here: this is SB3's PPO with the clipped
surrogate objective, GAE advantage estimation and configurable
hyperparameters / seeds. See docs/PPO.md for the undergraduate maths.
"""
from __future__ import annotations

import csv
import os
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from .env import MLFQTuningEnv

DEFAULT_HPARAMS = dict(learning_rate=3e-4, n_steps=512, batch_size=64,
                       n_epochs=10, gamma=0.99, gae_lambda=0.95,
                       clip_range=0.2, ent_coef=0.01,
                       policy_kwargs=dict(net_arch=[64, 64]))


class RewardLogCallback(BaseCallback):
    """Writes (timesteps, episodic reward) rows to a CSV for dashboards."""

    def __init__(self, csv_path):
        super().__init__()
        self.csv_path = csv_path
        self.rows = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep is not None:
                self.rows.append((self.num_timesteps, float(ep["r"])))
        return True

    def save(self):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timesteps", "episodic_reward"])
            w.writerows(self.rows)


def train(total_timesteps=20000, seed=0, model_path="models/ppo_mlfq",
          log_csv="results/training_rewards.csv", n_processes=40,
          decision_interval=20, hparams: dict | None = None, verbose=0):
    hp = {**DEFAULT_HPARAMS, **(hparams or {})}
    env = Monitor(MLFQTuningEnv(n_processes=n_processes,
                                decision_interval=decision_interval, seed=seed))
    eval_env = Monitor(MLFQTuningEnv(n_processes=n_processes,
                                     decision_interval=decision_interval, seed=seed + 999))
    model = PPO("MlpPolicy", env, seed=seed, verbose=verbose, **hp)
    reward_cb = RewardLogCallback(log_csv)
    eval_cb = EvalCallback(eval_env, eval_freq=max(1000, total_timesteps // 10),
                           n_eval_episodes=3, verbose=0)
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=[reward_cb, eval_cb])
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    model.save(model_path)
    reward_cb.save()
    return {"model_path": model_path + ".zip", "log_csv": log_csv,
            "wall_time_s": time.time() - t0, "timesteps": total_timesteps,
            "seed": seed, "episodes_logged": len(reward_cb.rows)}


def load_model(model_path):
    return PPO.load(model_path)


def ppo_schedule(processes, model=None, deterministic=True, cs_overhead=0.0,
                 decision_interval=20):
    """Run one trace under PPO control (untrained model if model=None)."""
    import time as _time
    env = MLFQTuningEnv(fixed_trace=processes, decision_interval=decision_interval,
                        cs_overhead=cs_overhead)
    if model is None:
        model = PPO("MlpPolicy", env, seed=123)  # random-init policy = untrained
    obs, _ = env.reset(seed=0)
    done, result, overhead = False, None, 0.0
    while not done:
        t0 = _time.perf_counter()
        action, _ = model.predict(obs, deterministic=deterministic)
        overhead += _time.perf_counter() - t0
        obs, _, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        if done:
            result = info.get("result")
    if result is not None:
        result.decision_overhead = overhead
    return result
