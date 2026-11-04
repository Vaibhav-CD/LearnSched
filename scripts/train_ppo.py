"""Train PPO. Smoke: python scripts/train_ppo.py --smoke ; Full: --timesteps 100000"""
import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from learnsched.train import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="fast 4000-step run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="models/ppo_mlfq")
    args = ap.parse_args()
    ts = 4000 if args.smoke else (args.timesteps or 50000)
    info = train(total_timesteps=ts, seed=args.seed, model_path=args.model)
    print(f"Trained {info['timesteps']} steps in {info['wall_time_s']:.1f}s -> {info['model_path']}")
    print(f"Episodes logged: {info['episodes_logged']} -> {info['log_csv']}")
