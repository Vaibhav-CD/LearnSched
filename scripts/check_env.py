"""Validate the Gymnasium environment. Usage: python scripts/check_env.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gymnasium.utils.env_checker import check_env
from learnsched.env import MLFQTuningEnv

if __name__ == "__main__":
    check_env(MLFQTuningEnv(n_processes=15), skip_render_check=True)
    print("Gymnasium env check: PASS")
