# src/data/generation.py

import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import os
import numpy as np
from utils.config import RAW_DIR, OBS_DIR, SEQ_LEN, DT, make_run_dirs


# -------------------------------
# Lorenz-63 Simulator
# -------------------------------
def lorenz63_step(x, sigma=10.0, rho=28.0, beta=8/3, dt=0.01):
    dx = sigma * (x[1] - x[0])
    dy = x[0] * (rho - x[2]) - x[1]
    dz = x[0] * x[1] - beta * x[2]
    return x + dt * np.array([dx, dy, dz])

def simulate_lorenz63(init, steps, dt=0.01):
    traj = np.zeros((steps, 3))
    x = init.copy()
    for i in range(steps):
        x = lorenz63_step(x, dt=dt)
        traj[i] = x
    return traj

# -------------------------------
# Observation Operators
# -------------------------------
def obs_operator(x, mode="x"):
    if mode == "x":
        return np.array([x[0]])
    elif mode == "xy":
        return np.array([x[0], x[1]])
    elif mode == "x2":
        return np.array([x[0] ** 2])
    else:
        raise ValueError(f"Unknown obs mode: {mode}")

def make_observations(trajs, mode="x", sigma_noise=0.05):
    obs = []
    for traj in trajs:
        y = np.array([obs_operator(x, mode) for x in traj])
        y_noisy = y + np.random.normal(0, sigma_noise, y.shape)
        obs.append(y_noisy)
    return np.array(obs)

# -------------------------------
# Dataset Generation Function
# -------------------------------
# data/generation.py
import os, sys, argparse
import numpy as np

# Make 'src' importable when running this file directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils.config import RAW_DIR, OBS_DIR, SPLITS_DIR, SEQ_LEN, DT
from utils.lorenz import simulate_lorenz63
from utils.observations import make_observations

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_traj", type=int, default=1500)
    p.add_argument("--steps",  type=int, default=200)
    p.add_argument("--dt",     type=float, default=DT)
    p.add_argument("--noise",  type=float, nargs="+", default=[0.05, 0.1, 0.5, 1.0])
    p.add_argument("--modes",  type=str, nargs="+", default=["x","xy","x2"])
    p.add_argument("--seed",   type=int, default=1234)
    args = p.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OBS_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # Warm-up then simulate trajectories
    warm = simulate_lorenz63(np.array([1.,1.,1.]), 2000, args.dt)[-1]
    trajs = []
    for _ in range(args.n_traj):
        ic = warm + rng.normal(0, 2.0, size=3)
        trajs.append(simulate_lorenz63(ic, args.steps, args.dt))
    trajectories = np.asarray(trajs)  # [N, T, 3]

    # Split
    train_traj = trajectories[:1000]
    test_traj  = trajectories[1000:]

    # Covariance & mean from training states
    train_states = train_traj.reshape(-1, 3)
    B = np.cov(train_states.T)
    eps = 1e-3
    B_reg = B + eps * np.eye(3)
    B_mean = train_states.mean(axis=0)

    # Save raw
    np.save(os.path.join(RAW_DIR, "train_traj.npy"), train_traj)
    np.save(os.path.join(RAW_DIR, "test_traj.npy"),  test_traj)
    np.save(os.path.join(RAW_DIR, "B.npy"),          B_reg)   # regularized
    np.save(os.path.join(RAW_DIR, "B_mean.npy"),     B_mean)

    # Save observations
    for mode in args.modes:
        for sig in args.noise:
            obs = make_observations(trajectories, mode=mode, sigma_noise=float(sig))
            np.save(os.path.join(OBS_DIR, f"obs_{mode}_n{sig}.npy"), obs)

    print("✅ Data written to")
    print("   RAW :", RAW_DIR)
    print("   OBS :", OBS_DIR)

if __name__ == "__main__":
    main()
