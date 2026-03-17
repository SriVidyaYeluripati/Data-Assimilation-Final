# src/data/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.config import RAW_DIR, OBS_DIR, SEQ_LEN, DT, make_run_dirs
import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


class AssimilationDataset(Dataset):
    """
    Unified dataset for both Resample and FixedMean setups.

    Args:
        trajectories (np.ndarray): shape [N, T, 3]
        observations (np.ndarray): shape [N, T, obs_dim]
        B (np.ndarray): background covariance matrix (3x3)
        B_mean (np.ndarray): mean background state (3,)
        seq_len (int): length of observation window
        split (str): "train" or "val"
        splits_dir (str): directory to save/load split indices
        background_mode (str): "resample" or "fixed"
        reuse_splits_dir (str): path to existing splits (used for fixed mode)
    """
    def __init__(self, trajectories, observations, B, B_mean,
                 seq_len=5, split="train", splits_dir=None,
                 background_mode="resample", reuse_splits_dir=None):
        self.seq_len = seq_len
        self.trajs = trajectories
        self.obs = observations
        self.B = B
        self.B_mean = B_mean
        self.split = split
        self.background_mode = background_mode

        # -------------------------------
        # Build samples (sliding windows)
        # -------------------------------
        self.samples = []
        N, T, _ = trajectories.shape
        for n in range(N):
            for t in range(seq_len - 1, T):
                x_true = trajectories[n, t]
                y_seq = observations[n, t - seq_len + 1:t + 1]
                self.samples.append((x_true, y_seq))

        # -------------------------------
        # Train/val split indices
        # -------------------------------
        n_total = len(self.samples)
        n_train = int(0.8 * n_total)

        # Case 1: reuse splits from another run
        if reuse_splits_dir is not None:
            train_idx = np.load(os.path.join(reuse_splits_dir, "train_indices.npy"))
            val_idx = np.load(os.path.join(reuse_splits_dir, "val_indices.npy"))
        else:
            # Case 2: create/load splits locally
            if splits_dir is not None:
                os.makedirs(splits_dir, exist_ok=True)
                train_idx_file = os.path.join(splits_dir, "train_indices.npy")
                val_idx_file = os.path.join(splits_dir, "val_indices.npy")
                if os.path.exists(train_idx_file) and os.path.exists(val_idx_file):
                    train_idx = np.load(train_idx_file)
                    val_idx = np.load(val_idx_file)
                else:
                    all_idx = np.arange(n_total)
                    np.random.shuffle(all_idx)
                    train_idx = all_idx[:n_train]
                    val_idx = all_idx[n_train:]
                    np.save(train_idx_file, train_idx)
                    np.save(val_idx_file, val_idx)
            else:
                # fallback: generate new random split
                all_idx = np.arange(n_total)
                np.random.shuffle(all_idx)
                train_idx = all_idx[:n_train]
                val_idx = all_idx[n_train:]

        self.indices = train_idx if split == "train" else val_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        x_true, y_seq = self.samples[real_idx]

        # -------------------------------
        # Background handling
        # -------------------------------
        if self.background_mode == "resample" and self.split == "train":
            # stochastic background
            x_b = np.random.multivariate_normal(self.B_mean, self.B)
        else:
            # deterministic mean
            x_b = self.B_mean.copy()

        return (
            torch.tensor(x_true, dtype=torch.float32),
            torch.tensor(y_seq, dtype=torch.float32),
            torch.tensor(x_b, dtype=torch.float32),
        )


class BaselineDataset(Dataset):
    """
    Dataset for the Baseline MLP (no background information).

    Returns (y_seq,) only — the true state is NOT included so that
    the baseline can be trained in a fully self-supervised manner,
    minimising only the observation cost J_o.

    Args:
        trajectories (np.ndarray): shape [N, T, 3]  (used for split bookkeeping only)
        observations (np.ndarray): shape [N, T, obs_dim]
        seq_len (int): length of observation window
        split (str): "train" or "val"
        splits_dir (str): directory to save/load split indices
    """
    def __init__(self, trajectories, observations,
                 seq_len=5, split="train", splits_dir=None):
        self.seq_len = seq_len
        self.obs = observations

        # Build samples from sliding windows
        self.samples = []
        N, T, _ = trajectories.shape
        for n in range(N):
            for t in range(seq_len - 1, T):
                y_seq = observations[n, t - seq_len + 1:t + 1]
                self.samples.append(y_seq)

        # Train / val split (reuse saved indices if available)
        n_total = len(self.samples)
        n_train = int(0.8 * n_total)

        if splits_dir is not None:
            os.makedirs(splits_dir, exist_ok=True)
            train_idx_file = os.path.join(splits_dir, "train_indices.npy")
            val_idx_file = os.path.join(splits_dir, "val_indices.npy")
            if os.path.exists(train_idx_file) and os.path.exists(val_idx_file):
                train_idx = np.load(train_idx_file)
                val_idx = np.load(val_idx_file)
            else:
                all_idx = np.arange(n_total)
                np.random.default_rng(42).shuffle(all_idx)
                train_idx = all_idx[:n_train]
                val_idx = all_idx[n_train:]
                np.save(train_idx_file, train_idx)
                np.save(val_idx_file, val_idx)
        else:
            all_idx = np.arange(n_total)
            np.random.default_rng(42).shuffle(all_idx)
            train_idx = all_idx[:n_train]
            val_idx = all_idx[n_train:]

        self.indices = train_idx if split == "train" else val_idx

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        y_seq = self.samples[real_idx]
        return torch.tensor(y_seq, dtype=torch.float32)

