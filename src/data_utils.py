"""Olivetti faces, splits, corruption, and nearest-neighbor classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_olivetti_faces


def load_olivetti(cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    bunch = fetch_olivetti_faces(
        data_home=str(cache_dir),
        shuffle=False,
        download_if_missing=True,
    )
    images = bunch.images.astype(np.float64)
    labels = bunch.target.astype(int)
    return images, labels


def split_by_subject(
    images: np.ndarray,
    labels: np.ndarray,
    n_train: int,
    n_val: int,
    n_test: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Split each subject independently so every identity appears in each part."""
    parts = {name: [] for name in ("train_x", "train_y", "val_x", "val_y", "test_x", "test_y")}
    for subject in np.unique(labels):
        index = np.where(labels == subject)[0]
        rng.shuffle(index)
        if len(index) < n_train + n_val + n_test:
            raise ValueError(f"Subject {subject} does not have enough images.")
        groups = {
            "train": index[:n_train],
            "val": index[n_train : n_train + n_val],
            "test": index[n_train + n_val : n_train + n_val + n_test],
        }
        for name, idx in groups.items():
            parts[f"{name}_x"].append(images[idx])
            parts[f"{name}_y"].append(labels[idx])
    return {name: np.concatenate(values, axis=0) for name, values in parts.items()}


def nearest_neighbor(train_features: np.ndarray, train_labels: np.ndarray, test_features: np.ndarray) -> np.ndarray:
    """1-NN with squared Euclidean distance, matching the papers' classifier."""
    test_sq = np.sum(test_features ** 2, axis=1, keepdims=True)
    train_sq = np.sum(train_features ** 2, axis=1)
    distances = test_sq + train_sq - 2.0 * test_features @ train_features.T
    return train_labels[np.argmin(distances, axis=1)]


def accuracy(predicted: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(predicted == truth))


def add_gaussian(images: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noisy = images + rng.normal(0.0, sigma, size=images.shape)
    return np.clip(noisy, 0.0, 1.0)


def add_salt_pepper(images: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    out = images.copy()
    n_images, height, width = out.shape
    n_pixels = height * width
    n_corrupt = int(round(amount * n_pixels))
    for i in range(n_images):
        chosen = rng.choice(n_pixels, size=n_corrupt, replace=False)
        flat = out[i].ravel()
        half = n_corrupt // 2
        flat[chosen[:half]] = 0.0
        flat[chosen[half:]] = 1.0
    return out


def add_block(images: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    out = images.copy()
    height, width = out.shape[1:]
    if block_size <= 0:
        return out
    max_row = height - block_size + 1
    max_col = width - block_size + 1
    for i in range(out.shape[0]):
        row = int(rng.integers(0, max_row))
        col = int(rng.integers(0, max_col))
        out[i, row : row + block_size, col : col + block_size] = 0.0
    return out


def reconstruction_ratio(original: np.ndarray, restored: np.ndarray) -> float:
    """Mean of 1 - ||A - A_hat||_F / ||A||_F over images."""
    ratios = []
    for image, recon in zip(original, restored):
        denominator = np.linalg.norm(image)
        ratios.append(1.0 - np.linalg.norm(image - recon) / max(denominator, 1e-12))
    return float(np.mean(ratios))
