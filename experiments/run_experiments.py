"""Train PCA, 2DPCA, LDA, and bilateral generalized 2DPCA on Olivetti faces.

The selected method is the real-matrix form of BiG2DQPCA
(Zhao, Jia, Gong, Zhang, arXiv:2306.07045, 2023).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import EigenfacesPCA, ImageLDA, TwoDPCA
from src.big2dpca import BilateralGeneralized2DPCA, fit_right_basis, subspace_angle_degrees
from src.data_utils import (
    accuracy,
    add_block,
    add_gaussian,
    add_salt_pepper,
    load_olivetti,
    nearest_neighbor,
    reconstruction_ratio,
    split_by_subject,
)

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
CACHE = ROOT / ".cache"

PCA_KS = [5, 10, 20, 30, 40, 60, 80, 120]
DPCA_KS = [1, 2, 4, 6, 8, 10, 12, 16]
LDA_KS = [1, 2, 5, 10, 15, 20, 25, 30, 39]
BIG_KS = [1, 2, 4, 6, 8, 10, 12, 16]
WEIGHTINGS = ("none", "left", "right", "both")
BLOCK_SIZES = [0, 8, 16, 24, 32]
N_SPLITS = 5
N_TRAIN, N_VAL, N_TEST = 5, 2, 3

COLORS = {
    "PCA": "#0072B2",
    "2DPCA": "#E69F00",
    "LDA": "#009E73",
    "BiG2DPCA": "#D55E00",
}


def check_matches_closed_form() -> dict[str, float]:
    """The s = p = 2 model, and a tight run of Algorithm 2, must match the covariance eigenvectors."""
    rng = np.random.default_rng(0)
    images = rng.normal(size=(36, 12, 9))
    centered = images - images.mean(axis=0)
    model = BilateralGeneralized2DPCA(s=2, p=2, seed=1).fit(images, n_components=3)

    gram_right = np.einsum("nmi,nmj->ij", centered, centered)
    evals, evecs = eigh(gram_right)
    truth_right = evecs[:, np.argsort(evals)[::-1][:3]]

    gram_left = np.einsum("nai,nci->ac", centered, centered)
    evals, evecs = eigh(gram_left)
    truth_left = evecs[:, np.argsort(evals)[::-1][:3]]

    right_angle = subspace_angle_degrees(model.right, truth_right)
    left_angle = subspace_angle_degrees(model.left, truth_left)
    iterated, _, _ = fit_right_basis(centered, 3, 2.0, 2.0, np.random.default_rng(1), tol=1e-8, max_iter=400)
    iterated_angle = subspace_angle_degrees(iterated, truth_right)
    print(
        f"Self-check principal angle (degrees): closed-form right={right_angle:.4f}, "
        f"left={left_angle:.4f}, iterative={iterated_angle:.4f}"
    )
    if max(right_angle, left_angle, iterated_angle) > 0.5:
        raise RuntimeError("s=p=2 solver does not match the covariance eigenbasis.")
    return {
        "right_angle_deg": right_angle,
        "left_angle_deg": left_angle,
        "iterative_angle_deg": iterated_angle,
    }


def nn_accuracy(train_f, train_y, query_f, query_y) -> float:
    return accuracy(nearest_neighbor(train_f, train_y, query_f), query_y)


def best_key(scores: dict) -> tuple:
    """Highest score, then the smaller configuration (k, and weighting name)."""
    return min(scores, key=lambda key: (-scores[key], key if isinstance(key, tuple) else (key, "")))


def mean_std(rows: list[float]) -> dict[str, float]:
    arr = np.asarray(rows, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0)}


def curve_stats(per_split: list[dict[int, float]], grid: list[int]) -> dict[str, list[float]]:
    return {
        "k": grid,
        "mean": [float(np.mean([row[k] for row in per_split])) for k in grid],
        "std": [float(np.std([row[k] for row in per_split], ddof=1)) for k in grid],
    }


def run_parameter_study(split: dict, k: int = 8) -> dict:
    """Vary (s, p) on one split. This is a sensitivity check, not model selection."""
    rng = np.random.default_rng(11)
    occluded = add_block(split["test_x"], 24, rng)
    settings = [(2.0, 2.0), (2.0, 1.0), (2.0, np.inf), (1.0, 2.0), (2.0, 0.5), (1.0, 1.0)]
    rows = []
    for s, p in settings:
        label_p = "inf" if np.isinf(p) else str(p)
        started = time.perf_counter()
        try:
            model = BilateralGeneralized2DPCA(s=s, p=p, seed=3).fit(split["train_x"], k)
            if not np.isfinite(model.right).all() or not np.isfinite(model.left).all():
                raise FloatingPointError("non-finite projection")
            clean = nn_accuracy(
                model.transform(split["train_x"], k),
                split["train_y"],
                model.transform(split["test_x"], k),
                split["test_y"],
            )
            blocked = nn_accuracy(
                model.transform(split["train_x"], k),
                split["train_y"],
                model.transform(occluded, k),
                split["test_y"],
            )
            status = "ok"
        except Exception as exc:  # numerical failure is a result worth reporting
            clean = blocked = None
            status = str(exc)
        rows.append(
            {
                "s": s,
                "p": label_p,
                "clean_accuracy": clean,
                "block_accuracy": blocked,
                "seconds": time.perf_counter() - started,
                "status": status,
            }
        )
        print(f"  parameter s={s}, p={label_p}: clean={clean}, block={blocked}, status={status}")
    return {"k": k, "rows": rows}


def evaluate() -> dict:
    images, labels = load_olivetti(CACHE)
    print(f"Olivetti loaded: {images.shape[0]} images, shape {images.shape[1:]}, {len(np.unique(labels))} subjects")

    split_curves = {name: [] for name in ("PCA", "2DPCA", "LDA", "BiG2DPCA")}
    selected_rows = []
    noise_rows = []
    block_rows = []
    weighting_rows = []
    fit_times = []
    stored_split0 = None

    for split_id in range(N_SPLITS):
        rng = np.random.default_rng(2026 + split_id)
        parts = split_by_subject(images, labels, N_TRAIN, N_VAL, N_TEST, rng)
        print(f"\nSplit {split_id + 1}/{N_SPLITS}")
        times = {}

        started = time.perf_counter()
        pca = EigenfacesPCA().fit(parts["train_x"], max(PCA_KS))
        times["PCA"] = time.perf_counter() - started

        started = time.perf_counter()
        dpca = TwoDPCA().fit(parts["train_x"], max(DPCA_KS))
        times["2DPCA"] = time.perf_counter() - started

        started = time.perf_counter()
        fisher = ImageLDA().fit(parts["train_x"], parts["train_y"], max(LDA_KS))
        times["LDA"] = time.perf_counter() - started

        started = time.perf_counter()
        big = BilateralGeneralized2DPCA(s=2.0, p=2.0, seed=100 + split_id).fit(parts["train_x"], max(BIG_KS))
        times["BiG2DPCA"] = time.perf_counter() - started
        fit_times.append(times)
        print("  fit seconds: " + ", ".join(f"{name}={seconds:.2f}" for name, seconds in times.items()))

        def collect(method, grid, transform):
            val_scores = {}
            test_scores = {}
            for k in grid:
                train_f = transform(parts["train_x"], k)
                val_scores[k] = nn_accuracy(train_f, parts["train_y"], transform(parts["val_x"], k), parts["val_y"])
                test_scores[k] = nn_accuracy(train_f, parts["train_y"], transform(parts["test_x"], k), parts["test_y"])
            return val_scores, test_scores

        pca_val, pca_test = collect("PCA", PCA_KS, pca.transform)
        dpca_val, dpca_test = collect("2DPCA", DPCA_KS, dpca.transform)
        fisher_val, fisher_test = collect("LDA", LDA_KS, fisher.transform)

        big_val = {}
        big_test_none = {}
        for k in BIG_KS:
            train_none = big.transform(parts["train_x"], k, "none")
            big_test_none[k] = nn_accuracy(train_none, parts["train_y"], big.transform(parts["test_x"], k, "none"), parts["test_y"])
            for weighting in WEIGHTINGS:
                train_f = big.transform(parts["train_x"], k, weighting)
                big_val[(k, weighting)] = nn_accuracy(
                    train_f, parts["train_y"], big.transform(parts["val_x"], k, weighting), parts["val_y"]
                )

        split_curves["PCA"].append(pca_test)
        split_curves["2DPCA"].append(dpca_test)
        split_curves["LDA"].append(fisher_test)
        split_curves["BiG2DPCA"].append(big_test_none)

        chosen = {
            "PCA": best_key(pca_val),
            "2DPCA": best_key(dpca_val),
            "LDA": best_key(fisher_val),
            "BiG2DPCA": best_key(big_val),
        }
        chosen_test = {
            "PCA": pca_test[chosen["PCA"]],
            "2DPCA": dpca_test[chosen["2DPCA"]],
            "LDA": fisher_test[chosen["LDA"]],
            "BiG2DPCA": nn_accuracy(
                big.transform(parts["train_x"], chosen["BiG2DPCA"][0], chosen["BiG2DPCA"][1]),
                parts["train_y"],
                big.transform(parts["test_x"], chosen["BiG2DPCA"][0], chosen["BiG2DPCA"][1]),
                parts["test_y"],
            ),
        }
        print("  selected: " + ", ".join(f"{name}={value}" for name, value in chosen.items()))
        print("  test acc: " + ", ".join(f"{name}={value:.3f}" for name, value in chosen_test.items()))

        noise_rng = np.random.default_rng(7000 + split_id)
        queries = {
            "clean": parts["test_x"],
            "gaussian_0.20": add_gaussian(parts["test_x"], 0.20, noise_rng),
            "salt_pepper_0.25": add_salt_pepper(parts["test_x"], 0.25, noise_rng),
            "block_24": add_block(parts["test_x"], 24, noise_rng),
        }
        noise_acc = {}
        projectors = {
            "PCA": lambda imgs, k: pca.transform(imgs, k),
            "2DPCA": lambda imgs, k: dpca.transform(imgs, k),
            "LDA": lambda imgs, k: fisher.transform(imgs, k),
            "BiG2DPCA": lambda imgs, k: big.transform(imgs, chosen["BiG2DPCA"][0], chosen["BiG2DPCA"][1]),
        }
        for condition, query in queries.items():
            noise_acc[condition] = {}
            for name in ("PCA", "2DPCA", "LDA"):
                k = chosen[name]
                noise_acc[condition][name] = nn_accuracy(
                    projectors[name](parts["train_x"], k), parts["train_y"], projectors[name](query, k), parts["test_y"]
                )
            k_big, weighting = chosen["BiG2DPCA"]
            noise_acc[condition]["BiG2DPCA"] = nn_accuracy(
                big.transform(parts["train_x"], k_big, weighting),
                parts["train_y"],
                big.transform(query, k_big, weighting),
                parts["test_y"],
            )

        block_acc = {size: {} for size in BLOCK_SIZES}
        for size in BLOCK_SIZES:
            query = parts["test_x"] if size == 0 else add_block(parts["test_x"], size, np.random.default_rng(9000 + 17 * split_id + size))
            for name in ("PCA", "2DPCA", "LDA"):
                k = chosen[name]
                block_acc[size][name] = nn_accuracy(
                    projectors[name](parts["train_x"], k), parts["train_y"], projectors[name](query, k), parts["test_y"]
                )
            k_big, weighting = chosen["BiG2DPCA"]
            block_acc[size]["BiG2DPCA"] = nn_accuracy(
                big.transform(parts["train_x"], k_big, weighting),
                parts["train_y"],
                big.transform(query, k_big, weighting),
                parts["test_y"],
            )

        weight_at_8 = {}
        for weighting in WEIGHTINGS:
            train_f = big.transform(parts["train_x"], 8, weighting)
            weight_at_8[weighting] = {
                "val": nn_accuracy(train_f, parts["train_y"], big.transform(parts["val_x"], 8, weighting), parts["val_y"]),
                "test": nn_accuracy(train_f, parts["train_y"], big.transform(parts["test_x"], 8, weighting), parts["test_y"]),
            }

        selected_rows.append({"chosen": {k: (list(v) if isinstance(v, tuple) else v) for k, v in chosen.items()}, "test": chosen_test})
        noise_rows.append(noise_acc)
        block_rows.append({str(size): scores for size, scores in block_acc.items()})
        weighting_rows.append(weight_at_8)

        if split_id == 0:
            centered = parts["train_x"] - parts["train_x"].mean(axis=0)
            _, _, right_hist = fit_right_basis(
                centered, 1, 2.0, 2.0, np.random.default_rng(0), tol=1e-8, max_iter=60
            )
            _, _, left_hist = fit_right_basis(
                centered.transpose(0, 2, 1), 1, 2.0, 2.0, np.random.default_rng(0), tol=1e-8, max_iter=60
            )
            stored_split0 = {
                "parts": parts,
                "pca": pca,
                "dpca": dpca,
                "fisher": fisher,
                "big": big,
                "occluded": queries["block_24"],
                "iter_right": right_hist[0],
                "iter_left": left_hist[0],
            }

    print("\nParameter study on split seed 2026")
    first = split_by_subject(images, labels, N_TRAIN, N_VAL, N_TEST, np.random.default_rng(2026))
    parameter_study = run_parameter_study(first, k=8)

    summary = {
        "dataset": {
            "name": "Olivetti faces",
            "images": int(images.shape[0]),
            "height": int(images.shape[1]),
            "width": int(images.shape[2]),
            "subjects": int(len(np.unique(labels))),
            "images_per_subject": 10,
            "protocol": {
                "splits": N_SPLITS,
                "train_per_subject": N_TRAIN,
                "validation_per_subject": N_VAL,
                "test_per_subject": N_TEST,
                "classifier": "1-nearest neighbor, Euclidean distance",
            },
        },
        "self_check": check_matches_closed_form(),
        "test_curves_unweighted_big": {name: curve_stats(rows, grid) for name, rows, grid in (
            ("PCA", split_curves["PCA"], PCA_KS),
            ("2DPCA", split_curves["2DPCA"], DPCA_KS),
            ("LDA", split_curves["LDA"], LDA_KS),
            ("BiG2DPCA", split_curves["BiG2DPCA"], BIG_KS),
        )},
        "validation_selected": selected_rows,
        "selected_summary": {},
        "noise": {},
        "block": {},
        "weighting_k8": {},
        "fit_seconds": {
            name: mean_std([row[name] for row in fit_times]) for name in ("PCA", "2DPCA", "LDA", "BiG2DPCA")
        },
        "parameter_study": parameter_study,
        "reconstruction": {},
    }

    for name in ("PCA", "2DPCA", "LDA", "BiG2DPCA"):
        summary["selected_summary"][name] = mean_std([row["test"][name] for row in selected_rows])
        choices = [row["chosen"][name] for row in selected_rows]
        summary["selected_summary"][name]["choices"] = choices

    conditions = list(noise_rows[0].keys())
    for condition in conditions:
        summary["noise"][condition] = {
            name: mean_std([row[condition][name] for row in noise_rows])
            for name in ("PCA", "2DPCA", "LDA", "BiG2DPCA")
        }
    for size in BLOCK_SIZES:
        summary["block"][str(size)] = {
            name: mean_std([row[str(size)][name] for row in block_rows])
            for name in ("PCA", "2DPCA", "LDA", "BiG2DPCA")
        }
    for weighting in WEIGHTINGS:
        summary["weighting_k8"][weighting] = {
            "val": mean_std([row[weighting]["val"] for row in weighting_rows]),
            "test": mean_std([row[weighting]["test"] for row in weighting_rows]),
        }

    summary["reconstruction"] = reconstruction_report(stored_split0)
    make_figures(images, labels, summary, stored_split0)
    return summary


def reconstruction_report(stored: dict) -> dict:
    train = stored["parts"]["train_x"]
    ratios = {"k": DPCA_KS, "PCA": [], "2DPCA": [], "BiG2DPCA": []}
    for k in DPCA_KS:
        ratios["2DPCA"].append(reconstruction_ratio(train, stored["dpca"].reconstruct(train, k)))
        ratios["BiG2DPCA"].append(reconstruction_ratio(train, stored["big"].reconstruct(train, k)))
    for k in (10, 40, 80, 120):
        ratios.setdefault("PCA_k", []).append(k)
        ratios.setdefault("PCA_ratio", []).append(reconstruction_ratio(train, stored["pca"].reconstruct(train, k)))
    # Keep a PCA curve on its own grid as well.
    ratios["PCA_grid"] = PCA_KS
    ratios["PCA"] = [reconstruction_ratio(train, stored["pca"].reconstruct(train, k)) for k in PCA_KS]
    return ratios


def _face_grid(images: np.ndarray, path: Path, title: str, columns: int) -> None:
    count = images.shape[0]
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.15, rows * 1.25))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, image in zip(axes.ravel(), images):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def make_figures(images, labels, summary, stored) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
    })

    gallery = []
    for subject in range(8):
        gallery.extend(images[labels == subject][:5])
    _face_grid(np.stack(gallery), FIGURES / "sample_faces.png", "Olivetti samples: 8 subjects, 5 images each", 5)

    eigen = stored["pca"].components[:, :8].T.reshape(8, 64, 64)
    eigen = (eigen - eigen.min(axis=(1, 2), keepdims=True)) / np.ptp(eigen, axis=(1, 2), keepdims=True)
    _face_grid(eigen, FIGURES / "eigenfaces.png", "First 8 eigenfaces (PCA)", 4)

    bases = []
    for i in range(8):
        image = np.outer(stored["big"].left[:, i], stored["big"].right[:, i])
        image = (image - image.min()) / max(np.ptp(image), 1e-12)
        bases.append(image)
    _face_grid(np.stack(bases), FIGURES / "bilateral_bases.png", r"First 8 bilateral rank-1 bases  $u_i v_i^{\top}$", 4)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), sharey=True)
    grids = {"PCA": PCA_KS, "2DPCA": DPCA_KS, "LDA": LDA_KS, "BiG2DPCA": BIG_KS}
    xlab = {
        "PCA": "Number of eigenfaces",
        "2DPCA": "Number of 2DPCA vectors",
        "LDA": "Number of discriminant directions",
        "BiG2DPCA": "Vectors per direction (k x k features)",
    }
    for ax, name in zip(axes.ravel(), ("PCA", "2DPCA", "LDA", "BiG2DPCA")):
        curve = summary["test_curves_unweighted_big"][name]
        k = np.asarray(curve["k"])
        mean = np.asarray(curve["mean"])
        std = np.asarray(curve["std"])
        ax.plot(k, mean, marker="o", color=COLORS[name], label=name)
        ax.fill_between(k, mean - std, mean + std, color=COLORS[name], alpha=0.2)
        ax.set_title(name)
        ax.set_xlabel(xlab[name])
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("Test accuracy")
    fig.suptitle("Test accuracy versus subspace size (mean ± std, 5 splits)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES / "accuracy_vs_components.png", bbox_inches="tight")
    plt.close(fig)

    conditions = ["clean", "gaussian_0.20", "salt_pepper_0.25", "block_24"]
    cond_labels = ["Clean", "Gaussian\nσ=0.20", "Salt-pepper\n25%", "24×24 block"]
    methods = ["PCA", "2DPCA", "LDA", "BiG2DPCA"]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(conditions))
    width = 0.18
    for offset, name in enumerate(methods):
        means = [summary["noise"][c][name]["mean"] for c in conditions]
        stds = [summary["noise"][c][name]["std"] for c in conditions]
        ax.bar(x + (offset - 1.5) * width, means, width, yerr=stds, label=name, color=COLORS[name], capsize=3)
    ax.set_xticks(x, cond_labels)
    ax.set_ylabel("Test accuracy at validation-chosen size")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_title("Robustness when only the test images are corrupted")
    fig.tight_layout()
    fig.savefig(FIGURES / "noise_bars.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name in methods:
        means = [summary["block"][str(size)][name]["mean"] for size in BLOCK_SIZES]
        stds = [summary["block"][str(size)][name]["std"] for size in BLOCK_SIZES]
        ax.errorbar(BLOCK_SIZES, means, yerr=stds, marker="o", label=name, color=COLORS[name], capsize=3)
    ax.set_xlabel("Occluding block size (pixels)")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.set_title("Accuracy versus block occlusion")
    fig.tight_layout()
    fig.savefig(FIGURES / "block_sweep.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    recon = summary["reconstruction"]
    height = 64
    ax.plot(recon["PCA_grid"], recon["PCA"], marker="o", label="PCA", color=COLORS["PCA"])
    ax.plot([k * height for k in recon["k"]], recon["2DPCA"], marker="o", label="2DPCA", color=COLORS["2DPCA"])
    ax.plot([k * k for k in recon["k"]], recon["BiG2DPCA"], marker="o", label="BiG2DPCA", color=COLORS["BiG2DPCA"])
    ax.set_xlabel("Number of stored feature coefficients")
    ax.set_ylabel("Mean reconstruction ratio")
    ax.set_title("Training-set reconstruction on split 1")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "reconstruction_curve.png", bbox_inches="tight")
    plt.close(fig)

    test = stored["parts"]["test_x"]
    show = test[:6]
    occluded = stored["occluded"][:6]
    panels = [
        ("Original", show),
        ("24×24 block", occluded),
        ("PCA recon. k=40", stored["pca"].reconstruct(occluded, 40)),
        ("2DPCA recon. k=8", stored["dpca"].reconstruct(occluded, 8)),
        ("BiG2DPCA recon. k=8", stored["big"].reconstruct(occluded, 8)),
    ]
    fig, axes = plt.subplots(len(panels), 6, figsize=(8.8, 8.0))
    for row, (title, batch) in enumerate(panels):
        for col in range(6):
            axes[row, col].imshow(np.clip(batch[col], 0, 1), cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
        axes[row, 0].text(
            -0.08, 0.5, title, transform=axes[row, 0].transAxes,
            ha="right", va="center", fontsize=8, clip_on=False,
        )
    fig.suptitle("Reconstructing block-occluded test faces from the clean training subspace", fontsize=12)
    fig.subplots_adjust(left=0.20, right=0.99, top=0.92, bottom=0.02, wspace=0.05, hspace=0.08)
    fig.savefig(FIGURES / "reconstruction_gallery.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    right_hist = stored["iter_right"]
    left_hist = stored["iter_left"]
    ax.plot(np.arange(len(right_hist)), right_hist, marker="o", label="First right vector", color=COLORS["BiG2DPCA"])
    ax.plot(np.arange(len(left_hist)), left_hist, marker="s", label="First left vector", color=COLORS["PCA"])
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Objective  $\sum_i \|F_i w\|_2^2$")
    ax.set_title("Convergence of the first projection vector (s = p = 2)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "convergence.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    x = np.arange(len(WEIGHTINGS))
    val_mean = [summary["weighting_k8"][w]["val"]["mean"] for w in WEIGHTINGS]
    test_mean = [summary["weighting_k8"][w]["test"]["mean"] for w in WEIGHTINGS]
    val_std = [summary["weighting_k8"][w]["val"]["std"] for w in WEIGHTINGS]
    test_std = [summary["weighting_k8"][w]["test"]["std"] for w in WEIGHTINGS]
    ax.bar(x - 0.16, val_mean, 0.32, yerr=val_std, label="Validation", color="#56B4E9", capsize=3)
    ax.bar(x + 0.16, test_mean, 0.32, yerr=test_std, label="Test", color=COLORS["BiG2DPCA"], capsize=3)
    ax.set_xticks(x, ["Unweighted", "Left", "Right", "Both"])
    ax.set_ylabel("Accuracy at k = 8")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.set_title("Data-driven weighting of BiG2DPCA features")
    fig.tight_layout()
    fig.savefig(FIGURES / "weighting.png", bbox_inches="tight")
    plt.close(fig)

    train_y = stored["parts"]["train_y"]
    keep = train_y < 8
    feat = stored["fisher"].transform(stored["parts"]["train_x"], 2)[keep]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    cmap = plt.get_cmap("tab10")
    for subject in range(8):
        points = feat[train_y[keep] == subject]
        ax.scatter(points[:, 0], points[:, 1], s=28, color=cmap(subject), label=f"ID {subject}")
    ax.set_xlabel("First discriminant coordinate")
    ax.set_ylabel("Second discriminant coordinate")
    ax.set_title("LDA: 8 identities in the training split")
    ax.legend(ncol=2, fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "lda_scatter.png", bbox_inches="tight")
    plt.close(fig)

    rows = summary["parameter_study"]["rows"]
    # Heatmap only for finite results is drawn as a grouped bar instead, which
    # stays readable when one (s, p) pair fails numerically.
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    labels = [f"s={row['s']:g}\np={row['p']}" for row in rows]
    clean = [0 if row["clean_accuracy"] is None else row["clean_accuracy"] for row in rows]
    blocked = [0 if row["block_accuracy"] is None else row["block_accuracy"] for row in rows]
    x = np.arange(len(rows))
    ax.bar(x - 0.16, clean, 0.32, label="Clean test", color=COLORS["PCA"])
    ax.bar(x + 0.16, blocked, 0.32, label="24×24 block", color=COLORS["BiG2DPCA"])
    ax.set_xticks(x, labels)
    ax.set_ylabel("Accuracy on one split, k = 8")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.set_title("Sensitivity of BiG2DPCA to the norm pair (s, p)")
    fig.tight_layout()
    fig.savefig(FIGURES / "parameter_study.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    print("Checking the s = p = 2 case against the covariance eigenbasis...")
    check = check_matches_closed_form()
    summary = evaluate()
    summary["self_check"] = check
    out = RESULTS / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
