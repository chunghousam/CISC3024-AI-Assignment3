"""Classical PCA, 2DPCA, and LDA baselines for face recognition."""

from __future__ import annotations

import numpy as np


def _positive_max_sign(vectors: np.ndarray) -> np.ndarray:
    """Flip each column so its largest-magnitude entry is positive."""
    out = vectors.copy()
    for j in range(out.shape[1]):
        pivot = int(np.argmax(np.abs(out[:, j])))
        if out[pivot, j] < 0:
            out[:, j] *= -1
    return out


class EigenfacesPCA:
    """Turk-Pentland eigenfaces: PCA on vectorized, mean-centered images."""

    def fit(self, images: np.ndarray, n_components: int) -> "EigenfacesPCA":
        images = np.asarray(images, dtype=np.float64)
        n, h, w = images.shape
        self.mean_image = images.mean(axis=0)
        x = images.reshape(n, h * w) - self.mean_image.ravel()
        # Rows of Vt are principal axes of the sample covariance.
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        k = min(n_components, vt.shape[0])
        self.components = _positive_max_sign(vt[:k].T)
        self.n_features = h * w
        self.image_shape = (h, w)
        return self

    def transform(self, images: np.ndarray, n_components: int | None = None) -> np.ndarray:
        images = np.asarray(images, dtype=np.float64)
        x = images.reshape(images.shape[0], -1) - self.mean_image.ravel()
        basis = self.components if n_components is None else self.components[:, :n_components]
        return x @ basis

    def reconstruct(self, images: np.ndarray, n_components: int) -> np.ndarray:
        coeff = self.transform(images, n_components)
        basis = self.components[:, :n_components]
        flat = coeff @ basis.T + self.mean_image.ravel()
        h, w = self.image_shape
        return flat.reshape(images.shape[0], h, w)


class TwoDPCA:
    """Yang et al. (2004) 2DPCA. Images stay matrices.

    The image covariance is G = sum_i A_i^T A_i. Its leading eigenvectors V
    give the feature matrix Y = A V.
    """

    def fit(self, images: np.ndarray, n_components: int) -> "TwoDPCA":
        images = np.asarray(images, dtype=np.float64)
        self.mean_image = images.mean(axis=0)
        centered = images - self.mean_image
        # G[i, j] = sum_n,m centered[n, m, i] * centered[n, m, j]
        gram = np.einsum("nmi,nmj->ij", centered, centered)
        evals, evecs = np.linalg.eigh(gram)
        order = np.argsort(evals)[::-1]
        k = min(n_components, evecs.shape[1])
        self.components = _positive_max_sign(evecs[:, order[:k]])
        self.eigenvalues = evals[order[:k]]
        self.image_shape = images.shape[1:]
        return self

    def transform(self, images: np.ndarray, n_components: int | None = None) -> np.ndarray:
        images = np.asarray(images, dtype=np.float64)
        centered = images - self.mean_image
        basis = self.components if n_components is None else self.components[:, :n_components]
        features = np.einsum("nmj,jk->nmk", centered, basis)
        return features.reshape(images.shape[0], -1)

    def reconstruct(self, images: np.ndarray, n_components: int) -> np.ndarray:
        centered = np.asarray(images, dtype=np.float64) - self.mean_image
        basis = self.components[:, :n_components]
        # A_hat = A V V^T, then add the mean face back.
        projected = np.einsum("nmj,jk,lk->nml", centered, basis, basis)
        return projected + self.mean_image


class ImageLDA:
    """Linear discriminant analysis for vectorized images.

    Face images have far more pixels than training samples, so the within-class
    scatter is singular. This uses the SVD form of LDA: whiten the within-class
    variation, then take the leading singular vectors of the scaled class means.
    At most C - 1 directions are returned. Projection subtracts the global mean.
    """

    def fit(self, images: np.ndarray, labels: np.ndarray, n_components: int) -> "ImageLDA":
        images = np.asarray(images, dtype=np.float64)
        labels = np.asarray(labels)
        n_samples, height, width = images.shape
        x = images.reshape(n_samples, height * width)
        classes = np.unique(labels)
        n_classes = len(classes)
        if n_samples <= n_classes:
            raise ValueError("LDA needs more samples than classes.")

        means = np.zeros((n_classes, x.shape[1]), dtype=np.float64)
        centered_blocks = []
        counts = []
        for index, subject in enumerate(classes):
            block = x[labels == subject]
            means[index] = block.mean(axis=0)
            centered_blocks.append(block - means[index])
            counts.append(block.shape[0])
        priors = np.asarray(counts, dtype=np.float64) / n_samples
        self.mean_vector = priors @ means

        centered = np.concatenate(centered_blocks, axis=0)
        std = centered.std(axis=0)
        std[std == 0.0] = 1.0
        within = np.sqrt(1.0 / (n_samples - n_classes)) * (centered / std)
        _, singular, vt = np.linalg.svd(within, full_matrices=False)
        rank = int(np.sum(singular > 1e-4))
        if rank < 1:
            raise RuntimeError("Within-class SVD rank is zero.")
        within_map = (vt[:rank] / std).T / singular[:rank]

        between_scale = np.sqrt(n_samples * priors / (n_classes - 1.0))
        class_centers = (between_scale * (means - self.mean_vector).T).T
        _, between_singular, between_vt = np.linalg.svd(class_centers @ within_map, full_matrices=False)
        rank_between = int(np.sum(between_singular > 1e-4 * between_singular[0]))
        n_keep = min(n_components, rank_between, n_classes - 1)
        self.components = within_map @ between_vt.T[:, :n_keep]
        self.image_shape = (height, width)
        self.n_classes = n_classes
        return self

    def transform(self, images: np.ndarray, n_components: int | None = None) -> np.ndarray:
        images = np.asarray(images, dtype=np.float64)
        x = images.reshape(images.shape[0], -1)
        basis = self.components if n_components is None else self.components[:, :n_components]
        return (x - self.mean_vector) @ basis
