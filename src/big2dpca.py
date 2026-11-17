"""Real-matrix bilateral generalized 2DPCA.

This is the real special case of BiG2DQPCA (Zhao, Jia, Gong, and Zhang,
arXiv:2306.07045, 2023). A grayscale image is a real matrix, which is the
quaternion matrix whose three imaginary parts are zero. Algorithm 2 of that
paper then reduces to the iteration below. Left and right projections are
separable, so the left basis is obtained by running the same right-vector
solver on transposed images.
"""

from __future__ import annotations

import numpy as np


def lp_norm(vector: np.ndarray, p: float) -> float:
    if np.isinf(p):
        return float(np.max(np.abs(vector)))
    return float(np.sum(np.abs(vector) ** p) ** (1.0 / p))


def normalize_lp(vector: np.ndarray, p: float) -> np.ndarray:
    norm = lp_norm(vector, p)
    if norm < 1e-15:
        vector = np.ones_like(vector)
        norm = lp_norm(vector, p)
    return vector / norm


def _signed_power(values: np.ndarray, exponent: float) -> np.ndarray:
    """Return |z|^exponent * sign(z), with 0^0 treated as 0."""
    magnitude = np.abs(values)
    if exponent == 0:
        powered = np.ones_like(magnitude)
        powered[magnitude == 0] = 0.0
    else:
        powered = magnitude ** exponent
    return np.sign(values) * powered


def _objective(images: np.ndarray, vector: np.ndarray, s: float) -> float:
    projected = np.einsum("nmi,i->nm", images, vector)
    return float(np.sum(np.abs(projected) ** s))


def _qr_orthogonalize(vector: np.ndarray, previous: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Orthonormalize `vector` against columns of `previous` by QR, as in Algorithm 2."""
    if previous.size == 0:
        matrix = vector.reshape(-1, 1)
    else:
        matrix = np.column_stack([previous, vector])
    q, r = np.linalg.qr(matrix)
    if abs(r[-1, -1]) < 1e-8:
        # The new direction collapsed into the span already removed by deflation.
        candidate = vector.copy()
        for _ in range(8):
            candidate = rng.normal(size=vector.shape[0])
            if previous.size:
                candidate = candidate - previous @ (previous.T @ candidate)
            if np.linalg.norm(candidate) > 1e-8:
                break
        q, r = np.linalg.qr(np.column_stack([previous, candidate]) if previous.size else candidate.reshape(-1, 1))
    direction = q[:, -1]
    if r[-1, -1] < 0:
        direction = -direction
    pivot = int(np.argmax(np.abs(direction)))
    if direction[pivot] < 0:
        direction = -direction
    return direction


def fit_right_basis(
    images: np.ndarray,
    n_components: int,
    s: float,
    p: float,
    rng: np.random.Generator,
    tol: float = 1e-4,
    max_iter: int = 200,
) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    """Compute successive right projection vectors of centered images.

    Parameters
    ----------
    images:
        Array of shape (N, m, n), already mean-centered.
    n_components:
        Number of right vectors k2. Must be smaller than n.
    s, p:
        Exponents in the objective ||F w||_s^s and the constraint ||w||_p = 1.
    """
    working = np.array(images, dtype=np.float64, copy=True)
    n_samples, _, width = working.shape
    if n_components >= width:
        raise ValueError("n_components must be smaller than the image width.")

    basis = np.zeros((width, n_components), dtype=np.float64)
    weights = np.zeros(n_components, dtype=np.float64)
    histories: list[list[float]] = []

    for t in range(n_components):
        w = normalize_lp(rng.normal(size=width), p)
        w0 = w.copy()
        value = _objective(working, w, s)
        history = [value]
        delta = 1.0
        iteration = 0
        while delta > tol and iteration < max_iter:
            hat = np.einsum("nmi,i->nm", working, w)
            coeff = _signed_power(hat, s - 1.0)
            y = np.einsum("nmi,nm->i", working, coeff)
            if 0 < p < 1:
                y = np.abs(w0) * (np.abs(w) + 1e-12) ** (1.0 - p) * y
                w_next = normalize_lp(y, p)
            elif p == 1:
                pivot = int(np.argmax(np.abs(y)))
                w_next = np.zeros_like(y)
                w_next[pivot] = 1.0 if y[pivot] == 0 else np.sign(y[pivot])
            elif np.isinf(p):
                w_next = np.sign(y)
                if not np.any(w_next):
                    w_next = w.copy()
            else:
                q = p / (p - 1.0)
                y = _signed_power(y, q - 1.0)
                w_next = normalize_lp(y, p)

            value_next = _objective(working, w_next, s)
            delta = abs(value_next - value) / max(abs(value), 1e-12)
            w = w_next
            value = value_next
            history.append(value)
            iteration += 1

        direction = _qr_orthogonalize(w, basis[:, :t], rng)
        basis[:, t] = direction
        weights[t] = _objective(working, direction, s)
        histories.append(history)
        # Deflate: F <- F (I - v v^T). Previous directions were already removed.
        coeff = np.einsum("nmi,i->nm", working, direction)
        working -= coeff[:, :, None] * direction[None, None, :]

    return basis, weights, histories


def _exact_l2_basis(images: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    """Top eigenvectors of sum_i A_i^T A_i, the exact s = p = 2 solution.

    Algorithm 2 is power iteration on this matrix when s = p = 2. The
    eigen-decomposition is the same optimum and avoids stopping early when
    neighboring eigenvalues are close.
    """
    gram = np.einsum("nmi,nmj->ij", images, images)
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1][:n_components]
    basis = evecs[:, order]
    for j in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, j])))
        if basis[pivot, j] < 0:
            basis[:, j] *= -1
    weights = evals[order]
    # One-step history: the closed form is already optimal.
    history = [[float(weight)] for weight in weights]
    return basis, weights.astype(np.float64), history


class BilateralGeneralized2DPCA:
    """Bilateral generalized 2DPCA with optional data-driven feature weights.

    Feature matrix of one centered image A:
        none:   P = U^T A V
        left:   P = (U diag(f(w_left)))^T A V
        right:  P = U^T A (V diag(f(w_right)))
        both:   P = (U diag(f(w_left)))^T A (V diag(f(w_right)))
    """

    WEIGHTINGS = ("none", "left", "right", "both")

    def __init__(self, s: float = 2.0, p: float = 2.0, seed: int = 0, tol: float = 1e-4, max_iter: int = 80):
        self.s = float(s)
        self.p = float(p) if not np.isinf(p) else np.inf
        self.seed = seed
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, images: np.ndarray, n_components: int) -> "BilateralGeneralized2DPCA":
        images = np.asarray(images, dtype=np.float64)
        self.mean_image = images.mean(axis=0)
        centered = images - self.mean_image
        rng = np.random.default_rng(self.seed)
        if self.s == 2.0 and self.p == 2.0:
            solve = lambda batch: _exact_l2_basis(batch, n_components)
        else:
            solve = lambda batch: fit_right_basis(
                batch, n_components, self.s, self.p, rng, tol=self.tol, max_iter=self.max_iter
            )
        # Left vectors live in the row space, so solve the right problem on A^T.
        self.left, self.left_weights, self.left_history = solve(centered.transpose(0, 2, 1))
        self.right, self.right_weights, self.right_history = solve(centered)
        self.image_shape = images.shape[1:]
        self.used_closed_form = self.s == 2.0 and self.p == 2.0
        return self

    def _weighted_factors(self, n_components: int, weighting: str) -> tuple[np.ndarray, np.ndarray]:
        u = self.left[:, :n_components]
        v = self.right[:, :n_components]
        if weighting == "none":
            return u, v
        if weighting == "left":
            return u * self.left_weights[:n_components], v
        if weighting == "right":
            return u, v * self.right_weights[:n_components]
        if weighting == "both":
            return u * self.left_weights[:n_components], v * self.right_weights[:n_components]
        raise ValueError(f"Unknown weighting '{weighting}'.")

    def transform(
        self,
        images: np.ndarray,
        n_components: int,
        weighting: str = "none",
    ) -> np.ndarray:
        images = np.asarray(images, dtype=np.float64)
        centered = images - self.mean_image
        u, v = self._weighted_factors(n_components, weighting)
        # P[n, i, j] = sum_{a,b} U[a, i] A[n, a, b] V[b, j]
        features = np.einsum("ai,nab,bj->nij", u, centered, v)
        return features.reshape(images.shape[0], -1)

    def reconstruct(self, images: np.ndarray, n_components: int) -> np.ndarray:
        """Reconstruct with the unweighted bilateral projection.

        Weights cancel in the paper's reconstruction formula, leaving
        A_hat = U U^T A V V^T.
        """
        centered = np.asarray(images, dtype=np.float64) - self.mean_image
        u = self.left[:, :n_components]
        v = self.right[:, :n_components]
        coeff = np.einsum("ai,nab,bj->nij", u, centered, v)
        restored = np.einsum("ai,nij,bj->nab", u, coeff, v)
        return restored + self.mean_image


def subspace_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    """Largest principal angle, in degrees, between two column spaces."""
    q1, _ = np.linalg.qr(first)
    q2, _ = np.linalg.qr(second)
    singular = np.linalg.svd(q1.T @ q2, compute_uv=False)
    singular = np.clip(singular, 0.0, 1.0)
    return float(np.degrees(np.arccos(singular).max()))
