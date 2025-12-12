from __future__ import annotations

"""Experimental quantum-ML baseline.

This is *optional* and only runs if you install requirements-quantum.txt.
It is designed as a small-sample demonstration, not a production trainer.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class QuantumResult:
    info: str


def try_quantum_baseline(X: np.ndarray, y: np.ndarray, seed: int = 0) -> QuantumResult:
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
    except Exception as e:  # pragma: no cover
        return QuantumResult(info=f"Quantum deps not installed: {e}")

    # Minimal 2-class demo (H vs not-H) to keep circuit tiny.
    rng = np.random.default_rng(seed)

    # Reduce features
    n_samples, n_features = X.shape
    n_wires = min(6, max(2, n_features))

    X2 = X[:, :n_wires]
    X2 = (X2 - np.nanmean(X2, axis=0)) / (np.nanstd(X2, axis=0) + 1e-9)
    X2 = np.clip(X2, -3, 3)

    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev)
    def circuit(x, weights):
        qml.AngleEmbedding(x, wires=range(n_wires))
        qml.BasicEntanglerLayers(weights, wires=range(n_wires))
        return qml.expval(qml.PauliZ(0))

    weights = pnp.array(rng.normal(scale=0.1, size=(2, n_wires)), requires_grad=True)

    def sigmoid(z):
        return 1 / (1 + pnp.exp(-z))

    def loss_fn(w):
        preds = [sigmoid(circuit(x, w)) for x in X2]
        preds = pnp.stack(preds)
        # y expected in {0,1}
        eps = 1e-7
        return -pnp.mean(y * pnp.log(preds + eps) + (1 - y) * pnp.log(1 - preds + eps))

    opt = qml.AdamOptimizer(stepsize=0.05)
    for _ in range(40):
        weights = opt.step(loss_fn, weights)

    final_loss = float(loss_fn(weights))
    return QuantumResult(info=f"Trained tiny QML demo (wires={n_wires}), final_loss={final_loss:.4f}")
