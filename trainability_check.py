"""Gradient variance of TNO Quantum's VariationalClassifier as the number of qubits grows.

Question: does the default "modulo_model" of tno.quantum.ml.classifiers.vc show a
barren plateau, and does small-angle initialisation help?

The script uses TNO's own model classes, so the circuit is exactly the package's:
angle encoding (RX) interleaved with StronglyEntanglingLayers, one qubit per feature.

It compares:
  * two tasks: binary (the package measures <Z> on qubit 0, a local cost) and
    three classes (the package sums bit-string probabilities modulo 3, a global cost).
    Theory for shallow circuits predicts the local cost decays more slowly; see the
    README for what the run actually showed;
  * two initialisations: the package default, uniform in [0, 2*pi),
    and small-angle, Normal(0, eps^2).

For each (task, init, n_qubits) it draws fresh random weights and a random input
for every sample, computes the gradient of the model output with respect to all
weights with PyTorch autograd, and reports the variance of each partial derivative
across samples, averaged over parameters.

Install:  pip install tno.quantum.ml.classifiers.vc matplotlib pandas
Run:      python trainability_check.py            (about 1 minute on a laptop)
Quick:    python trainability_check.py --quick    (3 to 8 qubits, 40 samples)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from tno.quantum.ml.classifiers.vc import ModelConfig
from tno.quantum.utils import BackendConfig

MODEL_OPTIONS = {"n_layers": 2, "n_trainable_sublayers": 2, "scaling": 0.5}  # package default


def build_model(n_classes: int):
    backend = BackendConfig(name="default.qubit", options={})
    config = ModelConfig(name="modulo_model", options=MODEL_OPTIONS)
    return config.get_instance(backend=backend, n_classes=n_classes)


def gradient_variance(n_qubits: int, n_classes: int, init: str, n_samples: int,
                      eps: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    model = build_model(n_classes)

    # Random inputs in [0, 1]^n, mapped by the package's own preprocessing to (-pi, pi].
    X = rng.uniform(0.0, 1.0, size=(n_samples, n_qubits))
    angles = model.preprocess(X, (np.zeros(n_qubits), np.ones(n_qubits)))  # sets model.n_qubits
    qfunc = model.get_qfunc()

    shape = (MODEL_OPTIONS["n_layers"] + 1, MODEL_OPTIONS["n_trainable_sublayers"], n_qubits, 3)
    grads = []
    for i in range(n_samples):
        if init == "uniform":  # what the package does when random_init=True
            w0 = 2 * np.pi * rng.random(shape)
        else:                  # small-angle initialisation
            w0 = eps * rng.standard_normal(shape)
        w = torch.tensor(w0, dtype=torch.float64, requires_grad=True)
        x = torch.tensor(angles[i], dtype=torch.float64)

        out = qfunc(w, x)
        target = out if out.dim() == 0 else out[0]  # binary: <Z_0>; multiclass: P(class 0)
        target.backward()
        grads.append(w.grad.detach().numpy().ravel().copy())

    grads = np.array(grads)                      # (n_samples, n_params)
    var_per_param = grads.var(axis=0, ddof=1)
    return {
        "n_qubits": n_qubits,
        "task": "binary" if n_classes == 2 else f"{n_classes} classes",
        "init": init,
        "mean_grad_variance": float(var_per_param.mean()),
        "median_grad_variance": float(np.median(var_per_param)),
        "n_params": grads.shape[1],
        "n_samples": n_samples,
    }


def decay_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Fit log(variance) = a + b * n. exp(b) is the factor lost per added qubit."""
    rows = []
    for (task, init), g in df.groupby(["task", "init"]):
        b, _ = np.polyfit(g["n_qubits"], np.log(g["mean_grad_variance"]), 1)
        rows.append({"task": task, "init": init, "slope_per_qubit": b,
                     "factor_per_added_qubit": float(np.exp(b))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-qubits", type=int, default=10)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    max_q = 8 if args.quick else args.max_qubits
    n_samples = 40 if args.quick else args.samples
    qubit_range = range(3, max_q + 1)  # 3 classes need at least 2 qubits; start at 3 for both

    results = []
    for n in qubit_range:
        for n_classes in (2, 3):
            for init in ("uniform", "small"):
                t0 = time.time()
                r = gradient_variance(n, n_classes, init, n_samples, args.eps, args.seed + n)
                results.append(r)
                print(f"n={n:2d}  {r['task']:9s}  init={init:7s}  "
                      f"mean Var[dL/dtheta]={r['mean_grad_variance']:.3e}  "
                      f"({time.time() - t0:.1f}s)")

    df = pd.DataFrame(results)
    df.to_csv("gradient_variance.csv", index=False)
    rates = decay_rate(df)
    rates.to_csv("decay_rates.csv", index=False)
    print("\nFitted decay (factor < 1 means variance shrinks with each added qubit):")
    print(rates.to_string(index=False))

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        for (task, init), g in df.groupby(["task", "init"]):
            ax.semilogy(g["n_qubits"], g["mean_grad_variance"], marker="o",
                        label=f"{task}, {'uniform [0, 2pi)' if init == 'uniform' else f'small angle (eps={args.eps})'}")
        ax.set_xlabel("Number of qubits (= number of features)")
        ax.set_ylabel("Mean variance of partial derivatives")
        ax.set_title("tno.quantum VariationalClassifier, default modulo_model")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig("gradient_variance.png", dpi=200)
        print("\nSaved gradient_variance.png, gradient_variance.csv, decay_rates.csv")
    except ImportError:
        print("matplotlib not installed: CSV files saved, no plot.")


if __name__ == "__main__":
    main()
