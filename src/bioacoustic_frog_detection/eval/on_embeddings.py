# src/bioacoustic_frog_detection/eval/on_embeddings.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import matplotlib.pyplot as plt

from bioacoustic_frog_detection.data.io import (
    Paths,
    load_embeddings,
    best_model_path,
    save_metrics,
)


@dataclass(frozen=True)
class EvalParams:
    batch_size: int
    threshold: float


def _get_params(cfg: Dict) -> EvalParams:
    # eval.* hat Vorrang, sonst train.*
    e = cfg.get("eval", {}) or {}
    t = cfg.get("train", {}) or {}
    batch_size = int(e.get("batch_size", t.get("batch_size", 32)))
    threshold = float(e.get("threshold", t.get("threshold", 0.5)))
    return EvalParams(batch_size=batch_size, threshold=threshold)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No Call (0)", "Call (1)"], yticklabels=["No Call (0)", "Call (1)"], ylabel="True", xlabel="Predicted", title="Confusion Matrix")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    _ensure_parent(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="best")
    _ensure_parent(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return float(pr_auc)


def evaluate(cfg: Dict) -> Dict:
    """
    Evaluierung auf vorbereiteten Embeddings (X.npy, y.npy).

    Erwartet in `cfg` mindestens `paths` (embeddings_path, y_path, models_dir, outputs_dir).
    Nutzt optionale `eval.batch_size` und `eval.threshold` (fällt sonst auf train.* zurück).

    Returns
    -------
    dict
        Zusammenfassung mit precision/recall/f1/accuracy/threshold/pr_auc.
    """
    # GPU speicherschonend
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    params = _get_params(cfg)

    # Pfade
    paths = Paths.from_cfg(type("CfgObj", (), {"paths": cfg["paths"]}))

    # Daten laden
    X, y = load_embeddings(paths.embeddings_path, paths.y_path)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError(f"Unerwartete Shapes: X={X.shape}, y={y.shape}")

    # Modell laden
    model_path = best_model_path(paths.models_dir)
    model = tf.keras.models.load_model(model_path, compile=False)

    # Vorhersagen
    y_prob = model.predict(X, batch_size=params.batch_size).reshape(-1)
    y_pred = (y_prob > params.threshold).astype(np.int32)

    # Kennzahlen
    report = classification_report(y, y_pred, digits=4, output_dict=True)
    metrics: Dict[str, float] = {
        "precision": float(report["1"]["precision"]),
        "recall": float(report["1"]["recall"]),
        "f1": float(report["1"]["f1-score"]),
        "accuracy": float(report["accuracy"]),
        "threshold": float(params.threshold),
    }

    # PR-AUC & Plots
    pr_plot = paths.outputs_dir / "pr_curve.png"
    cm_plot = paths.outputs_dir / "confusion_matrix.png"
    metrics["pr_auc"] = _plot_pr_curve(y, y_prob, out_path=pr_plot)
    _plot_confusion(y, y_pred, out_path=cm_plot)

    # Persistieren
    save_metrics(metrics, paths.outputs_dir, filename="eval_metrics.json")
    return metrics
