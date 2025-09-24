"""
Training-Skript für den Frog-Call-Classifier.

Erwartet vorbereitete Embeddings (X.npy, y.npy) aus scripts/make_embeddings.py.
Liest Hyperparameter aus configs/train.yaml (inkl. include: base.yaml).
"""

from __future__ import annotations
import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

from bioacoustic_frog_detection.data.io import (
    Paths,
    load_embeddings,
    ensure_dir,
    best_model_path,
    save_metrics,
)
from bioacoustic_frog_detection.models.classifier import (
    build_classifier,
    ClassifierConfig,
)


# ------------------------------------------------------------
# Config-Lader (leichtgewichtig, wie in make_embeddings.py)
# ------------------------------------------------------------
def _load_yaml(path: Path) -> dict:
    import yaml
    with path.open("r") as f:
        return yaml.safe_load(f)

def _deep_update(a: dict, b: dict) -> dict:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = v
    return a

class _CfgObj:
    def __init__(self, d): self.paths = d["paths"]

@dataclass(frozen=True)
class TrainCfg:
    paths: Paths
    input_dim: int
    units: int
    n_layers: int
    dropout: float
    focal_alpha: float
    focal_gamma: float
    learning_rate: float
    batch_size: int
    epochs: int
    patience: int
    min_delta: float
    val_split: float
    pos_weight: float
    threshold: float

def build_cfg(config_path: Path, best_params_path: Path | None = None) -> TrainCfg:
    raw = _load_yaml(config_path)
    # include
    if "include" in raw and raw["include"]:
        base = _load_yaml(Path(raw["include"]))
        cfg = _deep_update(base, raw)
    else:
        cfg = raw

    root = cfg.get("root_dir", "")
    def subst(v):
        return v.replace("${root_dir}", root) if isinstance(v, str) and "${root_dir}" in v else v
    if "paths" in cfg:
        cfg["paths"] = {k: subst(v) for k, v in cfg["paths"].items()}

    # Optional Overlay: beste Hyperparameter aus YAML (überschreibt cfg.model/train)
    if best_params_path is not None:
        try:
            import yaml
            with Path(best_params_path).open("r") as f:
                best = yaml.safe_load(f) or {}
            _deep_update(cfg, best)
        except Exception as e:
            logging.warning("Konnte best params nicht laden (%s): %s", best_params_path, e)

    paths = Paths.from_cfg(_CfgObj(cfg))

    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})

    return TrainCfg(
        paths=paths,
        input_dim=int(model_cfg.get("input_dim", 1024)),
        units=int(model_cfg.get("units", 256)),
        n_layers=int(model_cfg.get("n_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        focal_alpha=float(model_cfg.get("focal_alpha", 0.75)),
        focal_gamma=float(model_cfg.get("focal_gamma", 2.0)),
        learning_rate=float(train_cfg.get("learning_rate", 5e-4)),
        batch_size=int(train_cfg.get("batch_size", 32)),
        epochs=int(train_cfg.get("epochs", 50)),
        patience=int(train_cfg.get("patience", 10)),
        min_delta=float(train_cfg.get("min_delta", 1e-4)),
        val_split=float(train_cfg.get("val_split", 0.2)),
        pos_weight=float(train_cfg.get("pos_weight", 1.0)),
        threshold=float(train_cfg.get("threshold", 0.5)),
    )


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Frog-Call Classifier auf YAMNet-Embeddings.")
    parser.add_argument("--config", type=str, required=True, help="Pfad zu configs/train.yaml")
    parser.add_argument(
        "--best-params", type=str, default=None,
        help="Optional: YAML mit besten Parametern (Export aus scripts/tune.py)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = build_cfg(Path(args.config), Path(args.best_params) if args.best_params else None)
    if args.best_params:
        logging.info("Best-Params-Overlay aktiv: %s", args.best_params)

    # Daten laden
    X, y = load_embeddings(cfg.paths.embeddings_path, cfg.paths.y_path)
    if X.ndim != 2:
        raise ValueError(f"X erwartet 2D (N,D), erhalten: {X.shape}")
    logging.info("Loaded embeddings: X=%s, y=%s", X.shape, y.shape)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.val_split, random_state=42, stratify=y
    )

    # Modell-Konfiguration aus YAML
    clf_cfg = ClassifierConfig(
        input_shape=(X.shape[1],),
        hidden_layers=[cfg.units] * int(cfg.n_layers),
        dropout_rate=float(cfg.dropout),
        learning_rate=float(cfg.learning_rate),
        use_focal_loss=True,
        focal_alpha=float(cfg.focal_alpha),
        focal_gamma=float(cfg.focal_gamma),
    )
    model = build_classifier(clf_cfg)
    model.summary(print_fn=lambda s: logging.info(s))

    # Class weights (für Imbalance)
    class_weights = {0: 1.0, 1: cfg.pos_weight}

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.patience,
            min_delta=cfg.min_delta,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(3, cfg.patience // 2),
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Speichern
    model_path = best_model_path(cfg.paths.models_dir)
    ensure_dir(model_path.parent)
    model.save(model_path)
    logging.info("Best model saved to %s", model_path)

    # Kennzahlen
    y_val_prob = model.predict(X_val, batch_size=cfg.batch_size).reshape(-1)
    y_val_pred = (y_val_prob > cfg.threshold).astype(np.int32)
    precision = float(tf.keras.metrics.Precision()(y_val, y_val_pred).numpy())
    recall = float(tf.keras.metrics.Recall()(y_val, y_val_pred).numpy())
    pr_auc = float(tf.keras.metrics.AUC(curve="PR")(y_val, y_val_prob).numpy())

    metrics = {
        "val_precision": precision,
        "val_recall": recall,
        "val_pr_auc": pr_auc,
        "threshold": cfg.threshold,
    }
    save_metrics(metrics, cfg.paths.outputs_dir)
    logging.info("Validation metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    # GPU speicherschonend
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    main()