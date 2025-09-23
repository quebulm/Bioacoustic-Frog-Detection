# scripts/tune.py
from __future__ import annotations

"""
Optuna-Hyperparameter-Suche für den Frog-Call-Classifier.

Erwartet vorbereitete Embeddings (X.npy, y.npy).
Sucht u. a. Units, n_layers, Dropout, Focal-Loss-Parameter, Learning-Rate,
Batchgröße, pos_weight, Epochen, Patience.
Zielmetrik: F1 (Standard) oder PR-AUC (Average Precision) – per --metric wählbar.

Ausgabe: outputs/tuning.json mit best_value + best_params
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import optuna
import tensorflow as tf
from sklearn.metrics import f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

from bioacoustic_frog_detection.data.io import (
    Paths,
    load_embeddings,
    save_metrics,
)
from bioacoustic_frog_detection.models.classifier import build_classifier, ClassifierConfig


# ------------------------------------------------------------
# YAML-Loader (wie in train.py / make_embeddings.py)
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
    def __init__(self, d):
        self.paths = d["paths"]

def build_cfg(config_path: Path) -> Dict:
    raw = _load_yaml(config_path)
    if "include" in raw and raw["include"]:
        base = _load_yaml(Path(raw["include"]))
        cfg = _deep_update(base, raw)
    else:
        cfg = raw

    # ${root_dir} ersetzen
    root = cfg.get("root_dir", "")
    def subst(v):
        return v.replace("${root_dir}", root) if isinstance(v, str) and "${root_dir}" in v else v
    if "paths" in cfg:
        cfg["paths"] = {k: subst(v) for k, v in cfg["paths"].items()}
    return cfg


# ------------------------------------------------------------
# Objective-Factory
# ------------------------------------------------------------
def make_objective(X: np.ndarray, y: np.ndarray, cfg_train: Dict, metric: str):
    val_split = float(cfg_train.get("val_split", 0.2))

    def objective(trial: optuna.Trial) -> float:
        # Suchraum
        units = trial.suggest_int("units", 64, 512, log=True)
        n_layers = trial.suggest_int("n_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        focal_alpha = trial.suggest_float("focal_alpha", 0.5, 1.0)
        focal_gamma = trial.suggest_float("focal_gamma", 1.0, 4.0)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        pos_weight = trial.suggest_float("pos_weight", 1.0, 5.0)
        max_epochs = trial.suggest_int("epochs", 30, 100)
        patience = trial.suggest_int("patience", 5, 20)

        # Fester, reproduzierbarer Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42, stratify=y
        )

        # Modell (über ClassifierConfig)
        config = ClassifierConfig(
            input_shape=(X.shape[1],),
            hidden_layers=[units] * n_layers,
            dropout_rate=dropout,
            learning_rate=learning_rate,
            use_focal_loss=True,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
        )
        model = build_classifier(config)

        class_weights = {0: 1.0, 1: pos_weight}
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                min_delta=1e-4,
                restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=0,
            ),
        ]

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=0,
        )

        # Bewertung
        y_prob = model.predict(X_val, batch_size=batch_size).reshape(-1)
        if metric == "pr_auc":
            score = float(average_precision_score(y_val, y_prob))  # AP = PR-AUC
        else:  # f1
            y_pred = (y_prob > 0.5).astype(np.int32)
            score = float(f1_score(y_val, y_pred))

        # Ressourcen säubern (wichtig bei vielen Trials)
        tf.keras.backend.clear_session()
        return score

    return objective


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Optuna-Hyperparametertuning für Frog-Call-Classifier.")
    parser.add_argument("--config", type=str, required=True, help="Pfad zu configs/tune.yaml")
    parser.add_argument("--trials", type=int, default=5, help="Anzahl Optuna-Trials")
    parser.add_argument("--metric", type=str, choices=["f1", "pr_auc"], default="f1",
                        help="Zielmetrik für das Tuning")
    parser.add_argument("--study-name", type=str, default="frog_call_tuning")
    parser.add_argument("--export-params", type=str, default=None,
                        help="Optional: Pfad für YAML mit besten Parametern (z. B. configs/best_params.yaml)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # GPU speicherschonend
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    # Config + Daten
    cfg = build_cfg(Path(args.config))

    # Optuna-Einstellungen aus YAML (übersteuern CLI-Defaults)
    opt_cfg = cfg.get("optuna", {})
    n_trials = int(opt_cfg.get("trials", args.trials))
    metric = str(opt_cfg.get("metric", args.metric))
    study_name = str(opt_cfg.get("study_name", args.study_name))
    direction = str(opt_cfg.get("direction", "maximize"))
    logging.info("Optuna config → trials=%s, metric=%s, direction=%s, study_name=%s",
                 n_trials, metric, direction, study_name)

    paths = Paths.from_cfg(_CfgObj(cfg))
    X, y = load_embeddings(paths.embeddings_path, paths.y_path)
    logging.info("Loaded embeddings: X=%s, y=%s", X.shape, y.shape)

    # Studie
    study = optuna.create_study(direction=direction, study_name=study_name)
    objective = make_objective(X, y, cfg.get("train", {}), metric=metric)
    study.optimize(objective, n_trials=n_trials)

    logging.info("Best value (%s): %.6f", metric, study.best_value)
    logging.info("Best params: %s", json.dumps(study.best_trial.params, indent=2))

    # Ergebnisse speichern
    out = {
        "metric": metric,
        "best_value": study.best_value,
        "best_params": study.best_trial.params,
    }
    save_metrics(out, paths.outputs_dir, filename="tuning.json")

    # Optional: Beste Parameter als YAML exportieren (für direkten Include in train.yaml)
    if args.export_params:
        try:
            import yaml
        except Exception as e:
            logging.error("PyYAML nicht installiert – kann %s nicht schreiben: %s", args.export_params, e)
        else:
            best = study.best_trial.params
            export = {
                "model": {
                    "units": int(best.get("units")),
                    "n_layers": int(best.get("n_layers")),
                    "dropout": float(best.get("dropout")),
                    "focal_alpha": float(best.get("focal_alpha")),
                    "focal_gamma": float(best.get("focal_gamma")),
                },
                "train": {
                    "learning_rate": float(best.get("learning_rate")),
                    "batch_size": int(best.get("batch_size")),
                    "pos_weight": float(best.get("pos_weight")),
                    "epochs": int(best.get("epochs")),
                    "patience": int(best.get("patience")),
                },
            }
            export_path = Path(args.export_params)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            with export_path.open("w") as f:
                yaml.safe_dump(export, f, sort_keys=False)
            logging.info("Best params YAML geschrieben: %s", export_path)


if __name__ == "__main__":
    main()