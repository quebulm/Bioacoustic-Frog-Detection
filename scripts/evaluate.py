

# scripts/evaluate.py
from __future__ import annotations

"""
Dünner CLI-Wrapper für die Evaluation auf Embeddings (X.npy, y.npy).
Liest YAML-Config (inkl. include: base.yaml), ruft
bioacoustic_frog_detection.eval.on_embeddings.evaluate(cfg).
"""

import argparse
import json
import logging
from pathlib import Path

import tensorflow as tf

from bioacoustic_frog_detection.eval.on_embeddings import evaluate as eval_on_embeddings


# -------- YAML loader + include(base.yaml) --------

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


def build_cfg(config_path: Path) -> dict:
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


# -------- Main --------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Frog-Call Classifier on embeddings (X,y)")
    parser.add_argument("--config", type=str, required=True, help="Pfad zu configs/eval.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # GPU speicherschonend
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    cfg = build_cfg(Path(args.config))
    metrics = eval_on_embeddings(cfg)
    logging.info("Evaluation done: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()