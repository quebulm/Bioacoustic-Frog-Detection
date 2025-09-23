# scripts/make_embeddings.py
from __future__ import annotations

"""
Dünner CLI-Wrapper:
- lädt Config (features.yaml inkl. base.yaml)
- ruft die Library-Logik aus bioacoustic_frog_detection.data.features.extract_embeddings(...)
- speichert X/y via io.save_embeddings(...)

Die eigentliche Feature-Logik liegt in src/bioacoustic_frog_detection/data/features.py.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bioacoustic_frog_detection.data.io import (
    Paths,
    save_embeddings,
    load_labels_basename_keys,
)
from bioacoustic_frog_detection.data.features import extract_embeddings


# ------------------------------------------------------------
# Config-Laden (leichtgewichtig)
# ------------------------------------------------------------
@dataclass(frozen=True)
class FeaturesCfg:
    paths: Paths
    yamnet_hub_url: str
    segment_sec: float

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

def build_cfg(config_path: Path) -> FeaturesCfg:
    raw = _load_yaml(config_path)
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

    # Minimalobjekt, um Paths.from_cfg wiederzuverwenden
    class _Obj:
        def __init__(self, d): self.paths = d["paths"]
    paths = Paths.from_cfg(_Obj(cfg))

    seg = cfg.get("segment_sec")
    if seg is None:
        # optional aus base/audio übernehmen
        audio_cfg = cfg.get("audio", {})
        seg = float(audio_cfg.get("segment_sec", 2.0))
    else:
        seg = float(seg)

    hub_url = cfg.get("yamnet", {}).get("hub_url", "https://tfhub.dev/google/yamnet/1")

    return FeaturesCfg(paths=paths, yamnet_hub_url=hub_url, segment_sec=seg)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Erzeuge YAMNet-Embeddings (X) und Labels (y) aus Snippets + Raven-Labels.")
    parser.add_argument("--config", type=str, required=True, help="Pfad zu configs/features.yaml")
    parser.add_argument("--no-labels", action="store_true", help="Keine Raven-Labels laden; nur X (Embeddings) erzeugen.")
    parser.add_argument("--basenames", nargs="*", default=None, help="Optionale Liste von Basenames, die verarbeitet werden sollen.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = build_cfg(Path(args.config))

    # Labels optional laden
    if args.no_labels:
        labels_map = None
        logging.info("--no-labels aktiv: es werden keine Raven-Labels geladen. Es wird nur X erzeugt.")
    else:
        labels_map = load_labels_basename_keys(cfg.paths.labels_dir, cfg.paths.raw_dir)

    # Embeddings extrahieren (mit/ohne Labels)
    res = extract_embeddings(
        processed_dir=cfg.paths.processed_dir,
        yamnet_hub_url=cfg.yamnet_hub_url,
        segment_sec=cfg.segment_sec,
        basenames=args.basenames,
        labels_map=labels_map,
    )

    # Speichern
    if res.y is None:
        # Nur X sichern (Inference-Setup)
        np.save(cfg.paths.embeddings_path, res.X)
        logging.info("Fertig. Gespeichert: %s (nur X)", cfg.paths.embeddings_path)
    else:
        save_embeddings(res.X, res.y, cfg.paths.embeddings_path, cfg.paths.y_path)
        logging.info("Fertig. Gespeichert: %s (X), %s (y)", cfg.paths.embeddings_path, cfg.paths.y_path)


if __name__ == "__main__":
    main()