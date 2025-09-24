from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

from bioacoustic_frog_detection.data.io import (
    ensure_dir,
    load_wav,
    save_wav,
    snippet_filename,
    Paths,
    iter_raw_wavs,
)
from bioacoustic_frog_detection.data.filters import bandpass_filter, noise_reduction


# ------------------------------
# YAML loader + include(base) + ${root_dir} substitution
# ------------------------------
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

    # Defaults für preprocess
    pre = cfg.setdefault("preprocess", {})
    pre.setdefault("sr_orig", 44100)
    pre.setdefault("sr_target", 16000)
    pre.setdefault("window_size", 2.0)
    pre.setdefault("overlap", 0.5)
    pre.setdefault("lowcut", 700.0)
    pre.setdefault("highcut", 4000.0)
    pre.setdefault("bp_order", 5)
    pre.setdefault("n_jobs", 4)

    return cfg


# ------------------------------
# Snippet-Erzeugung für eine Datei
# ------------------------------
def process_file(
    wav_path: Path,
    out_dir: Path,
    *,
    sr_orig: int,
    sr_target: int,
    window_size: float,
    overlap: float,
    lowcut: float,
    highcut: float,
    bp_order: int,
) -> dict:
    """
    Verarbeitet eine WAV-Datei deterministisch:
      - lädt Signal (erzwingt sr_orig)
      - Bandpass (lowcut/highcut, order) + Noise Reduction
      - 2s-Snippets mit Overlap
      - Resample → sr_target
      - Speichert Snippets
    """
    basename = wav_path.stem
    out_subdir = out_dir / basename
    ensure_dir(out_subdir)

    # 1) Laden (erzwinge sr_orig)
    data, sr = load_wav(wav_path, sr=sr_orig)
    if data is None or len(data) == 0:
        raise RuntimeError(f"Leeres/ungültiges File: {wav_path}")
    if sr != sr_orig:
        # load_wav mit librosa(sr=sr_orig) sollte bereits resamplen; assert nur zur Sicherheit
        raise RuntimeError(f"Unerwartete SR nach load_wav: {sr} (erwartet {sr_orig})")

    # 2) Bandpass + Noise Reduction
    data = bandpass_filter(data, fs=sr, lowcut=lowcut, highcut=highcut, order=bp_order)
    data = noise_reduction(data, sr=sr)
    if len(data) == 0:
        raise RuntimeError(f"Kein Signal nach Filterung: {wav_path}")

    # 3) Fensterung
    duration = len(data) / sr
    step = window_size * (1.0 - overlap)
    starts = np.arange(0, duration - window_size + step, step)

    manifest_entries = []
    for start in starts:
        start_sample = int(start * sr)
        end_sample = int((start + window_size) * sr)
        if end_sample > len(data):
            continue

        segment = data[start_sample:end_sample]

        # 4) Resample → sr_target
        seg_16k = librosa.resample(segment, orig_sr=sr, target_sr=sr_target)

        # 5) Speichern
        out_name = snippet_filename(basename, int(start * 1000))
        out_path = out_subdir / out_name
        save_wav(out_path, seg_16k, sr_target)

        manifest_entries.append({"start": float(start), "path": str(out_path)})

    return {basename: {"snippets": manifest_entries}}


# ------------------------------
# Verzeichnis verarbeiten
# ------------------------------
def process_directory(in_dir: Path, out_dir: Path, *, cfg: dict) -> dict:
    pre = cfg["preprocess"]
    use_labels = bool(pre.get("use_labels", False))
    labels_dir = Path(cfg["paths"].get("labels_dir", "")) if use_labels else None
    wavs = list(iter_raw_wavs(Path(in_dir), use_labels=use_labels, label_dir=labels_dir))

    if not wavs:
        raise RuntimeError(f"Keine WAVs in {in_dir}")

    manifest: dict = {}
    with ProcessPoolExecutor(max_workers=int(pre["n_jobs"])) as ex:
        futures = {
            ex.submit(
                process_file,
                wav,
                out_dir,
                sr_orig=int(pre["sr_orig"]),
                sr_target=int(pre["sr_target"]),
                window_size=float(pre["window_size"]),
                overlap=float(pre["overlap"]),
                lowcut=float(pre["lowcut"]),
                highcut=float(pre["highcut"]),
                bp_order=int(pre["bp_order"]),
            ): wav
            for wav in wavs
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            try:
                result = fut.result()
                manifest.update(result)
            except Exception as e:
                print(f"[ERROR] {futures[fut]}: {e}")

    # Manifest speichern
    manifest_path = out_dir / "manifest.json"
    ensure_dir(manifest_path.parent)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Manifest gespeichert unter {manifest_path}")
    return manifest


# ------------------------------
# CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Preprocessing: Bandpass, Denoise, Windowing, Resample→16k.")
    parser.add_argument("--config", type=str, required=True, help="Pfad zu configs/preprocess.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = build_cfg(Path(args.config))

    # Pfade aus base.yaml
    paths = Paths.from_cfg(_CfgObj(cfg))
    in_dir = Path(paths.raw_dir)
    out_dir = Path(paths.processed_dir)
    ensure_dir(out_dir)

    # Start
    process_directory(in_dir, out_dir, cfg=cfg)


if __name__ == "__main__":
    main()