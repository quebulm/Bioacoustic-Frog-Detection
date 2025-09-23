# scripts/infer.py
from __future__ import annotations

"""
Inference-Pipeline:
- optionales Preprocessing der RAWs in ein dediziertes infer-processed Verzeichnis
- YAMNet-Embeddings auf 2s-Snippets
- Klassifikation mit gespeichertem Keras-Classifier
- Ausgabe: CSV + Raven .Table.1.selections.txt pro Aufnahme

Config: configs/infer.yaml (kann include: base.yaml nutzen)
"""

import argparse
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm

from bioacoustic_frog_detection.data.io import (
    Paths,
    ensure_dir,
    load_wav,
    save_wav,
    snippet_filename,
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
    def __init__(self, d: dict):
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
    return cfg


# ------------------------------
# Preprocessing pro File
# ------------------------------
def _preprocess_one_file(
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
) -> None:
    import librosa

    basename = wav_path.stem
    out_subdir = out_dir / basename
    ensure_dir(out_subdir)

    # Laden (erzwinge sr_orig)
    data, sr = load_wav(wav_path, sr=sr_orig)
    if data is None or len(data) == 0:
        raise RuntimeError(f"Leeres/ungültiges File: {wav_path}")
    if sr != sr_orig:
        raise RuntimeError(f"Unerwartete SR nach load_wav: {sr} (erwartet {sr_orig})")

    # Bandpass + NR
    data = bandpass_filter(data, fs=sr, lowcut=lowcut, highcut=highcut, order=bp_order)
    data = noise_reduction(data, sr=sr)
    if len(data) == 0:
        raise RuntimeError(f"Kein Signal nach Filterung: {wav_path}")

    # Fensterung
    duration = len(data) / sr
    step = window_size * (1.0 - overlap)
    starts = np.arange(0, max(0.0, duration - window_size + step), step)

    for start in starts:
        start_sample = int(start * sr)
        end_sample = int((start + window_size) * sr)
        if end_sample > len(data):
            continue
        segment = data[start_sample:end_sample]

        # Resample → sr_target
        seg_16k = librosa.resample(segment, orig_sr=sr, target_sr=sr_target)

        # Speichern
        out_name = snippet_filename(basename, int(start * 1000))
        out_path = out_subdir / out_name
        save_wav(out_path, seg_16k, sr_target)


def maybe_preprocess_for_infer(raw_dir: Path, processed_dir: Path, pre_cfg: dict) -> None:
    """Wenn processed_dir leer ist, RAWs preprocessen."""
    ensure_dir(processed_dir)
    has_any = any(processed_dir.glob("*/*.wav"))
    if has_any:
        logging.info("Found existing snippets in %s – skip preprocessing.", processed_dir)
        return

    wavs = [p for p in Path(raw_dir).iterdir() if p.suffix.lower() == ".wav"]
    if not wavs:
        raise RuntimeError(f"Keine WAVs in {raw_dir}")

    logging.info("Preprocessing %d WAVs → %s", len(wavs), processed_dir)
    for wav in tqdm(wavs, desc="Preprocess (infer)"):
        try:
            _preprocess_one_file(
                wav, processed_dir,
                sr_orig=int(pre_cfg.get("sr_orig", 44100)),
                sr_target=int(pre_cfg.get("sr_target", 16000)),
                window_size=float(pre_cfg.get("window_size", 2.0)),
                overlap=float(pre_cfg.get("overlap", 0.5)),
                lowcut=float(pre_cfg.get("lowcut", 700.0)),
                highcut=float(pre_cfg.get("highcut", 4000.0)),
                bp_order=int(pre_cfg.get("bp_order", 5)),
            )
        except Exception as e:
            logging.error("Preprocess failed for %s: %s", wav, e)


# ------------------------------
# YAMNet Embeddings
# ------------------------------
def load_yamnet(hub_url: str):
    return hub.load(hub_url)

def yamnet_embed(yamnet, waveform_16k: np.ndarray) -> np.ndarray:
    # yamnet(audio) -> (scores, embeddings, spectrogram)
    _, embeddings, _ = yamnet(waveform_16k)
    return np.mean(embeddings.numpy(), axis=0)  # (1024,)


# ------------------------------
# Klassifikation & Merging
# ------------------------------
def load_classifier(model_path: Path) -> tf.keras.Model:
    from tensorflow import keras
    try:
        model = keras.models.load_model(model_path)
        logging.info("Loaded model: %s", model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Konnte Modell nicht laden ({model_path}): {e}")

def collect_snippet_paths(processed_dir: Path) -> Dict[str, List[Path]]:
    """{basename: [snippet_path, ...]}"""
    out: Dict[str, List[Path]] = {}
    for sub in sorted(processed_dir.iterdir()):
        if not sub.is_dir():
            continue
        basename = sub.name
        files = sorted([p for p in sub.iterdir() if p.suffix.lower() == ".wav" and p.name.startswith(basename + "_")])
        if files:
            out[basename] = files
    return out

def batch_predict_for_recording(
    basename: str,
    snippet_files: List[Path],
    yamnet,
    classifier: tf.keras.Model,
    batch_size: int,
    window_size: float,
) -> List[Tuple[float, float]]:
    """
    Returns list of (start_sec, prob) per snippet file of a single recording.
    """
    import librosa
    pairs: List[Tuple[float, float]] = []

    # Vorbereiten: Startzeiten parsen + Embeddings sammeln
    starts: List[float] = []
    embs: List[np.ndarray] = []
    for fp in snippet_files:
        m = re.search(rf"{re.escape(basename)}_(\d+)\.wav$", fp.name)
        if not m:
            continue
        start_ms = float(m.group(1))
        start_s = start_ms / 1000.0

        wav, sr = librosa.load(str(fp), sr=None, mono=True)
        if sr != 16000:
            logging.warning("%s: sr=%d (expected 16000)", fp, sr)
        if wav.size == 0:
            continue

        emb = yamnet_embed(yamnet, wav.astype(np.float32))
        starts.append(start_s)
        embs.append(emb)

    if not embs:
        return []

    X = np.stack(embs)  # (N, 1024)
    probs = classifier.predict(X, batch_size=batch_size, verbose=0).reshape(-1)

    for s, p in zip(starts, probs):
        pairs.append((s, float(p)))

    # sort by start
    pairs.sort(key=lambda t: t[0])
    return pairs

def merge_positives(
    start_prob_pairs: List[Tuple[float, float]],
    *,
    threshold: float,
    window_size: float,
    merge_gap: float,
    min_event_len: float,
) -> List[Tuple[float, float, float]]:
    """
    Aus binären Snippet-Treffern (start, prob) → gemergte Events (begin, end, max_prob).
    """
    # Filter positive Fenster
    positives = [(s, s + window_size, p) for (s, p) in start_prob_pairs if p >= threshold]
    if not positives:
        return []

    # Merge, wenn Lücke <= merge_gap
    merged: List[Tuple[float, float, float]] = []
    cur_b, cur_e, cur_max = positives[0]
    for b, e, p in positives[1:]:
        if b <= cur_e + merge_gap:
            cur_e = max(cur_e, e)
            cur_max = max(cur_max, p)
        else:
            merged.append((cur_b, cur_e, cur_max))
            cur_b, cur_e, cur_max = b, e, p
    merged.append((cur_b, cur_e, cur_max))

    # Mindestlänge filtern
    merged = [(b, e, m) for (b, e, m) in merged if (e - b) >= min_event_len]
    return merged


# ------------------------------
# Outputs: CSV + Raven .Table.1.selections.txt
# ------------------------------
def write_csv(out_dir: Path, basename: str, events: List[Tuple[float, float, float]]) -> Path:
    import pandas as pd
    ensure_dir(out_dir)
    rows = [
        {"start_s": round(b, 3), "end_s": round(e, 3), "score": round(m, 4)}
        for (b, e, m) in events
    ]
    df = pd.DataFrame(rows)
    out_path = out_dir / f"{basename}_inference.csv"
    df.to_csv(out_path, index=False)
    return out_path

def write_raven_selection(
    out_dir: Path,
    basename: str,
    events: List[Tuple[float, float, float]],
    *,
    low_hz: float,
    high_hz: float,
    species_label: str,
) -> Path:
    """
    Schreibt Raven-kompatibles .Table.1.selections.txt
    – Header so, dass euer Parser (Begin/End bei Spalten 3/4) funktioniert.
    """
    ensure_dir(out_dir)
    path = out_dir / f"{basename}.Table.1.selections.txt"
    with path.open("w") as f:
        f.write(
            "Selection\tView\tChannel\tBegin Time (s)\tEnd Time (s)\tLow Freq (Hz)\tHigh Freq (Hz)\tAnnotation\tScore\n"
        )
        sel_id = 1
        for (b, e, m) in events:
            f.write(
                f"{sel_id}\tSpectrogram 1\t1\t{b:.6f}\t{e:.6f}\t{low_hz:.1f}\t{high_hz:.1f}\t{species_label}\t{m:.4f}\n"
            )
            sel_id += 1
    return path


# ------------------------------
# CLI main
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Inference on raw WAVs → snippets → embeddings → predictions → Raven labels")
    parser.add_argument("--config", type=str, required=True, help="Pfad zu configs/infer.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # GPU speicherschonend
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    cfg = build_cfg(Path(args.config))
    paths = cfg.get("paths", {})

    raw_dir = Path(paths["infer_raw_dir"])
    processed_dir = Path(paths["infer_processed_dir"])
    out_dir = Path(paths.get("infer_outputs_dir", processed_dir.parent / "Inference_Results"))
    ensure_dir(out_dir)

    # 1) Preprocess, falls nötig
    maybe_preprocess_for_infer(raw_dir, processed_dir, cfg.get("preprocess", {}))

    # 2) YAMNet + Classifier laden
    yam = load_yamnet(cfg.get("yamnet", {}).get("hub_url", "https://tfhub.dev/google/yamnet/1"))
    model_path = Path(cfg.get("model", {}).get("path", paths.get("models_dir", ""))) if "model" in cfg else None
    if not model_path or (model_path.is_dir()):
        # Default: models/best_model.keras unter root_dir (wie train)
        root_models = Path(paths.get("models_dir", "models"))
        model_path = root_models / "best_model.keras"
    classifier = load_classifier(model_path)

    # 3) Klassifikation
    threshold = float(cfg.get("infer", {}).get("threshold", 0.5))
    merge_gap = float(cfg.get("infer", {}).get("merge_gap", 0.4))
    min_event_len = float(cfg.get("infer", {}).get("min_event_len", 0.5))
    batch_size = int(cfg.get("yamnet", {}).get("batch_size", 64))
    window_size = float(cfg.get("preprocess", {}).get("window_size", 2.0))
    low_hz = float(cfg.get("infer", {}).get("low_freq_hz", cfg.get("preprocess", {}).get("lowcut", 700.0)))
    high_hz = float(cfg.get("infer", {}).get("high_freq_hz", cfg.get("preprocess", {}).get("highcut", 4000.0)))
    species_label = str(cfg.get("infer", {}).get("species_label", "Frog"))

    per_recording = collect_snippet_paths(processed_dir)
    if not per_recording:
        raise RuntimeError(f"Keine Snippets in {processed_dir}")

    csv_dir = out_dir / "csv"
    raven_dir = out_dir / "raven"
    ensure_dir(csv_dir)
    ensure_dir(raven_dir)

    for basename, files in tqdm(per_recording.items(), desc="Inference (recordings)"):
        pairs = batch_predict_for_recording(
            basename, files, yam, classifier, batch_size=batch_size, window_size=window_size
        )
        events = merge_positives(
            pairs,
            threshold=threshold,
            window_size=window_size,
            merge_gap=merge_gap,
            min_event_len=min_event_len,
        )

        csv_path = write_csv(csv_dir, basename, events) if events else None
        sel_path = write_raven_selection(
            raven_dir, basename, events, low_hz=low_hz, high_hz=high_hz, species_label=species_label
        ) if events else None

        if events:
            logging.info("Wrote %d events for %s → %s ; %s", len(events), basename, csv_path, sel_path)
        else:
            logging.info("No events for %s (threshold=%.3f)", basename, threshold)

    # 4) Gesamt-Metadatei
    summary = {
        "threshold": threshold,
        "merge_gap": merge_gap,
        "min_event_len": min_event_len,
        "processed_dir": str(processed_dir),
        "outputs_dir": str(out_dir),
        "model": str(model_path),
    }
    with (out_dir / "inference_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    logging.info("Inference done. Summary → %s", out_dir / "inference_summary.json")


if __name__ == "__main__":
    main()