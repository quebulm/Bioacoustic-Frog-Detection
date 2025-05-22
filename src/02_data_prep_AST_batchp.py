import os
import re
import torchaudio
from torchaudio import functional as F
import torch
from transformers import ASTFeatureExtractor, ASTModel
from tqdm import tqdm
import numpy as np
import shutil  # for copying result files without renaming

# Device für PyTorch (MPS auf Apple Silicon, sonst CPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# ------------------------------
#  Performance‑related settings
# ------------------------------
BATCH_SIZE = 8   # smaller to avoid GPU memory overflow after augmentation
USE_FP16   = True   # set True for faster inference (may slightly affect accuracy)

# ------------------------------
#  Data Augmentation settings
# ------------------------------
AUGMENT_DATA   = True   # set False to disable on‑the‑fly augmentation for labelled data
AUG_PER_SNIPPET = 3     # number of augmented variants to generate per original snippet

# ------------------------------
#  Checkpointing & Progress
# ------------------------------
SAVE_EVERY      = 5000                     # save after this many embeddings
CHECKPOINT_DIR  = "../data/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
ast_model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(device)
if USE_FP16:
    ast_model = ast_model.to(dtype=torch.float16)

# ------------------------------
#  Labels aus TXT-Dateien laden (mehrere Ordner möglich)
# ------------------------------

def load_labels(label_folders, processed_folders):
    """
    Lädt Annotationen aus .Table.1.selections.txt und ordnet sie den geschnittenen 2-Sekunden-WAV-Dateien zu.
    Unterstützt mehrere Label-Ordner und mehrere Processed-Ordner.

    Args:
        label_folders (str or list of str): Pfad bzw. Pfade zu den Label-Textdateien.
        processed_folders (str or list of str): Pfad bzw. Pfade zu den geschnittenen 16kHz-Dateien.

    Returns:
        dict: { "basename": [(start_sec, end_sec)] } mit allen Intervallen
    """
    # Falls nur ein einzelner Pfad übergeben wurde, in Liste umwandeln
    if isinstance(label_folders, str):
        label_folders = [label_folders]
    if isinstance(processed_folders, str):
        processed_folders = [processed_folders]

    labels = {}

    for lf in label_folders:
        for txt_file in os.listdir(lf):
            if not txt_file.endswith(".Table.1.selections.txt"):
                continue

            label_path = os.path.join(lf, txt_file)
            basename = txt_file.replace(".Table.1.selections.txt", "")

            # Überprüfe, ob die geschnittenen WAV-Dateien in einem der Processed-Ordner existieren
            processed_found = False
            for pf in processed_folders:
                processed_audio_path = os.path.join(pf, basename)
                if os.path.exists(processed_audio_path):
                    processed_found = True
                    break

            if not processed_found:
                tqdm.write(f"[WARN] Keine geschnittenen WAV-Dateien für {basename} in einem der Pfade, skip...")
                continue

            try:
                with open(label_path, "r") as f:
                    lines = f.readlines()

                time_intervals = []
                for line in lines[1:]:  # Header überspringen
                    parts = line.strip().split("\t")
                    if len(parts) < 5:
                        continue

                    start_time = float(parts[3])  # "Begin Time (s)"
                    end_time = float(parts[4])    # "End Time (s)"
                    time_intervals.append((start_time, end_time))

                if time_intervals:
                    labels[basename] = time_intervals
                    tqdm.write(f"[INFO] Labels für {basename} geladen: {time_intervals}")

            except Exception as e:
                tqdm.write(f"[ERROR] Fehler beim Laden von {label_path}: {e}")

    return labels


# ------------------------------
#  Label für ein 2-Sekunden-Fenster bestimmen
# ------------------------------

def get_window_label(intervals, start_sec, end_sec):
    """
    Prüft, ob ein 2-Sekunden-Fenster mit einem Froschruf überlappt.

    Args:
        intervals (list of tuples): [(start1, end1), (start2, end2), ...]
        start_sec (float): Startzeit des Fensters
        end_sec (float): Endzeit des Fensters

    Returns:
        int: 1, wenn ein Ruf im Fenster enthalten ist, sonst 0
    """
    for (label_start, label_end) in intervals:
        if (start_sec < label_end) and (end_sec > label_start):
            return 1  # Überlappung gefunden
    return 0  # Kein Treffer


# ------------------------------
#  AST Embedding für Audio berechnen
# ------------------------------

def get_ast_embedding(audio_waveform, sr=16000):
    """
    Berechnet das AST-Embedding für ein 2-Sekunden-Snippet.
    Args:
        audio_waveform (numpy array): Audio-Wellenform (Mono, 16kHz)
        sr (int): Sample-Rate (default: 16000)
    Returns:
        numpy array: AST-Embedding (Shape: (768,))
    """
    inputs = feature_extractor(audio_waveform, sampling_rate=sr, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = ast_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze().cpu().numpy()

def get_ast_embeddings_batch(waveforms, sr=16000):
    """
    Batch‑variant of AST embedding.
    Accepts a list of 1‑D numpy arrays (mono, 16 kHz) and returns
    a numpy array of shape (B, 768) with the CLS embeddings.
    """
    inputs = feature_extractor(
        waveforms,
        sampling_rate=sr,
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if USE_FP16:
        inputs["input_values"] = inputs["input_values"].half()
    with torch.no_grad():
        outputs = ast_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# ------------------------------
#  Simple Audio Augmentations
# ------------------------------

def augment_waveform(waveform, sr=16000):
    """
    Generates an augmented variant of the given mono waveform.
    Two augmentations are implemented:
        1. Additive Gaussian noise with a random SNR between 20‑30 dB.
        2. Circular time shift up to ±0.1 s.
    The augmented waveform retains the original length and sampling rate.
    """
    aug = waveform.copy()
    if np.random.rand() < 0.5:
        # Additive noise
        noise = np.random.randn(len(aug))
        sig_power = np.mean(aug ** 2)
        snr_db = np.random.uniform(20, 30)
        noise_power = sig_power / (10 ** (snr_db / 10))
        aug += np.sqrt(noise_power) * noise
    else:
        # Time shift
        max_shift = int(0.1 * sr)
        shift = np.random.randint(-max_shift, max_shift)
        aug = np.roll(aug, shift)
    # Clip to valid range
    aug = np.clip(aug, -1.0, 1.0)
    return aug.astype(waveform.dtype)

def count_total_snippets(processed_folders):
    """
    Counts all .wav snippets inside the given processed folders.
    Used to initialise a single tqdm progress bar for the whole run.
    """
    if isinstance(processed_folders, str):
        processed_folders = [processed_folders]

    total = 0
    for pf in processed_folders:
        for basename in os.listdir(pf):
            subdir = os.path.join(pf, basename)
            if not os.path.isdir(subdir):
                continue
            total += len([
                f for f in os.listdir(subdir)
                if f.endswith(".wav") and f.startswith(basename + "_")
            ])
    return total

# ------------------------------
#  Snippets + Labels extrahieren und Embeddings berechnen (mehrere Processed-Ordner möglich)
# ------------------------------

def collect_snippet_embeddings(processed_folders, labels_dict=None, return_labels=True):
    """
    Durchläuft alle 2-Sekunden-Snippets aus mehreren Processed-Ordnern und bestimmt ihr Label anhand der Labels.

    Args:
        processed_folders (str or list of str): Pfad bzw. Pfade zu den 16kHz Snippets.
        labels_dict (dict): Mapping { "basename": [(start_sec, end_sec)] }

    Returns:
        X (numpy array): Embeddings (N, 768)
        y (numpy array): Labels (N,)
    """
    if isinstance(processed_folders, str):
        processed_folders = [processed_folders]

    total_snippets = count_total_snippets(processed_folders)
    # If we create on‑the‑fly augmented copies, account for them in the progress bar
    if return_labels and AUGMENT_DATA:
        total_snippets *= (1 + AUG_PER_SNIPPET)
    pbar = tqdm(total=total_snippets, desc="Processing snippets", unit="snippet")

    X, y = [], []

    for pf_idx, pf in enumerate(processed_folders):
        tqdm.write(f"[INFO] Ordner {pf_idx + 1}/{len(processed_folders)}: {pf}")
        for basename in os.listdir(pf):
            subdir = os.path.join(pf, basename)
            if not os.path.isdir(subdir):
                continue

            if return_labels:
                if basename not in labels_dict:
                    tqdm.write(f"[WARN] Keine Labels für {basename}, Pfad: {subdir}, skip...")
                    continue
                intervals = labels_dict[basename]
            else:
                intervals = []

            # Alle Snippet-Dateien durchsuchen
            snippet_files = sorted([
                f for f in os.listdir(subdir)
                if f.endswith(".wav") and f.startswith(basename + "_")
            ])
            # snippet_files = tqdm(snippet_files, desc=basename, leave=False)

            if not snippet_files:
                tqdm.write(f"[WARN] Keine Snippet-Dateien gefunden für {basename}, Pfad: {subdir}, skip...")
                continue

            batch_wavs, batch_labels = [], []

            for idx, sfile in enumerate(snippet_files):
                match = re.search(rf"{basename}_(\d+)\.wav", sfile)
                if not match:
                    tqdm.write(f"[WARN] Konnte Startzeit nicht extrahieren: {sfile}, in {subdir}")
                    continue

                start_ms = float(match.group(1))
                start_sec = start_ms / 1000.0
                end_sec   = start_sec + 2.0

                if return_labels:
                    snippet_label = get_window_label(intervals, start_sec, end_sec)

                snippet_path = os.path.join(subdir, sfile)
                try:
                    wav_tensor, sr_ = torchaudio.load(snippet_path)
                    wav_tensor = wav_tensor.mean(dim=0)            # mono
                    if sr_ != 16000:
                        tqdm.write(f"[WARN] Sample Rate ist {sr_} statt 16000 – Datei übersprungen: {snippet_path}")
                        continue
                    wav_data = wav_tensor.numpy()
                except Exception as e:
                    tqdm.write(f"[ERROR] Fehler beim Laden {snippet_path}: {e}")
                    continue

                batch_wavs.append(wav_data)
                if return_labels:
                    batch_labels.append(snippet_label)

                # Optional data augmentation for labelled data
                if return_labels and AUGMENT_DATA:
                    for _ in range(AUG_PER_SNIPPET):
                        aug_wav = augment_waveform(wav_data, sr=16000)
                        batch_wavs.append(aug_wav)
                        batch_labels.append(snippet_label)

                # If batch is full (or exceeds) or last snippet, run inference
                if len(batch_wavs) >= BATCH_SIZE or idx == len(snippet_files) - 1:
                    embs = get_ast_embeddings_batch(batch_wavs, sr=16000)
                    X.extend(embs)
                    pbar.update(len(embs))
                    if return_labels:
                        y.extend(batch_labels)
                    batch_wavs, batch_labels = [], []

                    # checkpoint every SAVE_EVERY embeddings
                    if len(X) % SAVE_EVERY == 0:
                        np.save(os.path.join(CHECKPOINT_DIR, "X_ckpt.npy"), np.array(X))
                        if return_labels:
                            np.save(os.path.join(CHECKPOINT_DIR, "y_ckpt.npy"), np.array(y))

    pbar.close()
    X = np.array(X)
    if return_labels:
        y = np.array(y)
        return X, y
    else:
        return X


# ------------------------------
#  Haupt-Funktion
# ------------------------------

if __name__ == "__main__":
    label_folders = ["../data/Labels/v0"]
    processed_folders = ["../data/Processed_16k"]

    ONLY_X = False  # True = nur Embeddings ohne Labels erzeugen

    if ONLY_X:
        X = collect_snippet_embeddings(processed_folders, return_labels=False)
        print(f"Embeddings gesammelt (nur X): {X.shape}")
        np.save("../data/X_embeddings_infer_october.npy", X)
    else:
        labels_dict = load_labels(label_folders, processed_folders)
        X, y = collect_snippet_embeddings(processed_folders, labels_dict, return_labels=True)
        print(f"Embeddings gesammelt: {X.shape}, Labels: {y.shape}")
        np.save("../data/X_embeddings_v0_16k_aug.npy", X)
        np.save("../data/y_labels_v0_aug.npy", y)

    # copy the results into the checkpoint directory *without* renaming,
    # so that descriptive filenames are preserved
    import shutil
    shutil.copy("../data/X_embeddings_v0_16k_aug.npy",
                os.path.join(CHECKPOINT_DIR, "X_embeddings_v0_16k_aug.npy"))
    if not ONLY_X:
        shutil.copy("../data/X_embeddings_v0_16k_aug.npy",
                    os.path.join(CHECKPOINT_DIR, "X_embeddings_v0_16k_aug.npy"))
        shutil.copy("../data/y_labels_v0_aug.npy",
                    os.path.join(CHECKPOINT_DIR, "y_labels_v0_aug.npy"))

    print("Daten gespeichert!")