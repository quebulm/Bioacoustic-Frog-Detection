import os
import librosa
import numpy as np
import torch
import torchaudio
import soundfile as sf

from concurrent.futures import ProcessPoolExecutor, as_completed

# external noise‑reduction library
try:
    import noisereduce as nr
    HAS_NR = True
except ImportError:
    HAS_NR = False
    print("[WARN] noisereduce package not found – skipping noise reduction.")
from tqdm import tqdm
from scipy.signal import butter, lfilter

# ------------------------------
# SETTINGS
# ------------------------------

LOWCUT = 700      # Bandpass untere Grenze (Hz) – abgesenkt, um Fundamentals ≥ 0.3 kHz zu erhalten.
HIGHCUT = 4000    # Bandpass obere Grenze (Hz)
SR_ORIG = 44100   # Original-Samplingrate (Einlesen)
SR_TARGET = 16000 # Ziel-Samplingrate (Ausgabe)

WINDOW_SIZE = 2.0 # 2 Sekunden
OVERLAP = 0.5     # 50% Überlappung

USE_LABELS = False  # Falls False, werden Alle Dateien verarbeitet, nicht nur gelabelte

# GPU/CPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
print(f"Preprocessing läuft auf: {DEVICE}")

# ------------------------------
# LABEL-DATEIEN EINLESEN
# ------------------------------

def load_labeled_files(label_folder):
    """Liest alle Label-Dateien ein und gibt eine Liste von WAV-Dateien zurück, die ein Label haben."""
    labeled_files = set()
    if USE_LABELS:
        for txt_file in os.listdir(label_folder):
            if txt_file.endswith(".Table.1.selections.txt"):
                wav_file = txt_file.replace(".Table.1.selections.txt", ".wav")
                labeled_files.add(wav_file)
    return labeled_files

# ------------------------------
# FILTERS & NOISE REDUCTION
# ------------------------------

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Creates a Butterworth bandpass filter."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return b, a

def bandpass_filter(data, lowcut=LOWCUT, highcut=HIGHCUT, fs=SR_ORIG, order=5):
    """Applies a bandpass filter to the audio signal."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def noise_reduction(data, sr=SR_ORIG):
    """Adaptive, konservative Noise‑Reduction mit noisereduce.

    – Ziel: stationäres Rauschen (Wind, Regen, Insekten‑Chorus) dämpfen,
      ohne kurze Froschrufe (≤ 1 s) zu beschädigen.
    – prop_decrease < 1 bewirkt eine moderate Absenkung der Rauschanteile.
    – stationary=False erlaubt eine zeitlich adaptive Rauschschätzung.
    """
    if len(data) == 0 or not HAS_NR:
        return data

    # Sanfte Parameter – lieber etwas Rauschen übrig lassen,
    # dafür die Calls unverzerrt behalten.
    reduced = nr.reduce_noise(
        y=data,
        sr=sr,
        stationary=False,          # Noise‑Profil dynamisch nachführen
        prop_decrease=0.6,         # konservative Absenkung (60 %)
        freq_mask_smooth_hz=200,   # glättet das Frequenz‑Maskenprofil
        time_mask_smooth_ms=50     # glättet das Zeit‑Maskenprofil
    )
    return reduced

# ------------------------------
# WINDOWING & RESAMPLING
# ------------------------------

def extract_and_resample_segment(data, sr_orig, start_sec, window_size_sec, sr_target):
    """Schneidet ein 2s-Fenster aus `data` und resampelt es auf `sr_target`."""
    start_sample = int(start_sec * sr_orig)
    end_sample = int((start_sec + window_size_sec) * sr_orig)

    if end_sample > len(data):
        return None

    segment = data[start_sample:end_sample]

    # Resample auf 16k
    segment_16k = librosa.resample(segment, orig_sr=sr_orig, target_sr=sr_target)
    return segment_16k

# ------------------------------
# FULL AUDIO PROCESSING PIPELINE
# ------------------------------

def process_wav_file(wav_file, output_folder,
                     window_size=WINDOW_SIZE, overlap=OVERLAP):
    """Verarbeitet eine WAV-Datei und speichert geschnittene 16kHz-Dateien."""
    try:
        base_name = os.path.splitext(os.path.basename(wav_file))[0]
        output_path = os.path.join(output_folder, base_name)

        print(f"[INFO] Verarbeite Datei: {wav_file}")

        # Lade Datei
        data, sr = librosa.load(wav_file, sr=SR_ORIG)
        if len(data) == 0:
            print(f"[WARN] Leere Datei: {wav_file}, übersprungen.")
            return

        # 1) Bandpass + Noise Reduction
        data = bandpass_filter(data, fs=sr)
        data = noise_reduction(data, sr=sr)

        if len(data) == 0:
            print(f"[WARN] Kein Signal nach Filterung: {wav_file}, übersprungen.")
            return

        # Ordner erstellen
        os.makedirs(output_path, exist_ok=True)

        duration_sec = len(data) / sr
        step_sec = window_size * (1 - overlap)

        # 2) Fensterung + Resample -> WAV
        times = np.arange(0, duration_sec - window_size + step_sec, step_sec)
        for start_sec in times:
            snippet_16k = extract_and_resample_segment(
                data, sr_orig=sr,
                start_sec=start_sec,
                window_size_sec=window_size,
                sr_target=SR_TARGET
            )
            if snippet_16k is None:
                continue

            # Speichern
            out_fname = f"{base_name}_{int(start_sec * 1000)}.wav"
            out_path = os.path.join(output_path, out_fname)
            sf.write(out_path, snippet_16k, SR_TARGET)

        # Prüfen, ob der Ordner leer geblieben ist
        if not os.listdir(output_path):
            print(f"[WARN] Keine Snippets erzeugt für {base_name}, überprüfe Filtereinstellungen!")

    except Exception as e:
        print(f"[ERROR] Fehler bei {wav_file}: {e}")

def process_directory(input_folder, output_folder, label_folder,
                      window_size=WINDOW_SIZE, overlap=OVERLAP, use_labels=True):
    """Verarbeitet WAV-Dateien mit oder ohne Labels, basierend auf `use_labels`."""
    os.makedirs(output_folder, exist_ok=True)

    if use_labels:
        labeled_files = load_labeled_files(label_folder)
        print(f"[INFO] {len(labeled_files)} Dateien haben ein Label.")
        wav_files = [os.path.join(input_folder, f)
                     for f in os.listdir(input_folder)
                     if f.endswith(".wav") and f in labeled_files]
    else:
        wav_files = [os.path.join(input_folder, f)
                     for f in os.listdir(input_folder)
                     if f.endswith(".wav")]

    if not wav_files:
        print("[ERROR] Keine passenden WAV-Dateien gefunden.")
        return

    print(f"[INFO] Verarbeite {len(wav_files)} Dateien aus {input_folder} (Labels: {use_labels})")

    # Parallel processing with global progress bar
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_wav_file, wav_file, output_folder, window_size, overlap): wav_file for wav_file in wav_files}
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing all files"):
            pass

    print("[INFO] Alle Audiodateien wurden verarbeitet.")

# ------------------------------
# EXECUTION
# ------------------------------

if __name__ == "__main__":
    input_folder = "/Volumes/QuebSSD/bioacoustic_data/Data/Raw"
    output_folder = "/Volumes/QuebSSD/bioacoustic_data/Data/new_pre/Processed_16k_inf_all"
    label_folder = "/Volumes/QuebSSD/bioacoustic_data/Data/Labels"

    # Hier kannst du Labels aktivieren oder deaktivieren
    process_directory(input_folder, output_folder, label_folder, use_labels=USE_LABELS)