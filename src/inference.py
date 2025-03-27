import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import librosa
import re
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from tensorflow import keras

# ------------------------------
# 1 Modell laden
# ------------------------------

MODEL_PATH = "../models/frog_call_classifier"

# Versuche `.keras`, dann `.h5` als Fallback
try:
    classifier = keras.models.load_model(f"{MODEL_PATH}.keras")
    print(f"[INFO] Keras-Modell geladen: {MODEL_PATH}.keras")
except:
    print(f"[WARN] Keras-Modell nicht gefunden, versuche H5-Format ...")
    classifier = keras.models.load_model(f"{MODEL_PATH}.h5", compile=False)
    classifier.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                       loss=tf.keras.losses.BinaryCrossentropy(),
                       metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    print(f"[INFO] H5-Modell geladen und neu kompiliert: {MODEL_PATH}.h5")

# ------------------------------
# 2 YAMNet-Modell laden
# ------------------------------

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

def get_yamnet_embedding(audio_waveform, sr=16000):
    """Berechnet das YAMNet-Embedding für eine Audiodatei."""
    _, embeddings, _ = yamnet_model(audio_waveform)
    return np.mean(embeddings.numpy(), axis=0)  # Mittelwert über alle Frames

# ------------------------------
# 3 Audiodateien effizient verarbeiten
# ------------------------------

def process_snippet(snippet_path, basename):
    """
    Verarbeitet ein einzelnes Audio-Snippet:
    - Lädt die Datei
    - Berechnet das YAMNet-Embedding
    - Gibt Startzeit & Embedding zurück
    """
    try:
        match = re.search(rf"{basename}_(\d+)\.wav", snippet_path)
        if not match:
            return None  # Skip, falls das Format nicht passt

        start_ms = float(match.group(1))
        start_sec = start_ms / 1000.0  # Startzeit des Snippets

        # Lade Snippet (Mono, 16kHz)
        wav_data, sr_ = librosa.load(snippet_path, sr=None)
        if sr_ != 16000:
            print(f"[WARN] {snippet_path} hat sr={sr_}, erwartet 16000.")
        if len(wav_data) == 0:
            return None  # Skip leere Dateien

        # YAMNet Embedding berechnen
        emb = get_yamnet_embedding(wav_data, sr=16000)
        return start_sec, emb

    except Exception as e:
        print(f"[ERROR] Fehler beim Laden {snippet_path}: {e}")
        return None

def process_directory(processed_folder, output_folder, batch_size=32, max_workers=4):
    """
    Durchläuft alle Snippets in `processed_folder`, verarbeitet sie parallel und speichert die Vorhersagen als CSV.
    """
    os.makedirs(output_folder, exist_ok=True)

    for basename in tqdm(os.listdir(processed_folder), desc="Verarbeite Audiodateien"):
        subdir = os.path.join(processed_folder, basename)
        if not os.path.isdir(subdir):
            continue

        snippet_files = sorted([os.path.join(subdir, f) for f in os.listdir(subdir)
                                if f.endswith(".wav") and f.startswith(basename + "_")])

        if not snippet_files:
            print(f"[WARN] Keine Snippet-Dateien gefunden für {basename}, skip...")
            continue

        # Parallel Snippets verarbeiten
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda s: process_snippet(s, basename), snippet_files))

        # Entferne None-Werte (Fehlgeschlagene Dateien)
        results = [res for res in results if res is not None]

        if not results:
            print(f"[INFO] Keine gültigen Snippets für {basename}, skip...")
            continue

        # Extrahiere Startzeiten & Embeddings
        start_times, embeddings = zip(*results)
        embeddings = np.stack(embeddings)  # In NumPy-Array konvertieren für Batch-Verarbeitung

        # Modellvorhersagen als Batch
        probs = classifier.predict(embeddings, batch_size=batch_size)

        # Speichere nur Wahrscheinlichkeiten > 0.5
        predictions = [
            {"Sekunde": int(start_sec),
             "Minute": f"{int(start_sec // 60)}:{int(start_sec % 60):02d}",
             "Wahrscheinlichkeit": round(prob[0], 3)}
            for start_sec, prob in zip(start_times, probs) if prob[0] > 0.5
        ]

        # CSV-Datei speichern
        if predictions:
            df = pd.DataFrame(predictions)
            csv_path = os.path.join(output_folder, f"{basename}_inf.csv")
            df.to_csv(csv_path, index=False)
            print(f"[INFO] Ergebnisse gespeichert: {csv_path}")
        else:
            print(f"[INFO] Keine relevanten Vorhersagen für {basename}, keine CSV erstellt.")

# ------------------------------
# 4 Haupt-Funktion für Inferenz
# ------------------------------

if __name__ == "__main__":
    processed_folder = "../data/Processed_16k_inf"
    output_folder = "../data/Inference_Results"

    print(f"[INFO] Lade Snippets aus {processed_folder} ...")
    process_directory(processed_folder, output_folder)

    print("[INFO] Inferenz abgeschlossen.")