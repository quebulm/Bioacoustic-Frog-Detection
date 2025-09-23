import os
import re
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ------------------------------
#  Modell laden
# ------------------------------

MODEL_PATH = "../models/frog_call_classifier_v0_3"

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
#  YAMNet-Modell laden
# ------------------------------

yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

def get_yamnet_embedding(audio_waveform, sr=16000):
    """Berechnet das YAMNet-Embedding für eine Audiodatei."""
    _, embeddings, _ = yamnet_model(audio_waveform)
    return np.mean(embeddings.numpy(), axis=0)  # Mittelwert über alle Frames

# ------------------------------
#  Labels laden
# ------------------------------

def load_test_labels(label_folder):
    """
    Lädt Testlabels aus .Table.1.selections.txt-Dateien.

    Args:
        label_folder (str): Pfad zum Label-Verzeichnis.

    Returns:
        dict: { "basename": [(start_sec, end_sec)] } mit allen Intervallen.
    """
    labels = {}

    for txt_file in os.listdir(label_folder):
        if not txt_file.endswith(".Table.1.selections.txt"):
            continue

        label_path = os.path.join(label_folder, txt_file)
        basename = txt_file.replace(".Table.1.selections.txt", "")

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
                print(f"[INFO] Testlabels für {basename} geladen.")

        except Exception as e:
            print(f"[ERROR] Fehler beim Laden von {label_path}: {e}")

    return labels

# ------------------------------
# Inferenz auf Testdaten
# ------------------------------

def process_snippet(snippet_path, basename):
    """Verarbeitet ein einzelnes Audio-Snippet und gibt Startzeit & Embedding zurück."""
    try:
        match = re.search(rf"{basename}_(\d+)\.wav", snippet_path)
        if not match:
            return None

        start_ms = float(match.group(1))
        start_sec = start_ms / 1000.0  # Startzeit des Snippets

        wav_data, sr_ = librosa.load(snippet_path, sr=None)
        if sr_ != 16000:
            print(f"[WARN] {snippet_path} hat sr={sr_}, erwartet 16000.")
        if len(wav_data) == 0:
            return None

        emb = get_yamnet_embedding(wav_data, sr=16000)
        return start_sec, emb

    except Exception as e:
        print(f"[ERROR] Fehler beim Laden {snippet_path}: {e}")
        return None

def run_inference(test_folder, labels_dict, batch_size=32, max_workers=4, snippet_length=2.0):
    """
    Führt Inferenz auf Testdaten durch und vergleicht mit Ground-Truth-Labels.

    Args:
        test_folder (str): Pfad zu den Test-Snippets.
        labels_dict (dict): Ground-Truth-Labels.

    Returns:
        y_true (list): Wahre Labels.
        y_pred (list): Vorhergesagte Labels.
    """
    y_true, y_pred = [], []

    for basename in tqdm(os.listdir(test_folder), desc="Verarbeite Testdaten"):
        subdir = os.path.join(test_folder, basename)
        if not os.path.isdir(subdir) or basename not in labels_dict:
            continue

        intervals = labels_dict[basename]

        snippet_files = sorted([os.path.join(subdir, f) for f in os.listdir(subdir)
                                if f.endswith(".wav") and f.startswith(basename + "_")])

        if not snippet_files:
            print(f"[WARN] Keine Snippet-Dateien gefunden für {basename}, skip...")
            continue

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(lambda s: process_snippet(s, basename), snippet_files))

        results = [res for res in results if res is not None]
        if not results:
            continue

        start_times, embeddings = zip(*results)
        embeddings = np.stack(embeddings)

        probs = classifier.predict(embeddings, batch_size=batch_size)

        for start_sec, prob in zip(start_times, probs):
            end_sec = start_sec + snippet_length
            label = 1 if prob[0] > 0.637 else 0
            y_pred.append(label)

            # Überprüfen, ob das Zeitfenster mit einem echten Froschruf überlappt
            is_true_call = any((start_sec < end and end_sec > start) for start, end in intervals)
            y_true.append(1 if is_true_call else 0)

    return y_true, y_pred

# ------------------------------
#  Evaluation durchführen
# ------------------------------

def evaluate_model(y_true, y_pred):
    """Berechnet Accuracy, Precision, Recall, F1-Score und zeigt die Confusion Matrix."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n[INFO] Modell-Evaluation:")
    print(f"  - Accuracy:  {acc:.3f}")
    print(f"  - Precision: {prec:.3f}")
    print(f"  - Recall:    {rec:.3f}")
    print(f"  - F1-Score:  {f1:.3f}")

    # Confusion Matrix erstellen
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Call", "Call"], yticklabels=["No Call", "Call"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()


# ------------------------------
#  Haupt-Funktion für Evaluation
# ------------------------------

if __name__ == "__main__":
    label_folder = "../data/Labels/Test"
    test_folder = "../data/Test_16k"

    print(f"[INFO] Lade Testlabels aus {label_folder} ...")
    test_labels = load_test_labels(label_folder)

    print(f"[INFO] Starte Inferenz auf Testdaten aus {test_folder} ...")
    y_true, y_pred = run_inference(test_folder, test_labels)

    print("[INFO] Berechne Evaluationsmetriken ...")
    evaluate_model(y_true, y_pred)