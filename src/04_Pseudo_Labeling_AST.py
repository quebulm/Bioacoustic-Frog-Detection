import tensorflow as tf
import numpy as np
import os
import re
import pandas as pd
import keras

MODEL_PATH = "../models/best_model"
USE_PRECOMPUTED_EMBEDDINGS = True

try:
    classifier = keras.models.load_model(f"{MODEL_PATH}.keras")
    print(f"[INFO] Keras-Modell geladen: {MODEL_PATH}.keras")
except:
    print(f"[WARN] Keras-Modell nicht gefunden, versuche H5-Format ...")
    classifier = keras.models.load_model(f"{MODEL_PATH}.h5", compile=False)
    classifier.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    print(f"[INFO] H5-Modell geladen und neu kompiliert: {MODEL_PATH}.h5")

def process_directory(processed_folder, output_folder, embeddings, threshold=0.2,
                      batch_size=32, snippet_length=2.0):
    """
    Lädt die Dateinamen aller Snippets in `processed_folder`, ordnet ihnen nacheinander die
    vorberechneten `embeddings` zu, führt die Vorhersage durch und speichert pro Basename
    ein CSV sowie eine Raven‑Style .Table.1.selections.txt.
    Die Reihenfolge der Snippets MUSS dieselbe sein, in der die Embeddings erzeugt wurden.
    """
    os.makedirs(output_folder, exist_ok=True)

    idx = 0  # aktueller Index im Embedding‑Array
    basenames = sorted(
        d for d in os.listdir(processed_folder)
        if os.path.isdir(os.path.join(processed_folder, d))
    )

    for basename in basenames:
        subdir = os.path.join(processed_folder, basename)
        snippet_files = sorted(
            f for f in os.listdir(subdir)
            if f.endswith(".wav") and f.startswith(basename + "_")
        )

        if not snippet_files:
            print(f"[WARN] Keine Snippet‑Dateien für {basename}, skip ...")
            continue

        n = len(snippet_files)
        emb_batch = embeddings[idx:idx + n]
        idx += n

        if emb_batch.shape[0] != n:
            print(f"[ERROR] Zahl der Embeddings passt nicht zu {basename} – Abbruch.")
            return

        probs = classifier.predict(emb_batch, batch_size=batch_size)
        predictions = []
        for file_name, prob in zip(snippet_files, probs):
            start_match = re.search(rf"{basename}_(\d+)\.wav", file_name)
            if not start_match:
                continue
            start_ms = float(start_match.group(1))
            start_sec = start_ms / 1000.0
            if prob[0] > threshold:
                predictions.append(
                    {
                        "Sekunde": float(start_sec),
                        "Minute": f"{int(start_sec // 60)}:{int(start_sec % 60):02d}",
                        "Wahrscheinlichkeit": round(prob[0], 3)
                    }
                )

        if not predictions:
            print(f"[INFO] Keine relevanten Vorhersagen für {basename}.")
            continue

        # CSV speichern
        df = pd.DataFrame(predictions)
        csv_path = os.path.join(output_folder, f"{basename}_inf.csv")
        df.to_csv(csv_path, index=False)
        print(f"[INFO] CSV gespeichert: {csv_path}")

        # Raven‑Style TXT
        header_cols = [
            "Selection", "View", "Channel", "Begin Time (s)",
            "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)", "Species"
        ]
        lines = ["\t".join(header_cols)]
        for i, row in enumerate(predictions, 1):
            begin = row["Sekunde"]
            end = begin + snippet_length
            line_data = [
                str(i), "Spectrogram 1", "1",
                f"{begin:.6f}", f"{end:.6f}",
                "842.6", "2195.7", "lx"
            ]
            lines.append("\t".join(line_data))

        txt_path = os.path.join(output_folder, f"{basename}.Table.1.selections.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[INFO] Raven‑Format gespeichert: {txt_path}")

if __name__ == "__main__":
    #EMB_PATH = "../data/X_embeddings_Test_16k.npy"
    EMB_PATH = "../data/Embeddings/X_embeddings_Test_16k.npy"
    processed_folder = "../data/Test_16k"
    output_folder = "../data/Inference_Results/Test_16k"

    print(f"[INFO] Lade vorberechnete Embeddings aus {EMB_PATH} ...")
    X = np.load(EMB_PATH)
    print(f"[INFO] Shape von X: {X.shape}")

    process_directory(processed_folder, output_folder, X, threshold=0.6)
    print("[INFO] Inferenz + Label‑Export abgeschlossen.")