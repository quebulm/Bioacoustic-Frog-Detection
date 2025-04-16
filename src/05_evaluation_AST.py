import os
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import precision_recall_curve, auc

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

    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, marker='.')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve (AUC = {pr_auc:.3f})")
    plt.grid(True)
    plt.show()

# ------------------------------
#  Haupt-Funktion für Evaluation
# ------------------------------

if __name__ == "__main__":
    print("[INFO] Lade Modell ...")
    MODEL_PATH = "../models/best_model.h5"
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

    print("[INFO] Lade vorberechnete Embeddings und Labels ...")
    X_embeddings = np.load("../data/X_embeddings_Test_16k.npy")
    y_true = np.load("../data/y_labels_Test.npy")

    print("[INFO] Starte Inferenz ...")
    probs = classifier.predict(X_embeddings, batch_size=32)
    y_pred = [1 if p[0] > 0.637 else 0 for p in probs]

    print("[INFO] Berechne Evaluationsmetriken ...")
    evaluate_model(y_true, y_pred)