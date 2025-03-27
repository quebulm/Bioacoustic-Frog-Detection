import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc
import seaborn as sns

# ------------------------------
# Hyperparameter oben definieren
# ------------------------------
ALPHA = 0.8833093889136927
GAMMA = 2.9369526036291127
LEARNING_RATE = 0.00046871969344849544
POS_WEIGHT = 1.8292258553007865
BATCH_SIZE = 32
EPOCHS = 66
PATIENCE = 20
THRESHOLD = 0.5
MIN_DELTA = 0.0001

MODEL_SAVE_PATH = "../models/frog_call_classifier_v0_4"

# ------------------------------
#  Daten laden (YAMNet-Embeddings)
# ------------------------------
def load_data(data_folder="../data/"):
    X = np.load(os.path.join(data_folder, "X_embeddings_v0.npy"))  # YAMNet-Features (N, 1024)
    y = np.load(os.path.join(data_folder, "y_labels_v0.npy"))      # Labels (N,)

    print(f"[INFO] Geladene Daten: X: {X.shape}, y: {y.shape}")

    # Aufteilen in Trainings- & Validierungsset (80% Train, 20% Val), stratifiziert
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"[INFO] Trainingsdaten: {X_train.shape}, Validierungsdaten: {X_val.shape}")
    return X_train, X_val, y_train, y_val


# ------------------------------
#  Focal Loss (angepasst)
# ------------------------------
def focal_loss(alpha=ALPHA, gamma=GAMMA):
    """
    Focal Loss - stärkerer Fokus auf schwierige Beispiele.
    alpha > 0.5: begünstigt 'positive' Klasse (Ruf).
    gamma: erhöht den 'Fokus'-Effekt.
    """
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        fl = alpha * (1 - p_t) ** gamma * bce
        return tf.reduce_mean(fl)
    return loss


# ------------------------------
#  Klassifikationsmodell auf Basis der YAMNet-Features
# ------------------------------
def build_classifier():
    model = keras.Sequential([
        layers.Input(shape=(1024,)),  # Eingabe: YAMNet-Embeddings
        layers.BatchNormalization(),

        layers.Dense(203, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.030883388674336792),

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=focal_loss(alpha=ALPHA, gamma=GAMMA),
        metrics=[
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc")
        ]
    )

    return model


# ------------------------------
#  Training starten
# ------------------------------
def train_model(X_train, X_val, y_train, y_val, model_save_path=MODEL_SAVE_PATH):
    model = build_classifier()
    model.summary()

    # Klasse 1 (Froschruf) ist seltener → höheres Gewicht
    class_weights = {0: 1.0, 1: POS_WEIGHT}

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights
    )

    # Modell speichern
    model.save(f"{model_save_path}.h5", include_optimizer=False)
    model.save(f"{model_save_path}.keras")
    print(f"[INFO] Modell gespeichert unter {model_save_path}.h5 und {model_save_path}.keras")

    return model, history


# ------------------------------
#  Ergebnisse visualisieren & bewerten
# ------------------------------
def plot_learning_curves(history):
    """Zeigt die Entwicklung von Loss & Precision/Recall/PR-AUC"""
    plt.figure(figsize=(12, 5))

    # Verlustverlauf
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Trainings- und Validierungsverlust")
    plt.legend()

    # Precision-Recall-AUC-Verlauf
    plt.subplot(1, 2, 2)
    plt.plot(history.history["pr_auc"], label="Train PR-AUC")
    plt.plot(history.history["val_pr_auc"], label="Val PR-AUC")
    plt.xlabel("Epochs")
    plt.ylabel("PR-AUC")
    plt.title("Precision-Recall AUC-Verlauf")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba):
    """Berechnet und plottet die Precision-Recall-Kurve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.2f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()


def evaluate_model(model, X_val, y_val):
    """Berechnet Precision, Recall, F1-Score, plottet Confusion Matrix und Precision-Recall-Kurve."""
    y_pred_proba = model.predict(X_val)

    # Neue, optimierte Schwelle laut Hyperparameter-Suche
    threshold = THRESHOLD
    y_pred = (y_pred_proba > threshold).astype("int32")

    # Classification Report
    print("\nClassification Report (Threshold = {:.2f}):".format(threshold))
    print(classification_report(y_val, y_pred, digits=4))

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Kein Ruf (0)", "Ruf (1)"],
        yticklabels=["Kein Ruf (0)", "Ruf (1)"]
    )
    plt.xlabel("Vorhergesagt")
    plt.ylabel("Tatsächlich")
    plt.title(f"Confusion Matrix (Threshold={threshold})")
    plt.show()

    # Precision-Recall-Kurve plotten
    plot_precision_recall_curve(y_val, y_pred_proba)


# ------------------------------
#  Haupt-Funktion
# ------------------------------
if __name__ == "__main__":
    # Daten laden
    X_train, X_val, y_train, y_val = load_data()

    # Modell trainieren
    model, history = train_model(X_train, X_val, y_train, y_val)

    # Lernkurven plotten
    plot_learning_curves(history)

    # Modell evaluieren (inkl. Precision-Recall-Kurve)
    evaluate_model(model, X_val, y_val)

    # [I 2025 - 03 - 20 23: 0
    # 8: 12, 939] Trial
    # 77
    # finished
    # with value: 0.6746602717825739 and parameters: {'pos_weight': 1.8292258553007865, 'batch_size': 32,
    #                                                 'max_epochs': 66, 'patience': 20, 'alpha': 0.8833093889136927,
    #                                                 'gamma': 2.9369526036291127, 'n_layers': 1,
    #                                                 'learning_rate': 0.00046871969344849544, 'units_0': 203,
    #                                                 'dropout_rate_0': 0.030883388674336792}.Best is trial
    # 77
    # with value: 0.6746602717825739.