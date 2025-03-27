import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
import optuna
import gc

# GPU-Konfiguration: dynamisches Memory-Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

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
# Focal Loss
# ------------------------------
def focal_loss(alpha, gamma):
    """
    Focal Loss – verstärkt den Fokus auf schwierige Beispiele.
    alpha > 0.5 bevorzugt die positive Klasse (Ruf).
    gamma > 1.0 erhöht den Fokuseffekt.
    """
    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        # p_t: Wahrscheinlichkeit für die korrekte Klasse
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        fl = alpha * (1 - p_t) ** gamma * bce
        return tf.reduce_mean(fl)
    return loss


# ------------------------------
#  Dynamische Modell-Architektur
# ------------------------------
def build_classifier(trial, input_dim=1024):
    """
    Baut ein Modell basierend auf den von Optuna vorgeschlagenen Parametern:
    - n_layers: Anzahl Dense-Schichten (1-3)
    - units_i: Neuronenzahl pro Schicht (64-512, log-scale)
    - dropout_i: Dropout-Rate pro Schicht (0.0-0.5)
    - alpha, gamma: Focal-Loss-Parameter
    - learning_rate: Lernrate für Adam
    """

    # Hyperparameter für den Focal-Loss
    alpha = trial.suggest_float("alpha", 0.5, 1.0)
    gamma = trial.suggest_float("gamma", 1.0, 4.0)

    # Anzahl DNN-Schichten
    n_layers = trial.suggest_int("n_layers", 1, 3)

    # Lernrate
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)

    # Modell beginnen
    model = keras.Sequential()
    # Eingabeschicht mit Batch-Norm
    model.add(layers.Input(shape=(input_dim,)))
    model.add(layers.BatchNormalization())

    # Baue n_layers Dense-Blöcke dynamisch
    for i in range(n_layers):
        # Anzahl Neuronen
        units = trial.suggest_int(f"units_{i}", 64, 256, log=True)
        # Dropout pro Layer
        dropout_rate = trial.suggest_float(f"dropout_rate_{i}", 0.0, 0.5)

        model.add(layers.Dense(units, activation="relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))

    # Letzte Schicht (Binär-Klassifikation)
    model.add(layers.Dense(1, activation="sigmoid"))

    # Compiler
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=focal_loss(alpha=alpha, gamma=gamma),
        metrics=[
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc")  # PR-AUC
        ]
    )

    return model


# ------------------------------
#  Objective-Funktion für Optuna
# ------------------------------
def objective(trial):
    """
    Ziel: Maximierung des F1-Scores (beta=1).
    1) Baue Modell (inkl. Architektur-Parameter).
    2) Trainiere mit dynamischen Batchgrößen, Epochen, Class Weights, Patience etc.
    3) Werte F1 aus.
    """
    # Daten laden (fester Split)
    X_train, X_val, y_train, y_val = load_data()

    # --------------------------------
    # 4.1) Weitere Trainings-Hyperparameter
    # --------------------------------
    pos_weight = trial.suggest_float("pos_weight", 1.0, 5.0)
    batch_size = trial.suggest_categorical("batch_size", [32, 64])
    max_epochs = trial.suggest_int("max_epochs", 30, 100)  # z.B. 30 - 100
    patience = trial.suggest_int("patience", 5, 20)        # Early-Stopping-Patience

    # --------------------------------
    # 4.2) Modell bauen (inkl. alpha, gamma)
    # --------------------------------
    model = build_classifier(trial, input_dim=X_train.shape[1])
    class_weights = {0: 1.0, 1: pos_weight}

    # --------------------------------
    # 4.3) Callbacks: EarlyStopping, optional LR-Scheduler
    # --------------------------------
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=1e-4,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,         # Lernrate halbieren bei Stagnation
            patience=5,         # Warte 5 Epochen, bevor du LR reduzierst
            min_lr=1e-6,        # Untergrenze
            verbose=0
        )
    ]

    try:
        # --------------------------------
        # 4.4) Training
        # --------------------------------
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=0
        )

        # --------------------------------
        # 4.5) Vorhersage & F1-Bewertung
        # --------------------------------
        y_pred_proba = model.predict(X_val)
        # Fester Schwellenwert 0.5 (später kann dieser manuell angepasst werden)
        y_pred = (y_pred_proba > 0.5).astype("int32")

        f1 = fbeta_score(y_val, y_pred, beta=1.0)

        # Ergebnis an Optuna melden
        trial.report(f1, step=1)
        return f1

    finally:
        # Ressourcen freigeben
        tf.keras.backend.clear_session()
        gc.collect()


# ------------------------------
#  Haupt-Funktion: Optuna-Studie
# ------------------------------
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=150, n_jobs=1)

    print("\n[INFO] Beste Hyperparameter:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    print(f"\n[INFO] Best F1-Score: {study.best_value:.4f}")