import gc
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score
import optuna


# ------------------------------
#  Daten laden (YAMNet-Embeddings)
# ------------------------------

def load_data(data_folder="../data/"):
    X = np.load(os.path.join(data_folder, "X_embeddings_v0_16k_aug.npy"))  # AST-Features (N, 768)
    y = np.load(os.path.join(data_folder, "y_labels_v0_aug.npy"))      # Labels (N,)

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
def focal_loss(alpha=0.85, gamma=2.0):
    """
    Focal Loss – verstärkt den Fokus auf schwierige Beispiele.
    alpha > 0.5 bevorzugt die positive Klasse (Ruf).
    gamma=2.0 erhöht den Fokuseffekt.
    """

    def loss(y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        fl = alpha * (1 - p_t) ** gamma * bce
        return tf.reduce_mean(fl)

    return loss


# ------------------------------
#  Klassifikationsmodell auf Basis der AST-Features
# ------------------------------
def build_classifier(alpha, gamma, dropout_rate, learning_rate):
    inputs = keras.Input(shape=(768,))  # Eingabe: AST-Embeddings (768,)

    x = layers.BatchNormalization()(inputs)

    # Block 1
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Block 2
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Optional: Skip-Connection von Block 1
    # x = layers.add([x, skip1])  # Nur wenn du einen "short cut" willst

    # Block 3
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Block 4
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Block 5
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    # Output
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=focal_loss(alpha=alpha, gamma=gamma),
        metrics=[
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc")
        ]
    )
    return model


# ------------------------------
#  Trainings- und Evaluationsroutine für einen Optuna-Trial
# ------------------------------

def objective(trial):
    """
    Ziel: Maximierung des f0.5-Scores (höhere Gewichtung auf Präzision).
    Wir probieren hier einen breiteren und 'radikaleren' Suchraum aus,
    um mögliche Modelle zu finden, die weniger Falschklassifikationen haben.
    """

    # Hyperparameter-Suchspektrum:
    alpha = trial.suggest_float("alpha", 0.5, 1.0)
    gamma = trial.suggest_float("gamma", 1.0, 3.0)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    threshold = trial.suggest_float("threshold", 0.5, 0.8)
    pos_weight = trial.suggest_float("pos_weight", 1.5, 5.0)
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.4)

    # Daten laden (fester Split)
    X_train, X_val, y_train, y_val = load_data()

    # Modell erstellen
    model = build_classifier(alpha, gamma, dropout_rate, learning_rate)
    # Kein ausführliches model.summary() in jedem Trial
    class_weights = {0: 1.0, 1: pos_weight}

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            min_delta=0.0001,
            restore_best_weights=True
        )
    ]

    # Training
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=64,
        epochs=50,  #
        callbacks=callbacks,
        verbose=0,
        class_weight=class_weights
    )

    # Vorhersagen mit trainiertem Modell
    y_pred_proba = model.predict(X_val)
    y_pred = (y_pred_proba > threshold).astype("int32")

    # f0.4-Score (stärkeres Gewicht auf Präzision)
    f05 = fbeta_score(y_val, y_pred, beta=0.4)

    # Confusion Matrix berechnen und loggen
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_val, y_pred)
    cm_str = (
        f"\n[Trial {trial.number}] Confusion Matrix (Threshold = {threshold:.2f}):\n"
        f"               Predicted 0    Predicted 1\n"
        f"Actual 0       {cm[0, 0]:<14}{cm[0, 1]}\n"
        f"Actual 1       {cm[1, 0]:<14}{cm[1, 1]}\n"
    )
    print(cm_str)
    with open("confusion_log.txt", "a") as f:
        f.write(cm_str)

    # Optuna Feedback
    trial.report(f05, step=1)

    # Speicher bereinigen
    tf.keras.backend.clear_session()
    gc.collect()

    return f05


# ------------------------------
#  Haupt-Funktion: Optuna-Studie
# ------------------------------

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("\n[INFO] Beste Hyperparameter:")
    for key, value in study.best_trial.params.items():
        print(f"  {key}: {value}")

    best_params = study.best_params
    alpha_best = best_params["alpha"]
    gamma_best = best_params["gamma"]
    dropout_rate_best = best_params["dropout_rate"]
    learning_rate_best = best_params["learning_rate"]
    pos_weight_best = best_params["pos_weight"]
    threshold_best = best_params["threshold"]

    print("\n[INFO] Training final model with best hyperparams...")
    X_train_all, X_val_all, y_train_all, y_val_all = load_data()
    X_entire = np.concatenate([X_train_all, X_val_all], axis=0)
    y_entire = np.concatenate([y_train_all, y_val_all], axis=0)
    print("[INFO] Training on the entire dataset:", X_entire.shape, y_entire.shape)

    model_best = build_classifier(alpha_best, gamma_best, dropout_rate_best, learning_rate_best)
    class_weights = {0: 1.0, 1: pos_weight_best}

    model_best.fit(
        X_entire, y_entire,
        batch_size=64,
        epochs=50,
        class_weight=class_weights,
        verbose=1
    )

    model_best.save("../data/best_model.h5")
    print("[INFO] Best model saved as best_model.h5")