# Bioacoustic Frog Detection

## Projektbeschreibung

Dieses interdisziplinäre Forschungsprojekt beschäftigt sich mit der automatisierten Detektion und Klassifikation bioakustischer Signale von bedrohten Laubfroscharten der Gattung *Leptopelis* in Südafrika mittels Machine Learning. Ziel ist es, eine quantitative Analyse der Aktivitätsmuster dieser Froscharten zu ermöglichen, um anschließend Korrelationen mit Umweltfaktoren wie Wetterdaten herzustellen.

Die Forschung erfolgt in Zusammenarbeit mit der University of KwaZulu-Natal.

---

## Projektdatenstruktur

```
Bioacoustic-Frog-Detection/
├── Data/
│   ├── Raw/
│   ├── Labels/
│   ├── Processed_16k/
│   ├── Inference_Results/
│   ├── X_embeddings_v0.npy
│   └── y_labels_v0.npy
├── models/
├── src/
│   ├── 00_data_checker.py
│   ├── 01_preprocessing.py
│   ├── 02_data_prep.py
│   ├── 03_train.py
│   ├── 03_train_Optuna_allgm.py
│   ├── 04_Pseudo_Labeling.py
│   ├── 05_evaluation.py
│   └── inference.py
├── LICENSE
└── README.md
```

---

## Verwendete Technologien
- **Python**
- **TensorFlow & Keras** für Deep Learning
- **TensorFlow Hub (YAMNet)** zur Feature-Extraktion
- **Librosa, Soundfile** für Audioverarbeitung
- **Optuna** für Hyperparameter-Optimierung

---

## Projektschritte

1. **Datenerhebung & Annotation:**
   - Aufnahme von bioakustischen Daten und manuelle Annotation der Froschrufe.

2. **Vorverarbeitung:**
   - Noise Reduction und Bandpass-Filterung
   - Segmentierung der Audioaufnahmen in Snippets (2 Sekunden)
   - Resampling auf 16kHz

3. **Feature-Extraktion:**
   - Verwendung von YAMNet-Embeddings zur Repräsentation der Audiosegmente

4. **Modellentwicklung & Training:**
   - Erstellung und Training eines neuronalen Netzwerks mit optimierter Architektur und Hyperparametern (Optuna)

5. **Evaluierung:**
   - Umfassende Validierung des Modells mit klassischen Evaluationsmetriken (Accuracy, Precision, Recall, F1-Score, PR-AUC)

6. **Pseudo Labeling & Inferenz:**
   - Automatische Klassifikation unannotierter Audiodaten zur Erweiterung der Datenbasis

---

