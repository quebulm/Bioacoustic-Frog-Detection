# Bioacoustic Frog Detection

## Projektbeschreibung

Dieses interdisziplinäre Forschungsprojekt beschäftigt sich mit der automatisierten Detektion und Klassifikation bioakustischer Signale von bedrohten Laubfroscharten der Gattung *Leptopelis* in Südafrika mittels Machine Learning. Ziel ist es, eine quantitative Analyse der Aktivitätsmuster dieser Froscharten zu ermöglichen, um anschließend Korrelationen mit Umweltfaktoren wie Wetterdaten herzustellen.

Die Forschung erfolgt in Zusammenarbeit mit der University of KwaZulu-Natal.

---

## Projektdatenstruktur

```
Bioacoustic-Frog-Detection/
├── configs/
│   ├── base.yaml
│   ├── best_params.yaml
│   ├── eval.yaml
│   ├── features.yaml
│   ├── infer.yaml
│   ├── preprocess.yaml
│   ├── train.yaml
│   └── tune.yaml
├── scripts/
│   ├── evaluate.py
│   ├── infer.py
│   ├── make_embeddings.py
│   ├── preprocess.py
│   ├── train.py
│   └── tune.py
├── src/
│   └── bioacoustic_frog_detection/
│       ├── data/
│       │   ├── __init__.py
│       │   ├── features.py
│       │   ├── filters.py
│       │   └── io.py
│       ├── eval/
│       │   └── on_embeddings.py
│       └── models/
│           ├── __init__.py
│           ├── classifier.py
│           └── config.py
├── Makefile
```

```
<root_dir>/                    # über base.yaml setzen
├── Data/
│   ├── Raw/                   # Input-WAVs
│   ├── Raw_Infer/             # WAVs nur für Inferenz
│   ├── Processed_16k/         # Preprocessed Snippets (Training) - Wird erstellt 
│   ├── Processed_16k_infer/   # Preprocessed Snippets (Inference) - Wird erstellt 
│   ├── Labels/                # Annotationen / Ground Truth 
│   ├── Test_Labels/           # Test-Labels (separat)
│   ├── X_embeddings_v0.npy    # Training Embeddings - Wird erstellt 
│   ├── y_labels_v0.npy        # Training Labels - Wird erstellt 
│   └── Inference_Results/     # Ausgaben von infer.py (CSV für Raven Pro) - Wird erstellt 
├── models/                    # gespeicherte Modelle
└── outputs/                   # Logs, tuning.json, Metrics
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

