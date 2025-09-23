# -------- Settings --------
PYTHON ?= python
CONFIG_DIR= configs
export PYTHONPATH := src

PREPROCESS_CFG := $(CONFIG_DIR)/preprocess.yaml
FEATURES_CFG   := $(CONFIG_DIR)/features.yaml
TRAIN_CFG      := $(CONFIG_DIR)/train.yaml
EVAL_CFG       := $(CONFIG_DIR)/eval.yaml
INFER_CFG      := $(CONFIG_DIR)/infer.yaml
TUNE_CFG       := $(CONFIG_DIR)/tune.yaml
BEST_PARAMS   := $(CONFIG_DIR)/best_params.yaml

# -------- Top-level targets --------
.PHONY: all preprocess embed train train_tuned tune evaluate infer install precommit help

all: preprocess embed train evaluate infer

preprocess:
	@echo "==> Preprocessing"
	$(PYTHON) scripts/preprocess.py --config $(PREPROCESS_CFG)

embed:
	@echo "==> Embeddings"
	$(PYTHON) scripts/make_embeddings.py --config $(FEATURES_CFG)

train:
	@echo "==> Training"
	$(PYTHON) scripts/train.py --config $(TRAIN_CFG)

train_tuned:
	@echo "==> Training (with tuned params from $(BEST_PARAMS))"
	$(PYTHON) scripts/train.py --config $(TRAIN_CFG) --best-params $(BEST_PARAMS)

tune:
	@echo "==> Tuning (export best params to $(BEST_PARAMS))"
	$(PYTHON) scripts/tune.py --config $(TUNE_CFG) --export-params $(BEST_PARAMS)

evaluate:
	@echo "==> Evaluation"
	$(PYTHON) scripts/evaluate.py --config $(EVAL_CFG)

infer:
	@echo "==> Inference"
	$(PYTHON) scripts/infer.py --config $(INFER_CFG)

# -------- Dev convenience --------
install:
	@echo "==> Installing project (pyproject.toml)"
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .

precommit:
	@echo "==> Installing pre-commit hooks"
	$(PYTHON) -m pip install pre-commit
	pre-commit install

help:
	@echo "Targets:"
	@echo "  make preprocess   # Vorverarbeitung (Bandpass/Denoise/Segment/Resample)"
	@echo "  make embed        # YAMNet-Embeddings berechnen"
	@echo "  make train        # Modell trainieren"
	@echo "  make train_tuned  # Mit besten Parametern (configs/best_params.yaml) trainieren"
	@echo "  make tune         # Optuna laufen lassen und best params nach configs/best_params.yaml schreiben"
	@echo "  make evaluate     # Berichte/Metriken erzeugen"
	@echo "  make infer        # Inferenz auf neuen Audios"
	@echo "  make all          # preprocess → embed → train → evaluate"
	@echo "  make install      # Dependencies aus pyproject.toml installieren"
	@echo "  make precommit    # Git-Hooks (Black/Ruff/Isort) installieren"