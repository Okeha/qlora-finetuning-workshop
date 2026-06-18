# Makefile for Lora Fine-tune Pipeline

# Make Script to Run Finetuning Pipeline

# Make Script to Run Inference Script

TB_LOGDIR ?= finetuned_model/qlora-qwen3.5-0.8b/runs
TB_HOST ?= 0.0.0.0
TB_PORT ?= 6006


default: help

setup: ## setup environment and install dependencies
	@echo "Setting up environment, installing dependencies and activate virtual environment..."
	uv sync && source .venv/bin/activate


finetune: ## Start finetuning pipeline with custom dataset
	@echo "Starting Fine-Tuning Pipeline..."
	uv run python -m src.finetune.finetune


infer: ## Run inference with fine-tuned LoRA adapter
	@echo "Starting Inference..."
	uv run python -m src.inference.infer

tensorboard: ## Launch TensorBoard (override with TB_LOGDIR=..., TB_HOST=..., TB_PORT=...)
	@echo "Starting TensorBoard at http://$(TB_HOST):$(TB_PORT) using logdir: $(TB_LOGDIR)"
	uv run tensorboard --logdir "$(TB_LOGDIR)" --host "$(TB_HOST)" --port "$(TB_PORT)"

tb: tensorboard ## Alias for tensorboard

help: ## Display this help screen
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[$$()% a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
