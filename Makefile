# Makefile for Lora Fine-tune Pipeline

# Make Script to Run Finetuning Pipeline

# Make Script to Run Inference Script


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

help: ## Display this help screen
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[$$()% a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
