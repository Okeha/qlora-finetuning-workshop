# 🧠 QLoRA Fine-Tuning Workshop

> A hands-on, end-to-end workshop for fine-tuning a large language model on your own custom dataset using **QLoRA** — Parameter-Efficient Fine-Tuning with 4-bit quantization.

---

## ⚠️ Compatibility Notice

> **This workshop is designed and tested for WSL (Windows Subsystem for Linux) only.**
> Running directly on native Windows or macOS is not supported. Please ensure you have WSL 2 configured before proceeding.

---

## 📋 Prerequisites

Before you start, make sure your machine meets the following requirements:

| Requirement | Details |
|---|---|
| **OS** | WSL 2 (Ubuntu 20.04+ recommended) |
| **GPU** | NVIDIA GPU with **≥ 8 GB VRAM** (16 GB+ recommended for larger models) |
| **CUDA** | CUDA 11.8 or 12.x installed and accessible from WSL |
| **Python** | 3.13+ |
| **RAM** | 16 GB system RAM minimum |
| **Disk** | ~20 GB free space for model weights and checkpoints |
| **[`uv`](https://docs.astral.sh/uv/)** | Fast Python package manager — used for environment setup |

> **No GPU? No problem (kind of).** You can still follow along on CPU, but fine-tuning will be extremely slow. This workshop is best experienced with a capable NVIDIA GPU.

---

## 🗺️ Workshop Overview

The pipeline has three simple stages:

```
1. Generate Dataset  →  2. Fine-Tune the Model  →  3. Run Inference
```

---

## 🚀 Step-by-Step Guide

### Step 1 — Generate Your Dataset

Your training data is the heart of this workshop. You'll generate it using an AI assistant of your choice.

1. Open `src/config/prompts.yaml` and copy the `dataset_generation_prompt` value.
2. Paste it into one of the following AI providers:
   - **[Mistral AI](https://chat.mistral.ai)** — Le Chat
   - **[Microsoft Copilot](https://copilot.microsoft.com)** — powered by GPT-4.5 / GPT-5.5
3. The AI will generate a structured JSONL dataset for you (one JSON object per line).
4. Copy the generated JSONL and paste it into:

   ```
   src/finetune/data/custom_dataset.jsonl
   ```

> Expected JSONL format (one object per line):
> ```
> {"input_prompt": "What is X?", "expected_response": "Y is..."}
> {"input_prompt": "How do I do Z?", "expected_response": "To do Z, first..."}
> ```
> Make sure each line is a valid JSON object with `input_prompt` and `expected_response` fields.

---

### Step 2 — Configure Your Run

Open `src/config/params.yaml` and adjust values as needed (defaults are already tuned for small datasets, ~30-50 samples):

- **`model_name`** — Hugging Face model ID to fine-tune (default: `Qwen/Qwen3.5-0.8B`)
- **LoRA hyperparameters** — `rank` (8), `alpha` (16), `dropout` (0.1), `target_modules` (attention + MLP layers)
- **Training settings** — `batch_size` (2), `num_epochs` (6), `learning_rate` (2e-4), etc.
- **Paths** — `dataset_path` and `final_model_path` for your dataset and finetuned adapter output

> Defaults are production-ready for small datasets; adjust `num_epochs`, `learning_rate`, and `batch_size` if overfitting/underfitting occurs.

---

### Step 3 — Setup the Environment

Install all dependencies using `uv`:

```bash
uv sync
```

This will:
- Create and activate a `.venv` virtual environment
- Install all required packages (PyTorch, Transformers, PEFT, TRL, bitsandbytes, and more)

> **HuggingFace Token**: Some models require authentication. Create a `.env` file in the project root:
> ```
> HF_TOKEN=hf_your_token_here
> ```
> Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

---

### Step 4 — Fine-Tune the Model

Kick off QLoRA fine-tuning with a single command:

```bash
make finetune
```

This will:
1. Load JSONL dataset and split into train/validation
2. Load base model in 4-bit quantization (QLoRA)
3. Attach LoRA adapters to key attention/MLP layers
4. Train using TRL `SFTTrainer` with your custom dataset
5. Save finetuned adapter weights to `final_model_path`
6. Log metrics to TensorBoard (available at `output_dir/runs`)

> Training progress, eval metrics, and stage-by-stage logs are printed in pretty formatted tables. Checkpoints and the final adapter are saved to `output_dir`.

---

### Step 5 — Run Inference

Once training is complete, load your fine-tuned LoRA adapter and chat interactively:

```bash
make infer
```

You'll be prompted for prompts at `You:`. The fine-tuned model will generate responses using the LoRA adapter weights. Type `exit` to quit — this is your moment. 🎉

> Under the hood:
> - Loads base model in 4-bit quantization
> - Attaches your saved LoRA adapter
> - Uses sampling (temperature 0.7, top-p 0.9) for creative but coherent responses
> - Includes repetition penalty to reduce token repetition

---

## 📁 Project Structure

```
lora-finetune-pipeline/
├── Makefile                        # Commands: finetune, infer
├── main.py                         # Entry point (optional)
├── pyproject.toml                  # Python dependencies (managed by uv)
├── .env.example                    # Template for HuggingFace token
└── src/
    ├── config/
    │   ├── params.yaml             # Model name, LoRA & training hyperparameters + paths
    │   └── prompts.yaml            # (Optional) dataset generation prompt template
    ├── finetune/
    │   ├── dataloader.py           # JSONL dataset loader with train/val split
    │   ├── model.py                # Shared model manager for training + inference
    │   ├── finetune.py             # Main QLoRA training pipeline (TRL + PEFT)
    │   └── data/
    │       └── custom_dataset.jsonl # ← Your JSONL dataset (one object per line)
    ├── inference/
    │   └── infer.py                # LoRA adapter loader + interactive chat CLI
    └── utils/
        └── helpers.py              # (Optional) utility functions
```

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `torch` | Core deep learning framework |
| `transformers` | Model loading and tokenization |
| `peft` | LoRA / QLoRA adapter support |
| `trl` | Supervised fine-tuning trainer (SFTTrainer) |
| `bitsandbytes` | 4-bit quantization for QLoRA |
| `accelerate` | Multi-GPU and mixed-precision training |
| `datasets` | Dataset loading and processing |
| `wandb` / `tensorboard` | Training monitoring and metrics |

---

## 💡 What is QLoRA?

**QLoRA** (Quantized Low-Rank Adaptation) is a technique that makes fine-tuning large language models accessible on consumer hardware by:

1. **Quantizing** the base model to **4-bit precision** to drastically reduce memory usage.
2. **Injecting trainable LoRA adapters** into key layers — only these small adapters are updated during training, not the full model weights.

The result: you can fine-tune a 7B+ parameter model on a single consumer GPU with as little as 8 GB of VRAM.

---

## 🔑 Configuration

### HuggingFace Token (Optional for gated models)

Some models on the Hugging Face Hub (e.g. Llama, other gated models) require authentication:

```bash
cp .env.example .env
```

Edit `.env` and add your token:

```
HF_TOKEN=hf_your_token_here
```

Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with **read** access.

> `.env` is git-ignored — never commit your token.

### Hyperparameter Tuning

For datasets smaller than 50 samples, defaults in `params.yaml` are tuned to avoid overfitting:
- `batch_size: 2` + `gradient_accumulation_steps: 8` = effective batch of 16
- `num_epochs: 6` — enough to learn format without memorization
- `learning_rate: 2e-4` — conservative for adapter-only training
- `eval_steps: 5` — frequent eval to catch overfitting early
- `lora_rank: 8` — low rank to constrain capacity

For larger datasets (>100 samples), consider:
- Increasing `batch_size` to 4–8
- Reducing `num_epochs` to 3–4
- Increasing `lora_rank` to 16–32

---

## 📊 Project Status

| Area | Status |
|---|---|
| Project structure & layout | ✅ Complete |
| Dependency manifest (`pyproject.toml`) | ✅ Complete |
| Config files (`params.yaml`, `prompts.yaml`) | ✅ Complete — tuned defaults for small datasets |
| Dataset loader (`dataloader.py`) | ✅ Complete — JSONL format with train/val split |
| Shared model manager (`model.py`) | ✅ Complete — used by training & inference |
| Fine-tuning pipeline (`finetune.py`) | ✅ Complete — TRL SFTTrainer + PEFT LoRA + 4-bit quantization |
| Inference script (`infer.py`) | ✅ Complete — LoRA adapter loading + interactive chat |
| Makefile targets | ✅ Complete — `finetune` and `infer` |
| Pretty logging | ✅ Complete — rich-formatted stage logs + summary tables |
| GPU memory cleanup | ✅ Complete — post-training garbage collection + CUDA cache clearing |
| README | ✅ Complete |

---

## 🎯 What's Implemented

✨ **Full end-to-end QLoRA pipeline**:
- **JSONL dataset loader** with Hugging Face `datasets.load_dataset` integration
- **4-bit quantization** via `bitsandbytes` NF4 + double quantization
- **LoRA adapters** attached to attention & MLP layers via PEFT
- **TRL SFTTrainer** for supervised fine-tuning with train/eval datasets
- **TensorBoard logging** for metrics tracking
- **Pretty console output** with rich formatting (stage logs, summary tables, errors)
- **Interactive inference CLI** with temperature/top-p sampling
- **GPU memory cleanup** to prevent allocator fragmentation

---

## 🤝 Acknowledgements

Built with ❤️ using the Hugging Face ecosystem — [`transformers`](https://github.com/huggingface/transformers), [`peft`](https://github.com/huggingface/peft), [`trl`](https://github.com/huggingface/trl), and [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes).
