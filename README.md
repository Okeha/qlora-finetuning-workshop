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
3. The AI will generate a structured JSON dataset for you.
4. Copy the generated JSON and paste it into:

   ```
   src/finetune/data/custom_dataset.json
   ```

> Make sure the JSON follows the expected schema before moving on.

---

### Step 2 — Configure Your Run

Open `src/config/params.yaml` and fill in your desired values:

- **`model_name`** — the Hugging Face model ID you want to fine-tune (e.g. `mistralai/Mistral-7B-v0.1`)
- **LoRA hyperparameters** — `rank`, `alpha`, `dropout`, `target_modules`
- **Training settings** — `batch_size`, `num_epochs`, `learning_rate`, etc.

---

### Step 3 — Setup the Environment

Install all dependencies in one command:

```bash
make setup
```

This will:
- Create a `.venv` virtual environment using `uv`
- Install all required packages (PyTorch, Transformers, PEFT, TRL, bitsandbytes, and more)

---

### Step 4 — Fine-Tune the Model

Kick off QLoRA fine-tuning with a single command:

```bash
make finetune
```

Training progress and metrics will be logged to the terminal. Checkpoints are saved to the `output_dir` you configured in `params.yaml`.

---

### Step 5 — Run Inference

Once training is complete, load your fine-tuned model and chat with it:

```bash
make infer
```

You'll be prompted to enter a message in the terminal. The fine-tuned model will respond — this is your moment. 🎉

---

## 📁 Project Structure

```
lora-finetune-pipeline/
├── Makefile                        # One-line commands for setup, finetune, and infer
├── main.py                         # Entry point
├── pyproject.toml                  # Python dependencies (managed by uv)
└── src/
    ├── config/
    │   ├── params.yaml             # Model name, LoRA & training hyperparameters
    │   └── prompts.yaml            # System/user/assistant prompts + dataset generation prompt
    ├── finetune/
    │   └── data/
    │       └── custom_dataset.json # ← Paste your generated dataset here
    ├── inference/
    │   └── infer.py                # Loads the fine-tuned model and runs terminal chat
    └── utils/
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

## 🔑 Environment Setup — HuggingFace Token

Some models on the Hugging Face Hub (e.g. gated models like Llama) require authentication. Create a `.env` file in the project root before running anything:

```bash
cp .env.example .env
```

Then open `.env` and replace the placeholder with your actual token:

```
HF_TOKEN=hf_your_token_here
```

You can generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Make sure it has **read** access at minimum.

> `.env` is git-ignored — never commit your token.

---

## 📊 Project Status

| Area | Status |
|---|---|
| Project structure & layout | ✅ Done |
| Dependency manifest (`pyproject.toml`) | ✅ Done |
| Config files (`params.yaml`, `prompts.yaml`) | ✅ Boilerplate in place |
| Dataset scaffold (`custom_dataset.json`) | ✅ Ready for your data |
| Inference script (`infer.py`) | 🔧 Drafted — implementation pending |
| Fine-tuning pipeline | 🔲 Not started |
| Makefile targets | 🔲 Not started |
| README | ✅ Done |

---

## 🛣️ Next Steps

- [ ] **Write the fine-tuning pipeline** — implement `src/finetune/` using `SFTTrainer` (TRL) with QLoRA config via PEFT + bitsandbytes 4-bit quantization, wired up to `params.yaml`
- [ ] **Complete the inference script** — finish `src/inference/infer.py` to load the LoRA-adapted model and run interactive terminal chat
- [ ] **Wire up the Makefile** — add `setup`, `finetune`, and `infer` targets
- [ ] **Fill in `prompts.yaml`** — write the `dataset_generation_prompt` that workshop attendees will paste into their AI provider
- [ ] **Fill in `params.yaml`** — set sensible defaults for the workshop model and hyperparameters

---

## 🤝 Acknowledgements

Built with ❤️ using the Hugging Face ecosystem — [`transformers`](https://github.com/huggingface/transformers), [`peft`](https://github.com/huggingface/peft), [`trl`](https://github.com/huggingface/trl), and [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes).
