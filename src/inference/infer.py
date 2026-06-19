# Simple Script to Run Inference using the fine-tuned model

# Usage:
# Load QLoRA Finetuned Model.
# Accept Input Via Terminal (prompt for user input).
# Generate output and print to terminal.

import os
import warnings
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from peft import PeftModel
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "params.yaml"


def _quiet_runtime_noise() -> None:
    # Suppress non-actionable framework warnings that clutter interactive chat.
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"bitsandbytes.*")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")


class InferResponse:
    def __init__(self, config_path: str | Path = _CONFIG_PATH):
        """
        Initialize Model Response
        """
        _quiet_runtime_noise()
        load_dotenv()
        self.console = Console()
        self.config_path = Path(config_path)

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

        self.model_name = self.config["model_name"]
        self.final_model_path = self._resolve_final_model_path(self.config["final_model_path"])
        self.hf_token = os.getenv("HF_TOKEN")
        self.inference_cfg = self.config.get("inference", {})

        with self.console.status("[bold cyan]Loading base model and LoRA adapter...[/bold cyan]"):
            self.model, self.tokenizer = self._load_model_with_lora_config()

        self.console.print(
            Panel.fit(
                f"[bold green]Inference Ready[/bold green]\n"
                f"Model: [cyan]{self.model_name}[/cyan]\n"
                f"Adapter: [magenta]{self.final_model_path}[/magenta]",
                border_style="green",
            )
        )

    def _resolve_final_model_path(self, model_path: str) -> Path:
        raw_path = Path(model_path)
        if raw_path.is_absolute():
            resolved = raw_path
        else:
            project_root = Path(__file__).resolve().parents[2]
            resolved = project_root / raw_path

        if not resolved.exists():
            raise FileNotFoundError(f"Fine-tuned adapter path not found: {resolved}")

        return resolved

    def _build_quant_config(self):
        if not torch.cuda.is_available():
            return None

        compute_dtype = torch.bfloat16
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    
    def _load_model_with_lora_config(self):
        """
        Load Model with saved QLoRA tensors and weights
        """
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.final_model_path),
            token=self.hf_token,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {
            "token": self.hf_token,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        quant_config = self._build_quant_config()
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
        else:
            model_kwargs["dtype"] = torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs,
        )

        model = PeftModel.from_pretrained(
            base_model,
            str(self.final_model_path),
            is_trainable=False,
        )
        model.eval()
        return model, tokenizer
    
    def generate_response(self, prompt: str, max_new_tokens: int = 512):
        """
        Take user input from terminal and Generate Clean Model Response
        """
        temperature = float(self.inference_cfg.get("temperature", 0.7))
        top_p = float(self.inference_cfg.get("top_p", 0.9))
        repetition_penalty = float(self.inference_cfg.get("repetition_penalty", 1.1))
        max_new_tokens = int(self.inference_cfg.get("max_new_tokens", max_new_tokens))

        formatted_prompt = f"### Insutruction:\n{prompt}\n### Response:\n"
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def run_cli(self):
        self.console.print(
            Panel.fit(
                "[bold]QLoRA Chat[/bold]\nType [cyan]exit[/cyan] to quit.",
                border_style="blue",
            )
        )
        while True:
            prompt = self.console.input("\n[bold cyan]You[/bold cyan]: ").strip()
            if prompt.lower() in {"exit", "quit"}:
                self.console.print("[bold yellow]Goodbye.[/bold yellow]")
                break
            if not prompt:
                continue

            with self.console.status("[bold green]Thinking...[/bold green]"):
                response = self.generate_response(prompt)

            self.console.print(
                Panel(
                    Text(response, style="white"),
                    title="Assistant",
                    border_style="green",
                )
            )


def main():
    inference = InferResponse()
    inference.run_cli()


if __name__ == "__main__":
    main()




