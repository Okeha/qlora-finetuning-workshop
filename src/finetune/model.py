import os
import gc
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "params.yaml"


class LLM_Model:
    def __init__(self, config_path: str | Path = _CONFIG_PATH):
        """
        Shared model manager for inference and QLoRA finetuning.
        """
        load_dotenv()
        self.config_path = Path(config_path)

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.model_name = self.config["model_name"]
        self.hf_token = os.getenv("HF_TOKEN")
        self.device = self._check_gpu_availability()
        self.model = None
        self.processor = None

    def _check_gpu_availability(self):
        """
        Check GPU availability
        """
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"Using device: {device}")
        return device

    def _build_quant_config(self):
        compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

    def _build_lora_config(self) -> LoraConfig:
        lora_cfg = self.config.get("lora_hyperparameters", {})
        target_modules = lora_cfg.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        )
        if isinstance(target_modules, str):
            target_modules = [target_modules]

        return LoraConfig(
            r=int(lora_cfg.get("rank", 8)),
            lora_alpha=int(lora_cfg.get("alpha", 16)),
            lora_dropout=float(lora_cfg.get("dropout", 0.1)),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_for_inference(self, quantized: bool = False):
        """
        Load base model + tokenizer for inference.
        """
        print(f"Loading model: {self.model_name}")
        tokenizer = self._load_tokenizer()

        model_kwargs = {
            "token": self.hf_token,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if quantized:
            model_kwargs["quantization_config"] = self._build_quant_config()
        else:
            model_kwargs["dtype"] = torch.float16 if self.device != "cpu" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.model, self.processor = model, tokenizer

        print(f"Model loaded successfully on {self.device}")
        return self.model, self.processor

    def load_for_finetune(self):
        """
        Load 4-bit quantized base model + tokenizer for QLoRA finetuning.
        """
        print(f"Loading quantized model for finetuning: {self.model_name}")
        tokenizer = self._load_tokenizer()

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.hf_token,
            quantization_config=self._build_quant_config(),
            device_map="auto",
            trust_remote_code=True,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(model)

        training_cfg = self.config.get("training", {})
        if bool(training_cfg.get("gradient_checkpointing", True)):
            model.gradient_checkpointing_enable()

        self.model, self.processor = model, tokenizer
        self.cleanup_gpu()
        return self.model, self.processor

    def attach_lora_adapters(self, model=None):
        """
        Attach LoRA adapters to a prepared base model.
        """
        if model is None:
            if self.model is None:
                raise ValueError("No base model loaded. Call load_for_finetune first.")
            model = self.model

        peft_model = get_peft_model(model, self._build_lora_config())
        peft_model.print_trainable_parameters()
        self.model = peft_model
        self.cleanup_gpu()
        return peft_model

    @staticmethod
    def cleanup_gpu():
        """
        Release unreferenced tensors and clear CUDA allocator caches.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def generate_response(self, prompt: str, max_new_tokens: int = 512):
        """
        Generate Clean Model Response
        """
        print(f"\n\n Generating response for prompt: {prompt}")
        inputs = self.processor(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = self.processor.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response

    def get_model_and_processor(self):
        """
        Get loaded model and processor and all related properties
        """
        return {
            "model": self.model,
            "processor": self.processor,
            "model_name": self.model_name,
            "device": str(next(self.model.parameters()).device),
        }


def main():
    """
    Main function to run the model
    """
    model = LLM_Model()
    model.load_for_inference(quantized=False)
    print(model.generate_response("What is the capital of France?"))
    # print(model.get_model_and_processor())


if __name__ == "__main__":
    main()