import os
from pathlib import Path

import torch
import yaml
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "params.yaml"


class LLM_Model:
    def __init__(self):
        """
        Initialize LLM Model (expose self.model, self.processor)
        """
        load_dotenv()

        with open(_CONFIG_PATH, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_name = self.config["model_name"]
        self.hf_token = os.getenv("HF_TOKEN")

        device = self._check_gpu_availability()
        self.model, self.processor = self._load_model(device)

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

    def _load_model(self, device):
        """
        Load Model using Hugging Transformers Library to GPU
        """
        print(f"Loading model: {self.model_name}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=self.hf_token,
            device_map=device,
            dtype=torch.float16 if device != "cpu" else torch.float32,
        )

        print(f"Model loaded successfully on {device}")
        return model, tokenizer

    def _quantize_model(self, model):
        """
        Quantize Model to either 4 or 8 Bits
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        return bnb_config

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
    print(model.generate_response("What is the capital of France?"))
    print(model.get_model_and_processor())


if __name__ == "__main__":
    main()