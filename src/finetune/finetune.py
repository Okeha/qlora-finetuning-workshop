# Usage:
# Run the fine-tuning pipeline using the provided dataset and configuration.
# The script will handle the entire fine-tuning process, including:
# - Loading the base model and tokenizer.
# - Preparing the dataset for training.
# - Configuring the training parameters.
# - Executing the training loop.
# - Saving the fine-tuned model for later use in inference.
# - Configuring logging and monitoring for the training process.

import logging
import os
import time
import traceback
from pathlib import Path

import torch
import yaml
from rich import box
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table
from trl import SFTConfig, SFTTrainer

from src.finetune.dataloader import LoaderForDataset
from src.finetune.model import LLM_Model

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "params.yaml"


def _setup_logger() -> tuple[logging.Logger, Console]:
    console = Console()
    logger = logging.getLogger("qlora.finetune")

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        rich_handler = RichHandler(console=console, markup=True, show_path=False)
        formatter = logging.Formatter("%(message)s")
        rich_handler.setFormatter(formatter)
        logger.addHandler(rich_handler)
        logger.propagate = False

    return logger, console


class QLoraFinetuneLLM:
    def __init__(self, config_path: str | Path = _CONFIG_PATH):
        self.config_path = Path(config_path)
        self.model_manager = LLM_Model(config_path=self.config_path)
        self.logger, self.console = _setup_logger()

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f) or {}

    def _load_data(self):
        training_cfg = self.config.get("training", {})
        loader = LoaderForDataset(
            config_path=self.config_path,
            train_ratio=float(training_cfg.get("train_ratio", 0.9)),
            seed=int(training_cfg.get("seed", 42)),
        )
        return loader.train_dataset, loader.val_dataset

    def _build_sft_config(self) -> SFTConfig:
        training_cfg = self.config.get("training", {})

        output_dir = training_cfg.get("output_dir", "outputs/qlora")
        logging_dir = training_cfg.get("logging_dir", f"{output_dir}/runs")
        max_length = int(training_cfg.get("max_length", training_cfg.get("max_seq_length", 1024)))
        os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(logging_dir))

        return SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=int(training_cfg.get("batch_size", 2)),
            per_device_eval_batch_size=int(training_cfg.get("eval_batch_size", 2)),
            gradient_accumulation_steps=int(training_cfg.get("gradient_accumulation_steps", 8)),
            num_train_epochs=float(training_cfg.get("num_epochs", 6)),
            learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
            weight_decay=float(training_cfg.get("weight_decay", 0.01)),
            warmup_steps=int(training_cfg.get("warmup_steps", 5)),
            lr_scheduler_type=str(training_cfg.get("lr_scheduler", "cosine")),
            max_length=max_length,
            logging_steps=int(training_cfg.get("logging_steps", 1)),
            eval_steps=int(training_cfg.get("eval_steps", 5)),
            save_steps=int(training_cfg.get("save_steps", 20)),
            save_total_limit=int(training_cfg.get("save_total_limit", 2)),
            bf16=bool(training_cfg.get("bf16", torch.cuda.is_available())),
            fp16=bool(training_cfg.get("fp16", False)),
            report_to=["tensorboard"],
            packing=bool(training_cfg.get("packing", False)),
            dataset_text_field="text",
            eval_strategy="steps",
            save_strategy="steps",
            optim=str(training_cfg.get("optimizer", "paged_adamw_8bit")),
            seed=int(training_cfg.get("seed", 42)),
        )

    def _show_run_summary(self, train_dataset, eval_dataset, sft_config: SFTConfig) -> None:
        summary_table = Table(title="QLoRA Training Run Summary", box=box.SIMPLE_HEAVY)
        summary_table.add_column("Setting", style="cyan", no_wrap=True)
        summary_table.add_column("Value", style="magenta")

        summary_table.add_row("Model", self.config.get("model_name", "<missing>"))
        summary_table.add_row("Train Samples", str(len(train_dataset)))
        summary_table.add_row("Eval Samples", str(len(eval_dataset)))
        summary_table.add_row("Batch Size", str(sft_config.per_device_train_batch_size))
        summary_table.add_row("Grad Accum", str(sft_config.gradient_accumulation_steps))
        summary_table.add_row("Epochs", str(sft_config.num_train_epochs))
        summary_table.add_row("Learning Rate", str(sft_config.learning_rate))
        summary_table.add_row("Optimizer", str(sft_config.optim))
        summary_table.add_row("Output Dir", str(sft_config.output_dir))
        summary_table.add_row("TensorBoard", str(sft_config.logging_dir))

        self.console.print(summary_table)

    def train(self):
        start_time = time.time()
        self.console.print(Panel.fit("[bold green]Starting QLoRA Finetuning[/bold green]", border_style="green"))

        trainer = None
        model = None
        try:
            self.logger.info("[bold cyan]Step 1/5[/bold cyan] Loading dataset splits")
            train_dataset, eval_dataset = self._load_data()

            self.logger.info("[bold cyan]Step 2/5[/bold cyan] Building TRL SFT configuration")
            sft_config = self._build_sft_config()
            self._show_run_summary(train_dataset, eval_dataset, sft_config)

            self.logger.info("[bold cyan]Step 3/5[/bold cyan] Loading 4-bit quantized base model")
            base_model, tokenizer = self.model_manager.load_for_finetune()

            self.logger.info("[bold cyan]Step 4/5[/bold cyan] Attaching LoRA adapters")
            model = self.model_manager.attach_lora_adapters(base_model)

            self.logger.info("[bold cyan]Step 5/5[/bold cyan] Initializing SFTTrainer and starting training")
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
            )

            trainer.train()

            final_model_path = self.config.get("final_model_path", "outputs/qlora/final")
            self.logger.info(f"Saving finetuned adapter and tokenizer to: {final_model_path}")
            trainer.model.save_pretrained(final_model_path)
            tokenizer.save_pretrained(final_model_path)

            elapsed = time.time() - start_time
            self.console.print(
                Panel.fit(
                    f"[bold green]Training complete[/bold green]\nDuration: [cyan]{elapsed:.2f}s[/cyan]",
                    border_style="green",
                )
            )
        except Exception:
            self.console.print(Panel.fit("[bold red]Training failed[/bold red]", border_style="red"))
            self.logger.error(traceback.format_exc())
            raise
        finally:
            if trainer is not None:
                del trainer
            if model is not None:
                del model
            self.model_manager.cleanup_gpu()
            self.logger.info("GPU cleanup completed.")


def main():
    pipeline = QLoraFinetuneLLM()
    pipeline.train()


if __name__ == "__main__":
    main()



