
import json
from pathlib import Path

import yaml
from datasets import Dataset, DatasetDict, load_dataset as hf_load_dataset

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "params.yaml"


class LoaderForDataset:
    def __init__(self, config_path: str | Path = _DEFAULT_CONFIG_PATH, train_ratio: float = 0.8, seed: int = 42):
        self.config_path = Path(config_path)
        self.train_ratio = train_ratio
        self.seed = seed

        self.config = self._load_config()
        self.dataset_path = self._resolve_dataset_path(self.config["dataset_path"])

        dataset_dict = self.load_dataset(self.dataset_path)
        self.train_dataset = dataset_dict["train"]
        self.val_dataset = dataset_dict["validation"]

    def _load_config(self) -> dict:
        """Load project configuration and ensure required keys are present."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        if "dataset_path" not in config:
            raise KeyError("'dataset_path' is missing in params.yaml")
        if config["dataset_path"] in (None, "", "x"):
            raise ValueError("'dataset_path' in params.yaml is not configured.")

        return config

    def _resolve_dataset_path(self, dataset_path: str) -> Path:
        """Resolve dataset path from params.yaml. Relative paths are resolved from project root."""
        raw_path = Path(dataset_path)
        if raw_path.is_absolute():
            resolved = raw_path
        else:
            project_root = Path(__file__).resolve().parents[2]
            resolved = project_root / raw_path

        if not resolved.exists():
            raise FileNotFoundError(f"Dataset file not found: {resolved}")

        return resolved

    def load_dataset(self, dataset_path: str | Path | None = None) -> DatasetDict:
        """
        Load custom JSONL dataset with Hugging Face datasets.load_dataset.
        Falls back to manual line-by-line parsing if dataset loader fails.

        Expected JSONL format (one object per line):
        {"input_prompt": "...", "expected_response": "..."}
        """
        path = Path(dataset_path) if dataset_path is not None else self.dataset_path
        ext = path.suffix.lower()
        if ext not in {".jsonl", ".json"}:
            raise ValueError("Dataset file must be .jsonl (recommended) or .json")

        try:
            loaded = hf_load_dataset("json", data_files={"train": str(path)})
            train_dataset = loaded["train"]
        except Exception as parse_error:
            import json
            records = []
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(
                            f"Invalid JSON on line {line_num}: {e}. "
                            f"Content: {line[:100]}..."
                        ) from e
            if not records:
                raise ValueError("Dataset file is empty or contains no valid JSON lines.")
            train_dataset = Dataset.from_list(records)

        if len(train_dataset) == 0:
            raise ValueError("Dataset file is empty.")

        required_columns = {"input_prompt", "expected_response"}
        missing_columns = required_columns - set(train_dataset.column_names)
        if missing_columns:
            raise ValueError(
                f"Missing required columns: {sorted(missing_columns)}"
            )

        def _format_for_sft(sample: dict) -> dict:
            prompt = str(sample["input_prompt"]).strip()
            response = str(sample["expected_response"]).strip()
            if not prompt or not response:
                raise ValueError("Found row with empty input_prompt/expected_response.")

            return {
                "input_prompt": prompt,
                "expected_response": response,
                "text": f"### Instruction:\n{prompt}\n\n### Response:\n{response}",
            }

        formatted_train = train_dataset.map(_format_for_sft)
        return self.split_dataset(formatted_train, train_ratio=self.train_ratio)

    def split_dataset(self, dataset: Dataset, train_ratio: float = 0.8) -> DatasetDict:
        """
        Split the dataset into training and validation sets based on the specified ratio.
        """
        if not 0 < train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1.")

        split = dataset.train_test_split(test_size=1 - train_ratio, seed=self.seed, shuffle=True)
        return DatasetDict({"train": split["train"], "validation": split["test"]})