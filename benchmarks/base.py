from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from core.prompts import compose_prompt, sha256_text


def prompt_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


class Benchmark:
    name: str = ""
    split: str = "test"
    id_field: str = "id"
    image_field: str = "image"
    inference_fields: tuple[str, ...] = ("id", "question", "image")
    record_fields: tuple[str, ...] = ()
    required_columns: tuple[str, ...] = ()
    expected_total: int | None = None
    max_new_tokens: int = 4096
    TASK_PROMPT_TEMPLATE: str = ""

    def build_task_prompt(self, row: dict[str, Any]) -> str:
        values = {
            key: prompt_text(value)
            for key, value in row.items()
            if key != self.image_field
        }
        return self.TASK_PROMPT_TEMPLATE.format(**values)

    def task_prompt_signature(self) -> str:
        return self.TASK_PROMPT_TEMPLATE

    def task_prompt_sha256(self) -> str:
        return sha256_text(self.task_prompt_signature())

    def build_prompt(self, row: dict[str, Any]) -> str:
        return compose_prompt(self.build_task_prompt(row))

    def prepare_dataset(self, dataset: Any) -> Any:
        return dataset

    def normalize_sample_id(self, value: Any) -> str:
        return str(value).strip()

    def load_dataset(self, path: Path) -> tuple[Any, dict[str, Any]]:
        from datasets import Image as HFImage

        dataset, details = load_local_dataset(path, self.split)
        if self.image_field in dataset.column_names and not isinstance(
            dataset.features[self.image_field], HFImage
        ):
            dataset = dataset.cast_column(self.image_field, HFImage(decode=True))
        dataset = self.prepare_dataset(dataset)

        required = set(self.inference_fields) | set(self.required_columns)
        missing = sorted(required.difference(dataset.column_names))
        if missing:
            raise ValueError(f"{self.name} is missing columns: {', '.join(missing)}")
        if len(dataset) == 0:
            raise ValueError(f"{self.name} contains no rows")
        if self.expected_total is not None and len(dataset) != self.expected_total:
            raise ValueError(
                f"Expected {self.expected_total} {self.name} {self.split} rows, "
                f"found {len(dataset)}"
            )
        sample_ids = [str(value) for value in dataset[self.id_field]]
        duplicates = [key for key, count in Counter(sample_ids).items() if count > 1]
        if duplicates:
            raise ValueError(
                f"{self.name} contains duplicate {self.id_field}(s): {duplicates[:10]}"
            )

        details.update(
            {
                "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
                "columns": list(dataset.column_names),
                "rows": len(dataset),
            }
        )
        return dataset, details


def _select_split(dataset: Any, split: str) -> Any:
    from datasets import DatasetDict

    if isinstance(dataset, DatasetDict):
        if split not in dataset:
            raise KeyError(f"Split {split!r} not found; available splits: {list(dataset)}")
        return dataset[split]
    return dataset


def load_local_dataset(path: Path, split: str) -> tuple[Any, dict[str, Any]]:
    from datasets import DownloadConfig, load_dataset, load_from_disk

    offline_download_config = DownloadConfig(local_files_only=True)

    if not path.exists():
        raise FileNotFoundError(f"Dataset path not found: {path}")

    source_files: list[Path] = []
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Unsupported dataset file: {path}")
        source_files = [path]
        strategy = "parquet_file"
        dataset = load_dataset(
            "parquet",
            data_files={split: str(path)},
            split=split,
            download_config=offline_download_config,
        )
    else:
        split_parquets = sorted(path.rglob(f"{split}-*.parquet"))
        source_files = split_parquets or sorted(path.rglob("*.parquet"))
        if source_files:
            strategy = "parquet_directory"
            dataset = load_dataset(
                "parquet",
                data_files={split: [str(item) for item in source_files]},
                split=split,
                download_config=offline_download_config,
            )
        else:
            try:
                dataset = _select_split(load_from_disk(str(path)), split)
                strategy = "datasets_load_from_disk"
            except Exception as load_from_disk_error:
                try:
                    dataset = load_dataset(
                        str(path),
                        split=split,
                        download_config=offline_download_config,
                    )
                    strategy = "local_dataset_script"
                except Exception as local_dataset_error:
                    raise RuntimeError(
                        f"Could not load a dataset from {path}. load_from_disk "
                        f"failed with {type(load_from_disk_error).__name__}: "
                        f"{load_from_disk_error}; local dataset loading failed with "
                        f"{type(local_dataset_error).__name__}: {local_dataset_error}"
                    ) from local_dataset_error

    details = {
        "strategy": strategy,
        "source_files": [str(item.resolve()) for item in source_files],
    }
    return dataset, details
