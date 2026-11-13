# Adding a dataset

A new dataset needs three changes:

| File | Change |
|---|---|
| `benchmarks/<name>.py` | New file: columns, task prompt, optional preprocessing |
| `main.py` | One new line in `DATASET_PATHS`: `"<name>": "<path>"` |
| `eva_<name>.py` | New evaluation script in the project root |

Nothing in `core/`, `benchmarks/base.py` or the other datasets needs to change. The z-score statistics in `baselines/statistics/` depend on the model only, so they need no change either.

If the data is not in a supported format (see [Supported data](#supported-data)), convert it to parquet first.

## How a dataset is used

```
python main.py --dataset <name> --z-threshold ...
  -> core/inference.py: run()
       benchmarks/<name>.py: load_dataset() -> prepare_dataset() -> column / row-count / ID checks
       keep only inference_fields (answer columns are removed before inference)
       per sample: prompt = RELAY_INSTRUCTION + "\n\n" + build_task_prompt(row)
       image written to a temporary PNG -> relay (core/relay.py)
       one JSONL record -> results/<name>/responses_*.jsonl

python eva_<name>.py
  -> read the JSONL -> join with the local ground truth by ID -> accuracy -> shared report
```

A file in `benchmarks/` that defines `BENCHMARK = <Class>()` is registered automatically. Files whose names start with `_` (such as `_template.py`) are skipped.

## `benchmarks/<name>.py`

Start from `benchmarks/_template.py` and set these class attributes:

| Attribute | Meaning | Rules |
|---|---|---|
| `name` | Dataset name | Lower case, digits and underscores. Must equal the `DATASET_PATHS` key and `DATASET` in `eva_<name>.py`. |
| `split` | Split name | For a directory, files named `{split}-*.parquet` are used |
| `id_field` | Unique sample ID column | Unique and non-empty. If the data has no ID column, create one in `prepare_dataset` (for example from the row index, as `hallusionbench.py` does). |
| `image_field` | Image column | Exactly one image per sample |
| `inference_fields` | Columns visible during inference | Must include `id_field`, `image_field` and every column the prompt uses. Must not include answers or anything else derived from the ground truth. |
| `record_fields` | Extra columns copied into each result record | A subset of `inference_fields`, usually the question |
| `required_columns` | Columns that are only checked for presence | Usually the answer column |
| `expected_total` | Number of samples | Checked at load time; `None` skips the check |
| `max_new_tokens` | Default generation budget | The bundled datasets use 2048 |
| `TASK_PROMPT_TEMPLATE` | Task prompt template | See [Prompt rules](#prompt-rules) |

The file must end with `BENCHMARK = <Class>()`.

### Prompt rules

1. **Write only the dataset's own task prompt**: the task, the reasoning request and the final-answer format. Do not mention the relay, drafts, bidding or other models; `core/prompts.py` adds the relay instruction automatically.
2. **Formatting.**
   - The default rendering is `TASK_PROMPT_TEMPLATE.format(**row)`, without the image column. `None` becomes `""`, and other non-string values go through `str()`.
   - A list column such as options would render as `['a', 'b']`. In that case, override `build_task_prompt` and format it yourself, for example as `A. a\nB. b`.
   - Write literal braces as `{{ }}`.
3. **Require an explicit final-answer line** such as `Answer: <option letter>`, and keep it consistent with the extraction rule in `eva_<name>.py`.
4. **Do not write a literal `<image>`.** InternVL treats it as the image placeholder, and the image tokens are inserted automatically.
5. **Branching prompts.** If the prompt depends on the question type (see `mathvista.py`), override both `build_task_prompt(row)` and `task_prompt_signature()`. The signature must include every text constant the branches use, so that a changed prompt changes the prompt hash and blocks resuming old files.
6. **Do not override `build_prompt`.**

### Optional hooks

| Method | Override when |
|---|---|
| `prepare_dataset(self, dataset)` | You need to validate rows, add an ID column, or rename or drop columns. It receives and returns a Hugging Face `datasets.Dataset`. |
| `normalize_sample_id(self, value)` | `--sample-ids` values need normalization, for example turning `"007"` into `"7"` for integer IDs (see `hallusionbench.py`) |
| `build_task_prompt` / `task_prompt_signature` | The prompt branches or needs custom formatting |

### Supported data

- **Formats.** All of these are loaded offline:
  - a single `.parquet` file;
  - a directory with `{split}-*.parquet`, or every `*.parquet` file if none match;
  - a directory written by `datasets.save_to_disk`;
  - a local dataset script.

  Anything else, such as JSON plus an image folder, must be converted to parquet first. Store images as `{"bytes": ...}` or as file paths.
- **Image columns.** A Hugging Face `Image` feature, a `{"bytes"/"path"}` dictionary, or a path string. Columns that are not an `Image` feature are cast automatically.
- **One image per sample.** Samples without an image, or with several, need their own design first.
- **Column name clashes.** Result records use these field names: `schema_version`, `status`, `eval_prompt`, `eval_prompt_sha256`, `response`, `error`, `experiment`, `started_at`, `image_meta`, `relay_stats`, `usage`, `ori_input_token`, `cache_stats`, `timing`, `gpu`, `recorded_at`. A dataset column with one of these names that is not in `inference_fields` triggers the ground-truth leak check on the first sample. Rename or drop such columns in `prepare_dataset`.

### Example

```python
from __future__ import annotations

from .base import Benchmark


class Demo(Benchmark):
    name = "demo"
    split = "test"
    id_field = "qid"
    image_field = "image"
    inference_fields = ("qid", "question", "options", "image")
    record_fields = ("question",)
    required_columns = ("answer",)
    expected_total = 1000
    max_new_tokens = 2048

    TASK_PROMPT_TEMPLATE = (
        "Carefully analyze the image and answer the multiple-choice question step by step.\n\n"
        "Show your reasoning before giving the final answer.\n"
        "End your response with exactly one final line in the form: Answer: <option letter>\n\n"
        "Question:\n"
        "{question}\n"
        "Options:\n"
        "{options}"
    )


BENCHMARK = Demo()
```

## `eva_<name>.py`

### Conventions

1. **File header.** The file starts with `from __future__ import annotations` followed by three constants with exactly these names:
   - `INPUT_FILES`: result files; an empty list means `results/<name>/*.jsonl`;
   - `MAX_EVAL = None`: evaluate only the first N records of each file;
   - `VERBOSE = False`: print extra details.
2. **Ground truth.**
   - It comes only from the local dataset, at the path returned by `eu.dataset_path(DATASET)`.
   - Do not load the image column.
   - Result files never contain answers, and the script must not read answers from them.
3. **Reading results.** Use `eu.read_jsonl_snapshot(path, ID_ALIASES)`, where the first element of `ID_ALIASES` is the `id_field`. Then immediately apply `eu.first_n(records, MAX_EVAL)`.
4. **Accuracy.** Accuracy is correct answers divided by evaluated records. A record is evaluated when its ID exists in the ground truth.
   - These count as wrong and stay in the denominator: records with `status != "success"`, records without an extractable answer, and invalid questions.
   - Records with unknown IDs are skipped and reported through `note`.
5. **Extraction.**
   - Prefer the last explicit `Answer:` line.
   - Return `None` when nothing can be extracted; that counts as wrong.
   - Avoid lenient matching such as treating an empty string as a substring of the answer. `eva_mmstar.py` keeps such legacy logic for compatibility; do not copy it.
6. **Default output.**
   - Print it only through `eu.print_basic_report(...)`: accuracy, prompt tokens, generated tokens (attempted and committed) and latency.
   - Grouped accuracies and diagnostics belong behind `VERBOSE`.
7. **Several files.**
   - After each file, append `(path, {"ACC (%)": ..., **eu.basic_overall_metrics(basic)})` to a list, and finish with `eu.print_overall(list)`.
   - If the dataset has several headline metrics (as HallusionBench has aAcc / qAcc / fAcc), put all of them in `acc_line` and in that dictionary.
8. **Dependencies.** Only the standard library, `pyarrow` and `datasets` (offline). Never import torch.

### `core/eval_utils` (`from core import eval_utils as eu`)

| Function | Purpose |
|---|---|
| `eu.dataset_path(name)` | `DATASET_PATHS[name]` from `main.py`; raises if it is empty |
| `eu.resolve_input_files(INPUT_FILES, name)` | `INPUT_FILES` resolved against the project root, or `results/<name>/*.jsonl` when empty |
| `eu.read_jsonl_snapshot(path, id_aliases)` | Complete records keyed by `str(ID)`. An unterminated last line is ignored, and for repeated IDs the last record wins. |
| `eu.first_n(records, MAX_EVAL)` | The first N records (`None` = all) |
| `eu.response_text(record)` | The response text, accepting `response`, `pred`, `model_output` or `output` |
| `eu.summarize_basic(records)` | Prompt tokens, generated tokens and latency |
| `eu.format_accuracy(correct, total)` | For example `"61.83% (277/448)"` |
| `eu.print_basic_report(path, evaluated, expected, acc_line, basic, note=None)` | The shared default report |
| `eu.basic_overall_metrics(basic)` | Metrics for the multi-file summary |
| `eu.print_overall(results)` | Mean and standard deviation when more than one file was evaluated |

### Example

```python
from __future__ import annotations

INPUT_FILES: list = []
MAX_EVAL = None
VERBOSE = False

import re
from collections import OrderedDict
from pathlib import Path
from typing import Any

from core import eval_utils as eu

DATASET = "demo"
EXPECTED_TOTAL = 1000
ID_ALIASES = ("qid",)

ANSWER_RE = re.compile(r"(?im)^\s*\**\s*answer\s*[:：]\s*\**\s*\(?([A-Z])\)?")


def load_ground_truth(path: Path) -> "OrderedDict[str, dict[str, Any]]":
    import pyarrow.parquet as pq

    rows = pq.read_table(path, columns=["qid", "answer"]).to_pylist()
    ground_truth: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for row in rows:
        key = str(row["qid"])
        if key in ground_truth:
            raise ValueError(f"Duplicate id in ground truth: {key}")
        ground_truth[key] = row
    return ground_truth


def extract_answer(response: str) -> str | None:
    matches = ANSWER_RE.findall(response or "")
    return matches[-1].upper() if matches else None


def is_correct(prediction: str | None, gold: dict[str, Any]) -> bool:
    return prediction is not None and prediction == str(gold["answer"]).strip().upper()


def evaluate_records(records, ground_truth) -> dict[str, Any]:
    total = 0
    correct = 0
    for key, record in records.items():
        total += 1
        if record.get("status", "success") != "success":
            continue
        prediction = extract_answer(eu.response_text(record)[0])
        correct += int(is_correct(prediction, ground_truth[key]))
    return {"total": total, "correct": correct}


def main() -> None:
    ground_truth = load_ground_truth(eu.dataset_path(DATASET))
    processed = []
    for path in eu.resolve_input_files(INPUT_FILES, DATASET):
        try:
            records, diagnostics = eu.read_jsonl_snapshot(path, ID_ALIASES)
            records = eu.first_n(records, MAX_EVAL)
            unknown_ids = [key for key in records if key not in ground_truth]
            known = OrderedDict((k, r) for k, r in records.items() if k in ground_truth)
            result = evaluate_records(known, ground_truth)
            basic = eu.summarize_basic(known.values())
            eu.print_basic_report(
                path,
                evaluated=result["total"],
                expected=EXPECTED_TOTAL,
                acc_line=eu.format_accuracy(result["correct"], result["total"]),
                basic=basic,
                note=f"{len(unknown_ids)} record(s) with unknown id skipped" if unknown_ids else None,
            )
            if VERBOSE:
                print(f"[VERBOSE] snapshot diagnostics: {diagnostics}")
            accuracy = result["correct"] / result["total"] * 100 if result["total"] else 0.0
            processed.append((path, {"ACC (%)": accuracy, **eu.basic_overall_metrics(basic)}))
        except FileNotFoundError as error:
            print(f"\nError: {error}. Skipping.")
        except Exception as error:
            print(f"\nError while processing {path}: {type(error).__name__}: {error}")
    if not processed:
        raise RuntimeError("No input files were processed successfully")
    eu.print_overall(processed)


if __name__ == "__main__":
    main()
```

If the ground truth is not a single parquet file (for example a directory or a `save_to_disk` dataset), load it with `datasets` in offline mode and drop the image column; see `load_ground_truth` in `eva_logicvista.py`.

## Self-checks (no GPU needed)

Run these from the project root.

1. **Syntax.**

```bash
python3 -m py_compile benchmarks/<name>.py eva_<name>.py main.py
```

2. **Registration.** `--dataset {...}` in the help output must list `<name>`.

```bash
python3 main.py --help
```

3. **Prompt rendering.** Render a prompt with placeholder values. It should start with the relay instruction followed by the task prompt, contain no leftover braces and no `None`, and show every placeholder where expected. For branching prompts, render one row of each type.

```bash
python3 -c "from benchmarks import REGISTRY; b = REGISTRY['<name>']; row = {f: f'<{f}>' for f in b.inference_fields if f != b.image_field}; print(b.build_prompt(row)); print(b.task_prompt_sha256())"
```

4. **No relay wording in the dataset file.** This must print nothing.

```bash
grep -niE "relay|draft|bidding" benchmarks/<name>.py
```

5. **Evaluation on synthetic records.** Include a correct answer, a wrong answer, a record with `status=error`, a record without an answer and an unknown ID. Check that the numerator and denominator follow the accuracy rules, and that `MAX_EVAL=1` gives `n=1`.

```bash
python3 - <<'EOF'
import json, tempfile
from pathlib import Path
import eva_<name> as eva
records = [
    {"<id_field>": "1", "status": "success", "response": "...\nAnswer: A"},
    {"<id_field>": "2", "status": "success", "response": "Answer: C"},
    {"<id_field>": "3", "status": "error", "response": ""},
    {"<id_field>": "4", "status": "success", "response": "no final answer"},
    {"<id_field>": "zz", "status": "success", "response": "Answer: A"},
]
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "r.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    eva.INPUT_FILES = [str(path)]
    eva.eu.dataset_path = lambda name: Path(tmp)
    eva.load_ground_truth = lambda *args: {"1": {"answer": "A"}, "2": {"answer": "B"}, "3": {"answer": "C"}, "4": {"answer": "D"}}
    eva.main()
    eva.MAX_EVAL = 1
    eva.main()
EOF
```

The first report should show `n=4`, `ACC: 25.00% (1/4)` and a note about one unknown ID. The second should show `n=1`.

6. **Real data.** If `datasets` is installed and the data is available, check that loading passes every check:

```bash
python3 -c "from pathlib import Path; from benchmarks import REGISTRY; ds, d = REGISTRY['<name>'].load_dataset(Path('<dataset path>')); print(len(ds), ds.column_names, d['strategy'])"
```

7. **GPU smoke test.** Run three samples, then evaluate them by setting `INPUT_FILES = ["results/_smoke/<name>_n3.jsonl"]`. All three records should have `status` `success`, and the report should show accuracy, prompt tokens, generated tokens and latency.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --dataset <name> --z-threshold -0.3 --max-samples 3 --output-file results/_smoke/<name>_n3.jsonl
```

## Existing datasets to copy from

| Files | Question type | Useful for |
|---|---|---|
| `benchmarks/logicvista.py` + `eva_logicvista.py` | Single or multiple option letters (answer is a set) | Fixed prompt template; strict set comparison; invalid rows counted as wrong; grouped accuracy behind `VERBOSE` |
| `benchmarks/hallusionbench.py` + `eva_hallusionbench.py` | Yes / no | `prepare_dataset` validation and row-index IDs; composite identity fields; several headline metrics (aAcc / qAcc / fAcc) |
| `benchmarks/mathvista.py` + `eva_mathvista.py` | Multiple choice plus integer, float and list answers | Branching `build_task_prompt` with `task_prompt_signature`; normalization by answer type |
| `benchmarks/realworldqa.py` + `eva_realworldqa.py` | Option letter or short answer | Rewriting a question column in `prepare_dataset`; exact-match scoring |
| `benchmarks/mmstar.py` + `eva_mmstar.py` | Four options | Simplest structure, but do not copy its lenient matching |
