# Multi-Agent Relay Thinking

The official implementation of "Multi-Agent Relay Thinking"

## Abstract

Existing multi-agent collaboration (MAC) methods mainly follow a group-thinking topology: agents generate solutions independently and their outputs are then aggregated. This work studies an alternative relay-thinking topology, in which ownership of the reasoning is transferred dynamically between agents, so each new agent inherits and refines the reasoning trajectory built so far. We introduce Multi-Agent Confidence-guided Relay (MACR), a framework for dynamic relay collaboration among heterogeneous vision-language models. The key challenge is deciding when the reasoning should be relayed and which agent should continue it. MACR first calibrates sliding-window confidence across models, so that their scores become comparable. It then relays when the leading model's calibrated confidence drops below a threshold, and selects the next agent through a confidence-based bidding round. On four benchmarks, MACR outperforms the strongest single-model strategies in accuracy (+3.3% on MathVista, +3.9% on MMStar). It remains on par with multi-agent strategies while using 6.0× fewer tokens on HallusionBench and MMStar.

---

## Installation

The code needs Linux with CUDA GPUs. Each model takes about 17 GB of GPU memory, and one GPU per model is recommended. FlashAttention-2 is required. Run every command in this README from the repository root.

1. **Create and activate an environment**

   The code requires Python 3.9 or newer.

   ```bash
   conda create -n macr python=3.10 -y
   conda activate macr
   ```

2. **Install dependencies**

   The setup was tested with torch 2.8 and flash-attn 2.8.3. `flash-attn` must be built against the installed torch, so install it after the other packages.

   ```bash
   pip install -r requirements.txt
   pip install flash-attn --no-build-isolation
   ```

---

## 1. Reproducing Results: MACR on MathVista, LogicVista, MMStar and HallusionBench

**Note:** Use `main.py` with window size `w = 16` and threshold `τ = -0.3`. The model pool is Qwen3-VL-8B-Instruct, InternVL3.5-8B and MiniCPM-V-4.5, with each model running in its own worker process on its own GPU.

### Step 1: Download Models

Download the three checkpoints from Hugging Face. The `hf` command comes with `huggingface_hub`, which `transformers` installs.

| Model | URL | Destination |
|---|---|---|
| Qwen3-VL-8B-Instruct | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct | `models/Qwen3-VL-8B-Instruct` |
| InternVL3.5-8B | https://huggingface.co/OpenGVLab/InternVL3_5-8B | `models/InternVL3_5-8B` |
| MiniCPM-V-4.5 | https://huggingface.co/openbmb/MiniCPM-V-4_5 | `models/MiniCPM-V-4_5` |

```bash
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct
hf download OpenGVLab/InternVL3_5-8B --local-dir models/InternVL3_5-8B
hf download openbmb/MiniCPM-V-4_5 --local-dir models/MiniCPM-V-4_5
```

### Step 2: Download Datasets

Download the four benchmarks from Hugging Face. Only the parquet files are needed.

| Benchmark | URL | Split (samples) | Destination |
|---|---|---|---|
| MathVista | https://huggingface.co/datasets/AI4Math/MathVista | `testmini` (1,000) | `data/MathVista` |
| LogicVista | https://huggingface.co/datasets/lscpku/LogicVista | `test` (448) | `data/LogicVista` |
| MMStar | https://huggingface.co/datasets/Lin-Chen/MMStar | `val` (1,500) | `data/MMStar` |
| HallusionBench | https://huggingface.co/datasets/lmms-lab/HallusionBench | `image` (951) | `data/HallusionBench` |

```bash
hf download AI4Math/MathVista --repo-type dataset --include "data/testmini-*.parquet" --local-dir data/MathVista
hf download lscpku/LogicVista --repo-type dataset --include "data/test-*.parquet" --local-dir data/LogicVista
hf download Lin-Chen/MMStar mmstar.parquet --repo-type dataset --local-dir data/MMStar
hf download lmms-lab/HallusionBench --repo-type dataset --include "data/image-*.parquet" --local-dir data/HallusionBench
```

No separate preprocessing step is needed. The loaders read the parquet files directly, offline, and check the columns, the number of rows and the sample IDs. For HallusionBench, sample IDs (`pid`) are created from row order at load time. Answer columns are removed before inference starts, so the models never see them.

### Step 3: Configure Paths

* **File:** the configuration block at the top of `main.py`. The evaluation scripts read the same block.


```python
MODEL_PATHS = {
    "Qwen3-VL": "models/Qwen3-VL-8B-Instruct",
    "InternVL": "models/InternVL3_5-8B",
    "MiniCPM-V": "models/MiniCPM-V-4_5",
}
DATASET_PATHS = {
    "mmstar": "data/MMStar/mmstar.parquet",
    "mathvista": "data/MathVista",
    "logicvista": "data/LogicVista",
    "hallusionbench": "data/HallusionBench",
    "realworldqa": "",
}
DEVICES = {"Qwen3-VL": "cuda:0", "InternVL": "cuda:1", "MiniCPM-V": "cuda:2"}
```

* **Devices:** `DEVICES` uses CUDA indices counted after `CUDA_VISIBLE_DEVICES` is applied.
* **Statistics:** the calibration statistics in `baselines/statistics/` are used as they are and must not be removed. They hold μ and σ of the sliding-window mean log-probability for each model, at window lengths 4, 8, 16, 32 and 64, computed on 1,000 randomly sampled MMMU instances.

### Step 4: Run Inference

Run one command per benchmark. Each run starts three worker processes, one per model, and writes one JSON line per sample.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --dataset mathvista --z-threshold -0.3 --window-size 16
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --dataset logicvista --z-threshold -0.3 --window-size 16
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --dataset mmstar --z-threshold -0.3 --window-size 16
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --dataset hallusionbench --z-threshold -0.3 --window-size 16
```

* **Input:** `DATASET_PATHS[<dataset>]` and `MODEL_PATHS` from `main.py`.
* **Output:** `results/<dataset>/responses_<dataset>_<split>_flashattn2_kvledger_zneg0p3_w16_init1_max2048_seed<seed>_full.jsonl`.
* **Log:** terminal output is also mirrored to `logs/<dataset>/loginfo_<experiment>_<timestamp>.log`.
* **Resume:** if a run is interrupted, run the same command again. Samples that already have a record are skipped; add `--retry-errors` to rerun failed samples. The run refuses to append to a file written with different settings.
* **Smoke test:** to check the setup first, run three samples into a separate file:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --dataset mmstar --z-threshold -0.3 --window-size 16 --max-samples 3 --output-file results/_smoke/mmstar_n3.jsonl
```

### Step 5: Evaluate Results

Each benchmark has its own evaluation script. The scripts read the ground truth from `DATASET_PATHS` and match it to the records by sample ID. They do not need a GPU or torch.

```bash
python eva_mathvista.py
python eva_logicvista.py
python eva_mmstar.py
python eva_hallusionbench.py
```

* **Input:** with `INPUT_FILES = []` (the default, at the top of each script), the script evaluates every `results/<dataset>/*.jsonl`. To evaluate specific files, list them in `INPUT_FILES` as paths relative to the repository root.
* **Output** (printed to the terminal):
  * For each file: accuracy, prompt tokens per model, generated tokens (attempted, committed) and latency per sample.

| Script | Accuracy reported |
|---|---|
| `eva_mathvista.py` | Accuracy compatible with the official MathVista scoring, over 1,000 samples |
| `eva_logicvista.py` | Accuracy over all 448 samples |
| `eva_mmstar.py` | Accuracy over 1,500 samples |
| `eva_hallusionbench.py` | aAcc, qAcc and fAcc over 951 samples |


---

## 2. Customizing Models and Datasets

### Changing the Hyperparameters

Pass the threshold and window size on the command line; no code change is needed. The other arguments are listed by `python main.py --help`.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python main.py --dataset mmstar --z-threshold -0.5 --window-size 8
```

The z-score uses the statistics entry whose window length (4, 8, 16, 32 or 64) is closest to the actual window. The paper selected `w = 16` and `τ = -0.3` by a parameter search on 600 randomly sampled RealWorldQA instances, which do not overlap with any evaluation set. RealWorldQA is supported as `--dataset realworldqa` (Hugging Face `xai-org/RealworldQA`, 765 samples, evaluated by `eva_realworldqa.py`).

### Changing the Dataset

1. Create `benchmarks/<name>.py`, starting from `benchmarks/_template.py`. It declares the ID, image and prompt columns, the answer columns and the task prompt, and ends with `BENCHMARK = <Class>()`. It is registered automatically.
2. Add `"<name>": "<path>"` to `DATASET_PATHS` in `main.py`.
3. Write `eva_<name>.py`, starting from the script of the most similar benchmark.
4. Run `python main.py --dataset <name> --z-threshold -0.3`, then `python eva_<name>.py`.

Every sample must have exactly one image. The task prompt must not mention the relay; the relay instruction (`core/prompts.py`) is added in front of it automatically. See [docs/adding_a_dataset.md](docs/adding_a_dataset.md) for the field rules, prompt rules, evaluation conventions and self-checks.

### Changing the Model

1. Implement an adapter in `core/adapters.py` with the same interface as the existing ones: `prepare_sample`, `generate_bidding_window` and `generate_next_token`.
2. Register it in `MODEL_NAMES` and `ADAPTER_FACTORIES` in `core/inference.py`, and add its key to `MODEL_PATHS` and `DEVICES` in `main.py`.
3. Add `baselines/statistics/<model>_statistics.json` with `mu` and `sigma` for each window length. A model without statistics is an error.

The relay itself (`core/relay.py`) does not change.



---

## Acknowledgment

This work builds on the following open-source projects, models and datasets.

### Frameworks

- [PyTorch](https://github.com/pytorch/pytorch)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Hugging Face Datasets](https://github.com/huggingface/datasets)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)

### Models

- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- [InternVL3.5-8B](https://huggingface.co/OpenGVLab/InternVL3_5-8B)
- [MiniCPM-V-4.5](https://huggingface.co/openbmb/MiniCPM-V-4_5)

### Datasets

- [MathVista](https://huggingface.co/datasets/AI4Math/MathVista)
- [LogicVista](https://huggingface.co/datasets/lscpku/LogicVista)
- [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar)
- [HallusionBench](https://huggingface.co/datasets/lmms-lab/HallusionBench)
- [RealWorldQA](https://huggingface.co/datasets/xai-org/RealworldQA)
- [MMMU](https://huggingface.co/datasets/MMMU/MMMU)

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{liu2026multiagentrelaythinking,
      title={Multi-Agent Relay Thinking}, 
      author={Xu Liu and Jungang Qiao and Guikun Chen and Xiaojuan Qi and Wenguan Wang},
      year={2026},
      journal={arXiv preprint arXiv:2610.35219},
}
```

## Contributors

