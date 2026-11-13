import argparse
from pathlib import Path

MODEL_PATHS = {
    "Qwen3-VL": "",
    "InternVL": "",
    "MiniCPM-V": "",
}
DATASET_PATHS = {
    "mmstar": "",
    "mathvista": "",
    "logicvista": "",
    "hallusionbench": "",
    "realworldqa": "",
}
DEVICES = {"Qwen3-VL": "cuda:0", "InternVL": "cuda:1", "MiniCPM-V": "cuda:2"}

PROJECT_ROOT = Path(__file__).resolve().parent
STATS_DIR = PROJECT_ROOT / "baselines" / "statistics"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"


MODEL_CHOICES = ("Qwen3-VL", "InternVL", "MiniCPM-V")


def build_parser(dataset_choices: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MACR confidence-bidding relay inference")
    parser.add_argument("--dataset", required=True, choices=dataset_choices)
    parser.add_argument("--z-threshold", "--z_threshold", dest="z_threshold", type=float, required=True)
    parser.add_argument("--window-size", "--window_size", dest="window_size", type=int, default=16)
    parser.add_argument(
        "--initial-trial-windows",
        "--initial_trial_windows",
        dest="initial_trial_windows",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-new-tokens",
        "--max_new_tokens",
        dest="max_new_tokens",
        type=int,
        default=None,
        help="Defaults to the dataset file's max_new_tokens.",
    )
    parser.add_argument("--max-samples", "--max_samples", dest="max_samples", type=int)
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help="Run only these sample IDs (the dataset's id_field values).",
    )
    parser.add_argument(
        "--extend",
        action="store_true",
        help="Import the results of the same experiment run with a smaller --max-samples "
        "(the _n<M>.jsonl next to the output file), then run only the remaining samples.",
    )
    parser.add_argument(
        "--extend-from",
        default=None,
        help="With --extend: import from this JSONL instead of searching for _n<M>.jsonl.",
    )
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--cleanup-every", type=int, default=25)
    parser.add_argument(
        "--disable-candidate-cache-for",
        nargs="+",
        choices=MODEL_CHOICES,
        default=None,
        help="Disable winner candidate-KV promotion for these models; static input caching stays on.",
    )
    parser.add_argument("--dataset-path", default=None, help="Override DATASET_PATHS[<dataset>].")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--seed", type=int, default=323)
    return parser


def main(argv=None) -> None:
    from benchmarks import REGISTRY
    from macr_ext.config import ExtConfig, add_ext_arguments

    parser = build_parser(sorted(REGISTRY))
    add_ext_arguments(parser)
    args = parser.parse_args(argv)
    ext = ExtConfig.from_args(args)

    from core.inference import RuntimeConfig, run
    from macr_ext.runner import run_ext

    config = RuntimeConfig(
        model_paths=MODEL_PATHS,
        dataset_paths=DATASET_PATHS,
        devices=DEVICES,
        stats_dir=STATS_DIR,
        results_dir=RESULTS_DIR,
        logs_dir=LOGS_DIR,
    )
    benchmark = REGISTRY[args.dataset]
    if ext.delay_enabled or ext.answer_enabled:
        run_ext(benchmark, args, config, ext)
    else:
        run(benchmark, args, config)


if __name__ == "__main__":
    main()
