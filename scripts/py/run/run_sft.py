"""
Smoke script to run SFT on the rule-based scoring task.
"""

from __future__ import annotations

import argparse

from sft_rl_circuits.config import SFTConfig, TaskConfig, build_datasets
from sft_rl_circuits.training.sft_trainer import run_sft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SFT on the rule-based scoring task.")
    parser.add_argument("--output-dir", default="outputs/sft", help="Where to save checkpoints")
    parser.add_argument(
        "--model-name", default="openai-community/gpt2", help="HF model name or path"
    )
    parser.add_argument(
        "--max-steps", type=int, default=50, help="Max training steps for a quick run"
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device train batch size")
    parser.add_argument("--max-seq-length", type=int, default=256, help="Max sequence length")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_cfg = TaskConfig()
    bundle = build_datasets(task_cfg)
    sft_cfg = SFTConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        logging_steps=10,
        eval_steps=25,
        save_steps=25,
    )
    run_sft(cfg=sft_cfg, bundle=bundle)


if __name__ == "__main__":
    main()
