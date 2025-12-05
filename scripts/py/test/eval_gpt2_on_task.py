"""
Evaluate GPT-2 on a tiny subset of the rule-based scoring task.

This is a smoke test for the evaluation pipeline; accuracy will be low, but the
script should run end-to-end.
"""

from __future__ import annotations

import argparse
import sys

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("Install transformers/torch to run this script.") from exc

from sft_rl_circuits.tasks import dataset as task_dataset
from sft_rl_circuits.training import (
    AnswerPrefixParser,
    EvalConfig,
    GenerationConfig,
    evaluate_model_on_task,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 on the rule-based scoring task.")
    parser.add_argument(
        "--model", default="openai-community/gpt2", help="HF model name or local path"
    )
    parser.add_argument(
        "--device", default="cpu", help="Device string for torch (e.g., cpu, cuda:0)"
    )
    parser.add_argument(
        "--max-examples", type=int, default=8, help="Cap examples per split for speed"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading model {args.model} on {args.device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(args.device)
    model.eval()

    # Build a tiny dataset to keep runtime short.
    ds_config = task_dataset.TaskDatasetConfig(
        train_size=0,
        val_size=0,
        id_test_size=args.max_examples,
        ood_test_size=args.max_examples,
        seed=0,
        paraphrase_probability_id=0.0,
        paraphrase_probability_ood=0.0,
    )
    bundle = task_dataset.build_task_datasets(ds_config)

    eval_cfg = EvalConfig(
        device=args.device,
        batch_size=2,
        max_examples_per_split=args.max_examples,
        generation=GenerationConfig(max_new_tokens=4, do_sample=False),
    )
    parser = AnswerPrefixParser()

    with torch.no_grad():
        results = evaluate_model_on_task(
            model=model,
            tokenizer=tokenizer,
            dataset_bundle=bundle,
            parser=parser,
            eval_config=eval_cfg,
        )

    print("=== Metrics ===")
    for name, metrics in results.split_metrics.items():
        print(
            f"{name}: accuracy={metrics.accuracy:.3f}, format_validity={metrics.format_validity:.3f}, n={metrics.n_examples}"
        )
    if results.generalization_gaps:
        print("=== Generalization gaps (id_test - ood) ===")
        for name, gap in results.generalization_gaps.items():
            print(f"{name}: {gap:.3f}")


if __name__ == "__main__":
    sys.exit(main())
