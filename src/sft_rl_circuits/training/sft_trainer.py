"""
Simple SFT training loop built on Hugging Face Trainer.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import inspect
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ..config.training_config import SFTConfig
from ..tasks.dataset import TaskDatasetBundle
from .evaluation import AnswerPrefixParser, EvalConfig, evaluate_model_on_task
from .tokenization import TokenizedExample, collate_tokenized, tokenize_formatted_example


class TokenizedDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[object],
        tokenizer,
        max_length: int,
        pad_to_max_length: bool,
        mask_prompt_loss: bool,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max_length = pad_to_max_length
        self.mask_prompt_loss = mask_prompt_loss

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> TokenizedExample:
        ex = self.examples[idx]
        return tokenize_formatted_example(
            example=ex,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            pad_to_max_length=self.pad_to_max_length,
            mask_prompt_loss=self.mask_prompt_loss,
        )


def _ensure_pad_token(tokenizer, model):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id


def build_datasets_for_sft(
    bundle: TaskDatasetBundle,
    tokenizer,
    cfg: SFTConfig,
):
    train_ds = TokenizedDataset(
        bundle.train.formatted,
        tokenizer=tokenizer,
        max_length=cfg.max_seq_length,
        pad_to_max_length=cfg.pad_to_max_length,
        mask_prompt_loss=cfg.mask_prompt_loss,
    )
    val_ds = TokenizedDataset(
        bundle.val.formatted,
        tokenizer=tokenizer,
        max_length=cfg.max_seq_length,
        pad_to_max_length=cfg.pad_to_max_length,
        mask_prompt_loss=cfg.mask_prompt_loss,
    )
    return train_ds, val_ds


def _build_training_args(cfg: SFTConfig) -> TrainingArguments:
    kwargs = {
        "output_dir": cfg.output_dir,
        "per_device_train_batch_size": cfg.per_device_train_batch_size,
        "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "num_train_epochs": cfg.num_train_epochs,
        "max_steps": cfg.max_steps if cfg.max_steps is not None else -1,
        "warmup_steps": cfg.warmup_steps,
        "logging_steps": cfg.logging_steps,
        "eval_steps": cfg.eval_steps,
        "save_steps": cfg.save_steps,
        "save_total_limit": cfg.save_total_limit,
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "logging_strategy": "steps",
    }
    supported = set(inspect.signature(TrainingArguments).parameters)
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported}
    args = TrainingArguments(**filtered_kwargs)
    # If strategies were filtered out, best-effort set attributes when available.
    for name in ("evaluation_strategy", "save_strategy", "logging_strategy"):
        if name in kwargs and hasattr(args, name):
            setattr(args, name, kwargs[name])
    return args


def _build_collate_fn(tokenizer, pad_to_max_length: bool, pad_token_id: int):
    def collate(batch: Sequence[TokenizedExample]) -> Dict[str, torch.Tensor]:
        if pad_to_max_length:
            return {
                "input_ids": torch.tensor([ex.input_ids for ex in batch], dtype=torch.long),
                "attention_mask": torch.tensor(
                    [ex.attention_mask for ex in batch], dtype=torch.long
                ),
                "labels": torch.tensor([ex.labels for ex in batch], dtype=torch.long),
            }
        return collate_tokenized(batch, pad_token_id=pad_token_id)

    return collate


def run_sft(
    cfg: SFTConfig,
    bundle: TaskDatasetBundle,
    tokenizer=None,
    model=None,
    eval_config: Optional[EvalConfig] = None,
):
    """
    Run supervised fine-tuning on the provided dataset bundle.
    """
    tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.model_name)
    model = model or AutoModelForCausalLM.from_pretrained(cfg.model_name)
    _ensure_pad_token(tokenizer, model)

    train_ds, val_ds = build_datasets_for_sft(bundle, tokenizer, cfg)
    training_args = _build_training_args(cfg)
    pad_token_id = tokenizer.pad_token_id
    data_collator = _build_collate_fn(tokenizer, cfg.pad_to_max_length, pad_token_id)

    parser = AnswerPrefixParser()
    eval_config = eval_config or cfg.eval_config

    def compute_metrics(_) -> Dict[str, float]:
        results = evaluate_model_on_task(
            model=model,
            tokenizer=tokenizer,
            dataset_bundle=bundle,
            parser=parser,
            eval_config=eval_config,
        )
        return {f"{name}_acc": m.accuracy for name, m in results.split_metrics.items()}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    return trainer
