"""
Tokenization and loss masking utilities for SFT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from transformers import PreTrainedTokenizerBase

from ..tasks.formatting import ANSWER_PREFIX, FormattedExample


@dataclass
class TokenizedExample:
    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]
    metadata: Dict[str, object]


def tokenize_formatted_example(
    example: FormattedExample,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    pad_to_max_length: bool,
    mask_prompt_loss: bool = True,
) -> TokenizedExample:
    """
    Tokenize a FormattedExample and optionally mask losses on prompt tokens.
    """
    # Build a combined text: prompt + space + target
    combined = f"{example.prompt} {example.target[len(ANSWER_PREFIX):].strip()}"
    encoded = tokenizer(
        combined,
        max_length=max_length,
        padding="max_length" if pad_to_max_length else False,
        truncation=True,
        return_attention_mask=True,
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = input_ids.copy()

    if mask_prompt_loss:
        prompt_ids = tokenizer(example.prompt, add_special_tokens=False)["input_ids"]
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len

    return TokenizedExample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        metadata=example.metadata,
    )


def collate_tokenized(
    batch: Sequence[TokenizedExample], pad_token_id: int
) -> Dict[str, torch.Tensor]:
    max_len = max(len(ex.input_ids) for ex in batch)
    input_ids, attention_mask, labels = [], [], []
    for ex in batch:
        pad_length = max_len - len(ex.input_ids)
        input_ids.append(ex.input_ids + [pad_token_id] * pad_length)
        attention_mask.append(ex.attention_mask + [0] * pad_length)
        labels.append(ex.labels + [-100] * pad_length)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
