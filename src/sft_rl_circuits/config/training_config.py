"""
Training configuration schemas for SFT and future RL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..training.evaluation import EvalConfig, GenerationConfig


@dataclass
class SFTConfig:
    model_name: str = "openai-community/gpt2"
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    num_train_epochs: int = 3
    max_steps: Optional[int] = None
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    logging_steps: int = 50
    eval_steps: int = 200
    save_steps: int = 200
    save_total_limit: int = 2
    max_seq_length: int = 256
    pad_to_max_length: bool = False
    mask_prompt_loss: bool = True
    output_dir: str = "outputs/sft"
    eval_config: EvalConfig = field(default_factory=EvalConfig)
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)


__all__ = ["SFTConfig"]
