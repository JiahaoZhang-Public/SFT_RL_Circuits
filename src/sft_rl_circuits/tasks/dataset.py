"""
Dataset builders for the rule-based scoring game.

This module keeps generation separate from formatting so that callers can
choose between plain Python lists or Hugging Face Datasets without repeating
sampling logic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import formatting, generators, rules

try:
    from datasets import Dataset
except ImportError:
    Dataset = None  # type: ignore


DEFAULT_OOD_CATEGORIES = ("new_colors", "compositional", "paraphrase", "length_shift", "classification")


@dataclass
class TaskDatasetConfig:
    train_size: int = 1000
    val_size: int = 200
    id_test_size: int = 200
    ood_test_size: int = 200
    num_examples_by_ood: Dict[str, int] = field(default_factory=dict)
    seed: int = 0
    id_card_count_range: Tuple[int, int] = (3, 6)
    ood_card_count_range: Tuple[int, int] = (10, 15)
    paraphrase_probability_id: float = 0.0
    paraphrase_probability_ood: float = 0.6
    ood_categories: Tuple[str, ...] = DEFAULT_OOD_CATEGORIES

    def ood_size_for(self, category: str) -> int:
        return self.num_examples_by_ood.get(category, self.ood_test_size)


@dataclass
class TaskSplit:
    formatted: List[formatting.FormattedExample]
    raw: List[generators.GeneratedExample]

    def to_hf_dataset(self) -> "Dataset":
        if Dataset is None:
            raise ImportError("datasets is not installed; run `pip install datasets` to enable this feature.")
        data = [{"prompt": ex.prompt, "target": ex.target, **ex.metadata} for ex in self.formatted]
        return Dataset.from_list(data)


@dataclass
class TaskDatasetBundle:
    train: TaskSplit
    val: TaskSplit
    id_test: TaskSplit
    ood: Dict[str, TaskSplit]


def _format_examples(examples: Iterable[generators.GeneratedExample]) -> TaskSplit:
    formatted = formatting.format_batch(examples)
    return TaskSplit(formatted=formatted, raw=list(examples))


def _build_id_split(
    rule_pool: Sequence[rules.RuleSpec],
    num_examples: int,
    card_count_range: Tuple[int, int],
    rng: random.Random,
    paraphrase_probability: float,
    split: str,
) -> TaskSplit:
    examples = generators.generate_batch(
        rule_pool=rule_pool,
        num_examples=num_examples,
        card_count_range=card_count_range,
        rng=rng,
        paraphrase_probability=paraphrase_probability,
        allow_extra_colors=True,
        split=split,
    )
    return _format_examples(examples)


def _build_ood_split(category: str, config: TaskDatasetConfig, rng: random.Random) -> TaskSplit:
    card_count_range = config.id_card_count_range
    paraphrase_probability = config.paraphrase_probability_ood
    rule_pool: Sequence[rules.RuleSpec]

    if category == "new_colors":
        rule_pool = rules.OOD_RULE_FAMILIES["new_colors"]
    elif category == "compositional":
        rule_pool = rules.OOD_RULE_FAMILIES["compositional"]
    elif category == "paraphrase":
        rule_pool = rules.IN_DISTRIBUTION_RULES
    elif category == "length_shift":
        rule_pool = rules.IN_DISTRIBUTION_RULES
        card_count_range = config.ood_card_count_range
    elif category == "classification":
        rule_pool = rules.OOD_RULE_FAMILIES["classification"]
    else:
        raise ValueError(f"Unknown OOD category: {category}")

    examples = generators.generate_batch(
        rule_pool=rule_pool,
        num_examples=config.ood_size_for(category),
        card_count_range=card_count_range,
        rng=rng,
        paraphrase_probability=paraphrase_probability,
        allow_extra_colors=True,
        split=f"ood_{category}",
    )
    return _format_examples(examples)


def build_task_datasets(config: Optional[TaskDatasetConfig] = None) -> TaskDatasetBundle:
    config = config or TaskDatasetConfig()
    rng = random.Random(config.seed)

    train = _build_id_split(
        rule_pool=rules.IN_DISTRIBUTION_RULES,
        num_examples=config.train_size,
        card_count_range=config.id_card_count_range,
        rng=rng,
        paraphrase_probability=config.paraphrase_probability_id,
        split="train",
    )
    val = _build_id_split(
        rule_pool=rules.IN_DISTRIBUTION_RULES,
        num_examples=config.val_size,
        card_count_range=config.id_card_count_range,
        rng=rng,
        paraphrase_probability=config.paraphrase_probability_id,
        split="val",
    )
    id_test = _build_id_split(
        rule_pool=rules.IN_DISTRIBUTION_RULES,
        num_examples=config.id_test_size,
        card_count_range=config.id_card_count_range,
        rng=rng,
        paraphrase_probability=config.paraphrase_probability_id,
        split="id_test",
    )

    ood_splits: Dict[str, TaskSplit] = {}
    for category in config.ood_categories:
        ood_splits[category] = _build_ood_split(category, config, rng)

    return TaskDatasetBundle(train=train, val=val, id_test=id_test, ood=ood_splits)
