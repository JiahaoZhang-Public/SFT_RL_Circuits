"""
Task-related configuration objects for the textual scoring game.

These configs keep rule selection and dataset sampling parameters centralized so
that scripts can construct reproducible datasets without importing generation
logic directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Tuple

from ..tasks import dataset as task_dataset
from ..tasks import rules


@dataclass
class RuleSelectionConfig:
    """
    Controls which rules and OOD categories are used when sampling datasets.
    """

    include_ood: bool = True
    id_rule_names: Tuple[str, ...] = field(default_factory=tuple)
    ood_categories: Tuple[str, ...] = task_dataset.DEFAULT_OOD_CATEGORIES

    def resolve_id_rules(self) -> Tuple[rules.RuleSpec, ...]:
        """Return the chosen in-distribution rule objects."""
        if not self.id_rule_names:
            return rules.IN_DISTRIBUTION_RULES
        available = {rule.name: rule for rule in rules.all_rules(include_ood=True)}
        missing = [name for name in self.id_rule_names if name not in available]
        if missing:
            raise ValueError(f"Unknown rule names: {missing}")
        return tuple(available[name] for name in self.id_rule_names)


@dataclass
class TaskConfig:
    """
    End-to-end task configuration for dataset construction and rule selection.
    """

    dataset: task_dataset.TaskDatasetConfig = field(default_factory=task_dataset.TaskDatasetConfig)
    rules: RuleSelectionConfig = field(default_factory=RuleSelectionConfig)

    def to_dataset_config(self) -> task_dataset.TaskDatasetConfig:
        """
        Convert to a TaskDatasetConfig, applying OOD category overrides if needed.
        """
        ood_categories: Tuple[str, ...] = self.rules.ood_categories
        if not self.rules.include_ood:
            ood_categories = ()
        return replace(self.dataset, ood_categories=ood_categories)


def build_datasets(config: TaskConfig) -> task_dataset.TaskDatasetBundle:
    """
    Convenience wrapper around tasks.dataset.build_task_datasets using TaskConfig.
    """
    ds_config = config.to_dataset_config()
    return task_dataset.build_task_datasets(ds_config)
