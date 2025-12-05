"""
Quick utility to sample and print a few examples for the rule-based scoring task.
"""

from __future__ import annotations

from sft_rl_circuits.tasks import dataset


def main() -> None:
    config = dataset.TaskDatasetConfig(train_size=5, val_size=0, id_test_size=0, ood_test_size=3, seed=0)
    bundle = dataset.build_task_datasets(config)

    print("=== Train examples ===")
    for ex in bundle.train.formatted:
        print(ex.prompt)
        print(ex.target)
        print("---")

    print("=== OOD (length_shift) ===")
    for ex in bundle.ood["length_shift"].formatted:
        print(ex.prompt)
        print(ex.target)
        print("---")


if __name__ == "__main__":
    main()
