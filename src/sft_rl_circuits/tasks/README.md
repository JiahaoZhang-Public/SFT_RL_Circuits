# Tasks Module

This package implements the rule-based textual scoring game described in the top-level README.

## Current stage
- Rule primitives are defined in `rules.py` with ID and OOD rule pools (colors, predicates, compositional variants, classification head).
- Example generation is available via `generators.py` (card sampling, rule evaluation, rich intermediates).
- Prompt/target rendering lives in `formatting.py` (`Rule / Cards / Answer` layout).
- Dataset factories for ID/OOD splits are in `dataset.py`, with optional HF Dataset conversion.
- Unit coverage exists in `tests/unit/test_tasks.py` and a quick sampler script in `scripts/py/test/generate_samples.py`.

## Example
```python
from sft_rl_circuits.tasks import dataset

bundle = dataset.build_task_datasets()
print(bundle.train.formatted[0].prompt)
print(bundle.train.formatted[0].target)
```
