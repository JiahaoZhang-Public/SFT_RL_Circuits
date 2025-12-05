"""
Evaluation utilities for the rule-based scoring task.

The API follows a three-layer design:
1) Per-example evaluation inside `evaluate_split`.
2) Split-level aggregation into `SplitMetrics`.
3) Bundle-level aggregation into `EvalResults` with generalization gaps.

Validity semantics:
- Numeric tasks: outputs are valid only if we can parse an integer.
- Classification tasks: outputs are valid only if they match an expected label (e.g., BIG/SMALL).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import torch
except ImportError:
    torch = None  # type: ignore


@dataclass
class GenerationConfig:
    max_new_tokens: int = 8
    do_sample: bool = False
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    stop_tokens: Optional[List[str]] = None
    max_prompt_tokens: Optional[int] = None

    def to_generate_kwargs(self) -> Dict[str, object]:
        kwargs: Dict[str, object] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
        }
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        return kwargs


@dataclass
class EvalConfig:
    device: str = "cpu"
    batch_size: int = 4
    max_examples_per_split: Optional[int] = None
    splits_to_eval: Optional[Tuple[str, ...]] = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)


class AnswerParser(abc.ABC):
    @abc.abstractmethod
    def parse(self, raw_output: str) -> Tuple[bool, Optional[object]]:
        """
        Convert a raw decoded string into a structured answer.

        Returns
        -------
        is_valid : bool
            Whether the output could be parsed.
        parsed_answer : Optional[object]
            Parsed answer object if valid, else None.
        """
        raise NotImplementedError


class AnswerPrefixParser(AnswerParser):
    """
    Default parser expecting an "Answer:" prefix and returning the trailing token.
    """

    def __init__(self, prefix: str = "Answer:"):
        self.prefix = prefix

    def parse(self, raw_output: str) -> Tuple[bool, Optional[object]]:
        # Use the last non-empty line as the candidate.
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        candidate = lines[-1] if lines else raw_output.strip()
        if self.prefix in candidate:
            candidate = candidate.split(self.prefix, maxsplit=1)[-1].strip()
        # Fallback to the last whitespace-separated token.
        if not candidate:
            return False, None
        token = candidate.split()[-1]
        try:
            return True, int(token)
        except ValueError:
            return True, token.strip()


@dataclass
class SplitMetrics:
    split_name: str
    n_examples: int
    accuracy: float
    format_validity: float


@dataclass
class EvalResults:
    split_metrics: Dict[str, SplitMetrics]
    generalization_gaps: Dict[str, float] = field(default_factory=dict)


def _canonicalize_answer(value: object) -> object:
    if isinstance(value, str):
        value = value.strip()
        try:
            return int(value)
        except ValueError:
            return value.upper()
    return value


def _is_semantically_valid(
    parsed: object,
    response_type: Optional[str],
    expected_answer: object,
    allowed_labels: Optional[Sequence[object]],
) -> bool:
    if response_type == "numeric":
        return isinstance(parsed, int)
    if response_type == "classification":
        if not isinstance(parsed, str) and not isinstance(parsed, int):
            return False
        parsed_canon = _canonicalize_answer(parsed)
        if allowed_labels:
            label_set = {_canonicalize_answer(label) for label in allowed_labels}
            return parsed_canon in label_set
        expected = _canonicalize_answer(expected_answer)
        return isinstance(expected, (str, int)) and parsed_canon == expected
    return parsed is not None


def _get_prompt(ex: object) -> str:
    if hasattr(ex, "prompt"):
        return ex.prompt
    if isinstance(ex, dict) and "prompt" in ex:
        return ex["prompt"]
    raise ValueError("Example is missing a 'prompt' field.")


def _get_metadata(ex: object) -> Dict[str, object]:
    if hasattr(ex, "metadata"):
        return getattr(ex, "metadata")
    if isinstance(ex, dict) and "metadata" in ex:
        return ex["metadata"]
    raise ValueError(
        "Example is missing 'metadata'; ensure formatted examples are passed to evaluation."
    )


def _decode_outputs(outputs, tokenizer) -> List[str]:
    # Already decoded strings.
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], str):
        return outputs
    if torch is None:
        raise RuntimeError("Torch is required for decoding non-string outputs.")
    if hasattr(outputs, "to") and hasattr(outputs, "cpu"):
        outputs = outputs.cpu()
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def _apply_stop_tokens(
    texts: List[str], stop_tokens: Optional[List[str]], prompts: Optional[Sequence[str]] = None
) -> List[str]:
    if not stop_tokens:
        return texts
    cleaned: List[str] = []
    for idx_text, text in enumerate(texts):
        prefix = ""
        if prompts is not None:
            prompt = prompts[idx_text]
            if text.startswith(prompt):
                prefix = prompt
                suffix = text[len(prompt) :]
            else:
                suffix = text
        else:
            suffix = text

        # Find the earliest position of any stop token
        earliest_stop_idx = None
        for stop in stop_tokens:
            stop_idx = suffix.find(stop)
            if stop_idx != -1:
                if earliest_stop_idx is None or stop_idx < earliest_stop_idx:
                    earliest_stop_idx = stop_idx

        if earliest_stop_idx is not None:
            truncated_suffix = suffix[:earliest_stop_idx]
        else:
            truncated_suffix = suffix
        cleaned.append(prefix + truncated_suffix)
    return cleaned


def _generate_text(
    model,
    tokenizer,
    prompts: Sequence[str],
    gen_config: GenerationConfig,
    device: str,
) -> List[str]:
    gen_kwargs = gen_config.to_generate_kwargs()
    # If torch is unavailable or the model opts into text-only generation, call generate with raw prompts.
    if getattr(model, "text_only_generate", False) or torch is None:
        outputs = model.generate(prompts, **gen_kwargs)
        return _apply_stop_tokens(outputs, gen_config.stop_tokens, prompts)

    tok_kwargs: Dict[str, object] = {"return_tensors": "pt", "padding": True, "truncation": False}
    if gen_config.max_prompt_tokens is not None:
        tok_kwargs["truncation"] = True
        tok_kwargs["max_length"] = gen_config.max_prompt_tokens

    inputs = tokenizer(list(prompts), **tok_kwargs)
    if device and hasattr(inputs, "to"):
        inputs = inputs.to(device)
    output_ids = model.generate(**inputs, **gen_kwargs)
    decoded = _decode_outputs(output_ids, tokenizer)
    return _apply_stop_tokens(decoded, gen_config.stop_tokens, prompts)


def evaluate_split(
    model,
    tokenizer,
    examples: Sequence[object],
    parser: AnswerParser,
    eval_config: EvalConfig,
    split_name: str,
) -> SplitMetrics:
    max_examples = eval_config.max_examples_per_split
    if max_examples is not None:
        examples = examples[:max_examples]

    total = len(examples)
    if total == 0:
        return SplitMetrics(split_name=split_name, n_examples=0, accuracy=0.0, format_validity=0.0)

    correct = 0
    valid = 0
    batch_size = max(1, eval_config.batch_size)
    generation_config = eval_config.generation

    for start in range(0, total, batch_size):
        batch = examples[start : start + batch_size]
        prompts = [_get_prompt(ex) for ex in batch]
        model_outputs = _generate_text(
            model, tokenizer, prompts, generation_config, eval_config.device
        )
        for ex, raw_output in zip(batch, model_outputs):
            metadata = _get_metadata(ex)
            is_valid, parsed = parser.parse(raw_output)
            response_type = metadata.get("response_type")
            expected = metadata.get("answer")
            labels = metadata.get("classification_labels")
            if is_valid and _is_semantically_valid(parsed, response_type, expected, labels):
                valid += 1
                expected_canon = _canonicalize_answer(expected)
                parsed_canonical = _canonicalize_answer(parsed)
                if parsed_canonical == expected_canon:
                    correct += 1

    accuracy = correct / total
    format_validity = valid / total
    return SplitMetrics(
        split_name=split_name, n_examples=total, accuracy=accuracy, format_validity=format_validity
    )


def compute_generalization_gaps(
    split_metrics: Dict[str, SplitMetrics],
    id_split_name: str = "id_test",
) -> Dict[str, float]:
    if id_split_name not in split_metrics:
        return {}
    id_acc = split_metrics[id_split_name].accuracy
    return {
        name: id_acc - metrics.accuracy
        for name, metrics in split_metrics.items()
        if name != id_split_name
    }


def evaluate_model_on_task(
    model,
    tokenizer,
    dataset_bundle,
    parser: AnswerParser,
    eval_config: Optional[EvalConfig] = None,
) -> EvalResults:
    eval_config = eval_config or EvalConfig()
    split_metrics: Dict[str, SplitMetrics] = {}

    # Collect splits from bundle. Expected attributes: id_test, ood (dict of splits).
    candidate_splits: List[Tuple[str, Sequence[object]]] = []
    if hasattr(dataset_bundle, "id_test"):
        candidate_splits.append(("id_test", dataset_bundle.id_test.formatted))
    if hasattr(dataset_bundle, "ood"):
        for name, split in dataset_bundle.ood.items():
            candidate_splits.append((f"ood_{name}", split.formatted))

    if eval_config.splits_to_eval is not None:
        allowed = set(eval_config.splits_to_eval)
        candidate_splits = [(name, data) for name, data in candidate_splits if name in allowed]

    for split_name, split_examples in candidate_splits:
        split_metrics[split_name] = evaluate_split(
            model=model,
            tokenizer=tokenizer,
            examples=split_examples,
            parser=parser,
            eval_config=eval_config,
            split_name=split_name,
        )

    gaps = compute_generalization_gaps(split_metrics, id_split_name="id_test")
    return EvalResults(split_metrics=split_metrics, generalization_gaps=gaps)
