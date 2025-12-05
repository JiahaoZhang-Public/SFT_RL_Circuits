from sft_rl_circuits.tasks import dataset
from sft_rl_circuits.training import (
    AnswerPrefixParser,
    EvalConfig,
    GenerationConfig,
    evaluate_split,
    compute_generalization_gaps,
    evaluate_model_on_task,
)


class DummyTokenizer:
    """
    Minimal tokenizer stub for text-only generation.
    """

    def batch_decode(self, outputs, skip_special_tokens=True):
        return outputs


class DummyModel:
    """
    Echoes the ground-truth answer for each prompt.
    """

    text_only_generate = True

    def __init__(self, answer_map):
        self.answer_map = answer_map

    def generate(self, prompts, **kwargs):
        return [f"{prompt}\nAnswer: {self.answer_map[prompt]}" for prompt in prompts]


def _build_answer_map(bundle):
    mapping = {}
    for split in [bundle.id_test] + list(bundle.ood.values()):
        for ex in split.formatted:
            mapping[ex.prompt] = ex.metadata["answer"]
    return mapping


def test_evaluation_pipeline_runs_and_returns_metrics():
    cfg = dataset.TaskDatasetConfig(
        train_size=0, val_size=0, id_test_size=3, ood_test_size=2, seed=0
    )
    bundle = dataset.build_task_datasets(cfg)
    answer_map = _build_answer_map(bundle)
    model = DummyModel(answer_map)
    tokenizer = DummyTokenizer()
    parser = AnswerPrefixParser()

    eval_cfg = EvalConfig(batch_size=2, generation=GenerationConfig(max_new_tokens=4))
    results = evaluate_model_on_task(
        model=model,
        tokenizer=tokenizer,
        dataset_bundle=bundle,
        parser=parser,
        eval_config=eval_cfg,
    )

    assert "id_test" in results.split_metrics
    id_metrics = results.split_metrics["id_test"]
    assert id_metrics.n_examples == 3
    assert 0.0 <= id_metrics.accuracy <= 1.0
    assert id_metrics.accuracy == 1.0
    assert id_metrics.format_validity == 1.0

    # Check at least one OOD split evaluated.
    assert results.generalization_gaps
    for gap in results.generalization_gaps.values():
        assert gap == 0.0  # dummy model is perfect


def test_compute_generalization_gaps_handles_missing_id():
    gaps = compute_generalization_gaps(split_metrics={}, id_split_name="id_test")
    assert gaps == {}


def test_max_examples_per_split_truncates():
    cfg = dataset.TaskDatasetConfig(
        train_size=0, val_size=0, id_test_size=5, ood_test_size=0, seed=0
    )
    bundle = dataset.build_task_datasets(cfg)
    answer_map = _build_answer_map(bundle)
    model = DummyModel(answer_map)
    tokenizer = DummyTokenizer()
    parser = AnswerPrefixParser()
    eval_cfg = EvalConfig(max_examples_per_split=2)
    metrics = evaluate_split(
        model=model,
        tokenizer=tokenizer,
        examples=bundle.id_test.formatted,
        parser=parser,
        eval_config=eval_cfg,
        split_name="id_test",
    )
    assert metrics.n_examples == 2


def test_splits_to_eval_filters():
    cfg = dataset.TaskDatasetConfig(
        train_size=0, val_size=0, id_test_size=2, ood_test_size=1, seed=0
    )
    bundle = dataset.build_task_datasets(cfg)
    answer_map = _build_answer_map(bundle)
    model = DummyModel(answer_map)
    tokenizer = DummyTokenizer()
    parser = AnswerPrefixParser()
    eval_cfg = EvalConfig(splits_to_eval=("id_test",))
    results = evaluate_model_on_task(
        model=model,
        tokenizer=tokenizer,
        dataset_bundle=bundle,
        parser=parser,
        eval_config=eval_cfg,
    )
    assert set(results.split_metrics.keys()) == {"id_test"}


def test_stop_tokens_do_not_confuse_parser():
    ex = type(
        "Formatted",
        (),
        {
            "prompt": "Rule: x\nCards:\n\nAnswer:",
            "metadata": {"answer": 7, "response_type": "numeric", "classification_labels": None},
        },
    )
    stop = "Extra"
    text_with_stop = f"{ex.prompt}\nAnswer: 7\nExtra text"
    model = FixedModel({ex.prompt: text_with_stop})
    tokenizer = DummyTokenizer()
    parser = AnswerPrefixParser()
    eval_cfg = EvalConfig(generation=GenerationConfig(stop_tokens=[stop]))
    metrics = evaluate_split(
        model=model,
        tokenizer=tokenizer,
        examples=[ex],
        parser=parser,
        eval_config=eval_cfg,
        split_name="id_test",
    )
    assert metrics.format_validity == 1.0
    assert metrics.accuracy == 1.0


class FixedModel:
    text_only_generate = True

    def __init__(self, output_map):
        self.output_map = output_map

    def generate(self, prompts, **kwargs):
        return [self.output_map[p] for p in prompts]


def test_invalid_numeric_output_is_not_counted_valid():
    # One numeric example expecting integer answer.
    ex = type(
        "Formatted",
        (),
        {
            "prompt": "Rule: x\nCards:\n\nAnswer:",
            "metadata": {"answer": 5, "response_type": "numeric"},
        },
    )
    model = FixedModel({ex.prompt: f"{ex.prompt}\nAnswer: BIG"})
    tokenizer = DummyTokenizer()
    parser = AnswerPrefixParser()

    metrics = evaluate_split(
        model=model,
        tokenizer=tokenizer,
        examples=[ex],
        parser=parser,
        eval_config=EvalConfig(batch_size=1),
        split_name="id_test",
    )
    assert metrics.n_examples == 1
    assert metrics.format_validity == 0.0
    assert metrics.accuracy == 0.0


def test_classification_validity_allows_label_set_but_accuracy_separate():
    ex = type(
        "Formatted",
        (),
        {
            "prompt": "Rule: x\nCards:\n\nAnswer:",
            "metadata": {
                "answer": "BIG",
                "response_type": "classification",
                "classification_labels": ["BIG", "SMALL"],
            },
        },
    )
    model = FixedModel({ex.prompt: f"{ex.prompt}\nAnswer: SMALL"})
    tokenizer = DummyTokenizer()
    parser = AnswerPrefixParser()

    metrics = evaluate_split(
        model=model,
        tokenizer=tokenizer,
        examples=[ex],
        parser=parser,
        eval_config=EvalConfig(batch_size=1),
        split_name="id_test",
    )
    assert metrics.n_examples == 1
    # Output is in label set, so valid.
    assert metrics.format_validity == 1.0
    # But not correct.
    assert metrics.accuracy == 0.0


def test_classification_without_labels_treats_valid_as_correct_match_only():
    ex = type(
        "Formatted",
        (),
        {
            "prompt": "Rule: x\nCards:\n\nAnswer:",
            "metadata": {
                "answer": "BIG",
                "response_type": "classification",
                "classification_labels": None,
            },
        },
    )
    model = FixedModel({ex.prompt: f"{ex.prompt}\nAnswer: SMALL"})
    tokenizer = DummyTokenizer()
    parser = AnswerPrefixParser()

    metrics = evaluate_split(
        model=model,
        tokenizer=tokenizer,
        examples=[ex],
        parser=parser,
        eval_config=EvalConfig(batch_size=1),
        split_name="id_test",
    )
    # Without label set, validity collapses to exact match.
    assert metrics.format_validity == 0.0
    assert metrics.accuracy == 0.0
