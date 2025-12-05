import random


from sft_rl_circuits.tasks import dataset, formatting, generators, rules


def test_rule_evaluation_color_add_subtract():
    rule = rules.IN_DISTRIBUTION_RULES[0]  # RED minus BLUE
    cards = [
        rules.Card(color=rules.Color.RED, value=3),
        rules.Card(color=rules.Color.BLUE, value=5),
        rules.Card(color=rules.Color.RED, value=4),
    ]
    result = rule.evaluate(cards)
    assert result.total == 2
    assert result.per_color_sums[rules.Color.RED] == 7
    assert result.per_color_sums[rules.Color.BLUE] == 5
    # Ensure clause bookkeeping is present
    assert len(result.clause_results) == 2
    assert result.clause_results[0].weighted_contribution == 7
    assert result.clause_results[1].weighted_contribution == -5


def test_sample_cards_cover_required_colors():
    rule = rules.IN_DISTRIBUTION_RULES[0]  # needs RED and BLUE
    rng = random.Random(0)
    cards = generators.sample_cards_for_rule(
        rule=rule,
        num_cards=5,
        rng=rng,
        allow_extra_colors=True,
    )
    colors = {card.color for card in cards}
    for color in rule.required_colors():
        assert color in colors


def test_formatting_produces_prompt_and_target():
    rng = random.Random(1)
    rule = rules.IN_DISTRIBUTION_RULES[1]
    example = generators.generate_example(
        rule=rule,
        card_count_range=(3, 3),
        rng=rng,
        paraphrase_probability=0.0,
        allow_extra_colors=False,
        split="train",
    )
    formatted = formatting.format_example(example)
    assert formatted.prompt.startswith("Rule:")
    assert "Cards:" in formatted.prompt
    assert formatted.prompt.endswith("Answer:")
    assert formatted.target.startswith(formatting.ANSWER_PREFIX)
    assert formatted.metadata["rule_name"] == rule.name
    assert formatted.metadata["response_type"] == rule.response_type.value


def test_dataset_builder_creates_expected_sizes():
    config = dataset.TaskDatasetConfig(
        train_size=10,
        val_size=4,
        id_test_size=3,
        ood_test_size=2,
        seed=42,
        ood_categories=("new_colors", "length_shift"),
    )
    bundle = dataset.build_task_datasets(config)
    assert len(bundle.train.formatted) == 10
    assert len(bundle.val.formatted) == 4
    assert len(bundle.id_test.formatted) == 3
    assert set(bundle.ood.keys()) == {"new_colors", "length_shift"}
    assert len(bundle.ood["new_colors"].formatted) == 2
    assert len(bundle.ood["length_shift"].formatted) == 2
    # Ensure length shift actually increases cards
    short_cards = bundle.train.raw[0].cards
    long_cards = bundle.ood["length_shift"].raw[0].cards
    assert len(long_cards) >= len(short_cards)
