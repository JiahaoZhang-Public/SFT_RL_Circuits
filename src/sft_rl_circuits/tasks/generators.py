"""
Sampling utilities for the rule-based scoring game.

This module turns `RuleSpec` objects into concrete examples by sampling cards,
computing the ground-truth answer, and surfacing intermediate labels that are
useful for mechanistic analysis.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from . import rules


DEFAULT_VALUE_RANGE = (0, 9)


@dataclass
class SamplingConfig:
    id_card_count_range: Tuple[int, int] = (3, 6)
    ood_card_count_range: Tuple[int, int] = (10, 15)
    value_range: Tuple[int, int] = DEFAULT_VALUE_RANGE
    extra_colors: Tuple[rules.Color, ...] = (
        rules.Color.RED,
        rules.Color.BLUE,
        rules.Color.GREEN,
        rules.Color.YELLOW,
    )
    paraphrase_probability_id: float = 0.0
    paraphrase_probability_ood: float = 0.5


@dataclass
class GeneratedExample:
    rule: rules.RuleSpec
    rule_text: str
    cards: List[rules.Card]
    result: rules.RuleResult
    paraphrased: bool
    split: str

    def answer_text(self) -> str:
        return (
            self.result.classification
            if self.result.classification is not None
            else str(self.result.total)
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "rule_name": self.rule.name,
            "rule_text": self.rule_text,
            "cards": [{"color": card.color.value, "value": card.value} for card in self.cards],
            "answer": self.answer_text(),
            "response_type": self.rule.response_type.value,
            "classification_labels": (
                [self.rule.classification.above_label, self.rule.classification.below_label]
                if self.rule.classification
                else None
            ),
            "paraphrased": self.paraphrased,
            "split": self.split,
            "intermediates": {
                "total": self.result.total,
                "classification": self.result.classification,
                "per_color_sums": {c.value: v for c, v in self.result.per_color_sums.items()},
                "clause_contributions": [
                    {
                        "description": clause_result.clause.describe(),
                        "aggregated": clause_result.aggregated_value,
                        "weighted_contribution": clause_result.weighted_contribution,
                        "matched_indices": clause_result.matched_indices,
                    }
                    for clause_result in self.result.clause_results
                ],
            },
        }


def _ensure_required_colors(
    cards: List[rules.Card],
    required_colors: Sequence[rules.Color],
    value_range: Tuple[int, int],
    rng: random.Random,
) -> None:
    missing = [color for color in required_colors if color not in {card.color for card in cards}]
    for color in missing:
        value = rng.randint(*value_range)
        replace_idx = rng.randrange(len(cards))
        cards[replace_idx] = rules.Card(color=color, value=value)


def sample_cards_for_rule(
    rule: rules.RuleSpec,
    num_cards: int,
    value_range: Tuple[int, int] = DEFAULT_VALUE_RANGE,
    rng: Optional[random.Random] = None,
    allow_extra_colors: bool = True,
) -> List[rules.Card]:
    rng = rng or random.Random()
    available_colors = list(rule.required_colors())
    if allow_extra_colors:
        # Add extra colors so the model sees irrelevant distractors.
        for color in rules.Color:
            if color not in available_colors:
                available_colors.append(color)
    if not available_colors:
        # If the rule does not constrain colors, fall back to all colors.
        available_colors = list(rules.Color)

    cards = [
        rules.Card(color=rng.choice(available_colors), value=rng.randint(*value_range))
        for _ in range(num_cards)
    ]
    _ensure_required_colors(cards, rule.required_colors(), value_range, rng)
    return cards


def generate_example(
    rule: rules.RuleSpec,
    card_count_range: Tuple[int, int],
    rng: Optional[random.Random] = None,
    paraphrase_probability: float = 0.0,
    allow_extra_colors: bool = True,
    split: str = "train",
) -> GeneratedExample:
    rng = rng or random.Random()
    num_cards = rng.randint(*card_count_range)
    cards = sample_cards_for_rule(
        rule=rule,
        num_cards=num_cards,
        value_range=DEFAULT_VALUE_RANGE,
        rng=rng,
        allow_extra_colors=allow_extra_colors,
    )
    paraphrased = rng.random() < paraphrase_probability
    rule_text = rule.describe(use_paraphrase=paraphrased, rng=rng)
    result = rule.evaluate(cards)
    return GeneratedExample(
        rule=rule,
        rule_text=rule_text,
        cards=cards,
        result=result,
        paraphrased=paraphrased,
        split=split,
    )


def generate_batch(
    rule_pool: Sequence[rules.RuleSpec],
    num_examples: int,
    card_count_range: Tuple[int, int],
    rng: Optional[random.Random] = None,
    paraphrase_probability: float = 0.0,
    allow_extra_colors: bool = True,
    split: str = "train",
) -> List[GeneratedExample]:
    rng = rng or random.Random()
    examples: List[GeneratedExample] = []
    for _ in range(num_examples):
        rule = rules.sample_rule(rule_pool, rng)
        examples.append(
            generate_example(
                rule=rule,
                card_count_range=card_count_range,
                rng=rng,
                paraphrase_probability=paraphrase_probability,
                allow_extra_colors=allow_extra_colors,
                split=split,
            )
        )
    return examples
