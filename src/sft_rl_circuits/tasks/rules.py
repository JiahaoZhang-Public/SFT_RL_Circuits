"""
Rule definitions and utilities for the textual scoring game task.

The rules are intentionally compositional so that we can create clear
in-distribution (ID) vs. out-of-distribution (OOD) splits by swapping
colors, adding predicates, or paraphrasing the natural language surface
form. Each rule is represented as a set of clauses with optional
post-processing (e.g., thresholding into BIG/SMALL).
"""

from __future__ import annotations

import enum
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


class Color(str, enum.Enum):
    RED = "RED"
    BLUE = "BLUE"
    GREEN = "GREEN"
    YELLOW = "YELLOW"


class PredicateType(str, enum.Enum):
    GREATER_THAN = "greater_than"
    LESS_EQUAL = "less_equal"
    EVEN = "even"
    ODD = "odd"


class AggregateType(str, enum.Enum):
    SUM = "sum"
    COUNT = "count"


class ResponseType(str, enum.Enum):
    NUMERIC = "numeric"
    CLASSIFICATION = "classification"


@dataclass
class ValuePredicate:
    kind: PredicateType
    threshold: Optional[int] = None

    def matches(self, value: int) -> bool:
        if self.kind == PredicateType.GREATER_THAN:
            assert self.threshold is not None
            return value > self.threshold
        if self.kind == PredicateType.LESS_EQUAL:
            assert self.threshold is not None
            return value <= self.threshold
        if self.kind == PredicateType.EVEN:
            return value % 2 == 0
        if self.kind == PredicateType.ODD:
            return value % 2 != 0
        raise ValueError(f"Unknown predicate kind: {self.kind}")

    def verbalize(self) -> str:
        if self.kind == PredicateType.GREATER_THAN:
            assert self.threshold is not None
            return f"greater than {self.threshold}"
        if self.kind == PredicateType.LESS_EQUAL:
            assert self.threshold is not None
            return f"less than or equal to {self.threshold}"
        if self.kind == PredicateType.EVEN:
            return "even"
        if self.kind == PredicateType.ODD:
            return "odd"
        raise ValueError(f"Unknown predicate kind: {self.kind}")


@dataclass
class RuleClause:
    colors: Optional[Tuple[Color, ...]] = None
    predicate: Optional[ValuePredicate] = None
    weight: int = 1
    aggregate: AggregateType = AggregateType.SUM

    def describe(self) -> str:
        parts: List[str] = []
        if self.colors:
            if len(self.colors) == 1:
                parts.append(f"{self.colors[0].value.title()} cards")
            else:
                joined = " or ".join(color.value.title() for color in self.colors)
                parts.append(f"{joined} cards")
        else:
            parts.append("all cards")

        if self.predicate:
            parts.append(f"with numbers {self.predicate.verbalize()}")

        action = "add" if self.weight >= 0 else "subtract"
        agg = "the total" if self.aggregate == AggregateType.SUM else "the count"
        return f"{action} {agg} of {' '.join(parts)}"


@dataclass
class ClassificationHead:
    threshold: int
    above_label: str = "BIG"
    below_label: str = "SMALL"

    def predict(self, total: int) -> str:
        return self.above_label if total > self.threshold else self.below_label


@dataclass
class ClauseResult:
    clause: RuleClause
    aggregated_value: int
    weighted_contribution: int
    matched_indices: List[int]


@dataclass
class RuleResult:
    total: int
    classification: Optional[str]
    clause_results: List[ClauseResult]
    per_color_sums: Dict[Color, int]


@dataclass
class RuleSpec:
    name: str
    clauses: Tuple[RuleClause, ...]
    description: str
    paraphrases: Tuple[str, ...] = field(default_factory=tuple)
    classification: Optional[ClassificationHead] = None

    @property
    def response_type(self) -> ResponseType:
        return ResponseType.CLASSIFICATION if self.classification else ResponseType.NUMERIC

    def describe(self, use_paraphrase: bool, rng: random.Random) -> str:
        if use_paraphrase and self.paraphrases:
            return rng.choice(self.paraphrases)
        return self.description

    def required_colors(self) -> Tuple[Color, ...]:
        colors: List[Color] = []
        for clause in self.clauses:
            if clause.colors:
                for color in clause.colors:
                    if color not in colors:
                        colors.append(color)
        return tuple(colors)

    def evaluate(self, cards: Sequence["Card"]) -> RuleResult:
        per_color_sums: Dict[Color, int] = defaultdict(int)
        for card in cards:
            per_color_sums[card.color] += card.value

        total = 0
        clause_results: List[ClauseResult] = []

        for clause in self.clauses:
            aggregated_value = 0
            matched_indices: List[int] = []
            for idx, card in enumerate(cards):
                if clause.colors and card.color not in clause.colors:
                    continue
                if clause.predicate and not clause.predicate.matches(card.value):
                    continue
                matched_indices.append(idx)
                if clause.aggregate == AggregateType.SUM:
                    aggregated_value += card.value
                elif clause.aggregate == AggregateType.COUNT:
                    aggregated_value += 1
                else:
                    raise ValueError(f"Unknown aggregate: {clause.aggregate}")

            weighted_contribution = clause.weight * aggregated_value
            total += weighted_contribution
            clause_results.append(
                ClauseResult(
                    clause=clause,
                    aggregated_value=aggregated_value,
                    weighted_contribution=weighted_contribution,
                    matched_indices=matched_indices,
                )
            )

        classification_label = self.classification.predict(total) if self.classification else None
        return RuleResult(
            total=total,
            classification=classification_label,
            clause_results=clause_results,
            per_color_sums=dict(per_color_sums),
        )


@dataclass
class Card:
    color: Color
    value: int


def _color_add_subtract_rule(positive: Color, negative: Color) -> RuleSpec:
    description = f"Add all {positive.value} numbers and subtract all {negative.value} numbers."
    paraphrases = (
        f"Sum the {positive.value} cards, subtract the {negative.value} cards.",
        f"Take every {positive.value} card value and minus every {negative.value} card value.",
    )
    clauses = (
        RuleClause(colors=(positive,), weight=1, aggregate=AggregateType.SUM),
        RuleClause(colors=(negative,), weight=-1, aggregate=AggregateType.SUM),
    )
    return RuleSpec(
        name=f"{positive.value.lower()}_minus_{negative.value.lower()}",
        clauses=clauses,
        description=description,
        paraphrases=paraphrases,
    )


def _threshold_sum_rule(threshold: int) -> RuleSpec:
    predicate = ValuePredicate(kind=PredicateType.GREATER_THAN, threshold=threshold)
    description = f"Add all numbers greater than {threshold}."
    paraphrases = (
        f"Sum the values of every card above {threshold}.",
        f"Only keep cards with numbers higher than {threshold} and add them up.",
    )
    clauses = (RuleClause(predicate=predicate, weight=1, aggregate=AggregateType.SUM),)
    return RuleSpec(
        name=f"sum_greater_than_{threshold}",
        clauses=clauses,
        description=description,
        paraphrases=paraphrases,
    )


def _parity_rule() -> RuleSpec:
    odd_predicate = ValuePredicate(kind=PredicateType.ODD)
    even_predicate = ValuePredicate(kind=PredicateType.EVEN)
    description = "Add all odd numbers and subtract all even numbers."
    paraphrases = (
        "Sum odds and take away evens.",
        "Add the odd-valued cards, minus the even-valued ones.",
    )
    clauses = (
        RuleClause(predicate=odd_predicate, weight=1, aggregate=AggregateType.SUM),
        RuleClause(predicate=even_predicate, weight=-1, aggregate=AggregateType.SUM),
    )
    return RuleSpec(
        name="odd_minus_even",
        clauses=clauses,
        description=description,
        paraphrases=paraphrases,
    )


def _color_predicate_rule(color: Color, predicate: ValuePredicate) -> RuleSpec:
    description = f"Add all {color.value} numbers that are {predicate.verbalize()}."
    paraphrases = (
        f"Only consider {color.value} cards; add the ones {predicate.verbalize()}.",
        f"Compute the total of {color.value} card values that are {predicate.verbalize()}.",
    )
    clauses = (
        RuleClause(colors=(color,), predicate=predicate, weight=1, aggregate=AggregateType.SUM),
    )
    return RuleSpec(
        name=f"{color.value.lower()}_{predicate.kind.value}",
        clauses=clauses,
        description=description,
        paraphrases=paraphrases,
    )


def _threshold_classifier_rule(threshold: int) -> RuleSpec:
    base_rule = _threshold_sum_rule(threshold)
    return RuleSpec(
        name=f"{base_rule.name}_classifier",
        clauses=base_rule.clauses,
        description=f"If the total is greater than {threshold}, answer BIG. Otherwise, answer SMALL.",
        paraphrases=(
            f"Return BIG when the sum exceeds {threshold}; otherwise return SMALL.",
            f"Check if the total is above {threshold}. If yes, say BIG, else say SMALL.",
        ),
        classification=ClassificationHead(threshold=threshold),
    )


# In-distribution rules: limited colors and single conditions.
IN_DISTRIBUTION_RULES: Tuple[RuleSpec, ...] = (
    _color_add_subtract_rule(Color.RED, Color.BLUE),
    _threshold_sum_rule(4),
    _parity_rule(),
)

# OOD variants grouped by shift type to make reporting clearer.
OOD_RULE_FAMILIES: Dict[str, Tuple[RuleSpec, ...]] = {
    "new_colors": (_color_add_subtract_rule(Color.GREEN, Color.YELLOW),),
    "compositional": (
        _color_predicate_rule(
            Color.RED, ValuePredicate(kind=PredicateType.GREATER_THAN, threshold=4)
        ),
        _color_predicate_rule(
            Color.BLUE, ValuePredicate(kind=PredicateType.LESS_EQUAL, threshold=3)
        ),
    ),
    "paraphrase": IN_DISTRIBUTION_RULES,
    "classification": (_threshold_classifier_rule(10),),
}


def all_rules(include_ood: bool = True) -> Tuple[RuleSpec, ...]:
    if not include_ood:
        return IN_DISTRIBUTION_RULES
    pooled: List[RuleSpec] = list(IN_DISTRIBUTION_RULES)
    for rules in OOD_RULE_FAMILIES.values():
        pooled.extend(rules)
    return tuple(pooled)


def sample_rule(rule_pool: Sequence[RuleSpec], rng: Optional[random.Random] = None) -> RuleSpec:
    rng = rng or random.Random()
    return rng.choice(rule_pool)
