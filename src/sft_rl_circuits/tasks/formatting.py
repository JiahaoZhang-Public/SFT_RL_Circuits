"""
Utilities to render structured task examples into plain-text prompts for GPT-2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from . import generators, rules


ANSWER_PREFIX = "Answer:"


def format_cards(cards: Iterable[rules.Card]) -> str:
    return "\n".join(f"{card.color.value} {card.value}" for card in cards)


@dataclass
class FormattedExample:
    prompt: str
    target: str
    metadata: Dict[str, object]


def build_prompt(rule_text: str, cards: Iterable[rules.Card]) -> str:
    card_block = format_cards(cards)
    return f"Rule: {rule_text}\nCards:\n{card_block}\n\n{ANSWER_PREFIX}"


def format_example(example: generators.GeneratedExample) -> FormattedExample:
    prompt = build_prompt(example.rule_text, example.cards)
    target = f"{ANSWER_PREFIX} {example.answer_text()}"
    metadata = example.to_dict()
    return FormattedExample(prompt=prompt, target=target, metadata=metadata)


def format_batch(examples: Iterable[generators.GeneratedExample]) -> List[FormattedExample]:
    return [format_example(example) for example in examples]
