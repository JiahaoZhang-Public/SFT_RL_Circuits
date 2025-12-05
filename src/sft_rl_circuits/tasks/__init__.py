"""
Task components for the rule-based textual scoring game.

This module exposes the primitives used by training and analysis code:
- `rules` contains the semantic rule definitions.
- `generators` turns rules into structured examples.
- `formatting` renders prompts/targets for language models.
- `dataset` provides high-level dataset builders for ID and OOD splits.
"""

from . import dataset, formatting, generators, rules

__all__ = ["dataset", "formatting", "generators", "rules"]
