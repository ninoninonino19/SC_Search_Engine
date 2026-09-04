"""Porter stemmer, checked against the examples in the 1980 paper.

Only whole-algorithm results are asserted. The paper's per-step illustrations
(`relational -> relate`) are not final outputs — later steps keep running — and
testing against them is how people convince themselves a correct implementation
is broken.
"""

from __future__ import annotations

import pytest

from indexer.porter import _measure, stem


@pytest.mark.parametrize(
    "word,expected_measure",
    [
        ("tr", 0),
        ("ee", 0),
        ("tree", 0),
        ("y", 0),
        ("by", 0),
        ("trouble", 1),
        ("oats", 1),
        ("trees", 1),
        ("ivy", 1),
        ("troubles", 2),
        ("private", 2),
        ("oaten", 2),
        ("orrery", 2),
    ],
)
def test_measure_matches_the_paper(word: str, expected_measure: int) -> None:
    assert _measure(word) == expected_measure


@pytest.mark.parametrize(
    "word,expected",
    [
        ("caresses", "caress"),
        ("ponies", "poni"),
        ("cats", "cat"),
        ("feed", "feed"),
        ("agreed", "agre"),
        ("matting", "mat"),
        ("mating", "mate"),
        ("meeting", "meet"),
        ("milling", "mill"),
        ("messing", "mess"),
        ("meetings", "meet"),
        ("falling", "fall"),
        ("sing", "sing"),
        ("controll", "control"),
        ("roll", "roll"),
    ],
)
def test_known_stems(word: str, expected: str) -> None:
    assert stem(word) == expected


def test_short_words_pass_through() -> None:
    assert stem("is") == "is"
    assert stem("v") == "v"


def test_inflections_collapse_to_one_stem() -> None:
    """The property that actually matters for retrieval: one term, one posting."""
    family = ["connect", "connected", "connecting", "connection", "connections"]
    assert len({stem(word) for word in family}) == 1
