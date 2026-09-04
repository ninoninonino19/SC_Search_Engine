"""The Porter stemming algorithm, written out rather than imported.

Porter (1980), the original — not Porter2/Snowball. Roughly 150 lines, no
dependencies, and worth writing by hand once: the whole algorithm is five
passes of suffix rewriting gated on a single measure, `m`, which counts the
vowel-consonant alternations in the stem.

    measure("tree")     = 0     [c]vc? no full VC pair
    measure("trouble")  = 1     tr-ou-bl-e
    measure("orrery")   = 2

The gates matter: `relational -> relate` fires because the stem has m > 0, while
`rational` keeps its `-ate` because it does not. That is the entire trick.

Notes on this corpus specifically: Porter is aggressive with Latin, turning
`mandamus` into `mandamu` and `certiorari` into `certiorari` by luck rather than
design. The protected-terms list in `indexer.tokenizer` handles that; this file
stays a faithful implementation of the published algorithm.
"""

from __future__ import annotations

VOWELS = frozenset("aeiou")


def _is_consonant(word: str, index: int) -> bool:
    letter = word[index]
    if letter in VOWELS:
        return False
    # `y` is a consonant unless the letter before it is one ("toy" vs "syzygy").
    if letter == "y":
        return index == 0 or not _is_consonant(word, index - 1)
    return True


def _measure(stem: str) -> int:
    """Number of VC sequences in `stem` — Porter's `m`."""
    count = 0
    index = 0
    length = len(stem)

    while index < length and _is_consonant(stem, index):
        index += 1
    while index < length:
        while index < length and not _is_consonant(stem, index):
            index += 1
        if index >= length:
            break
        count += 1
        while index < length and _is_consonant(stem, index):
            index += 1
    return count


def _has_vowel(stem: str) -> bool:
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(stem: str) -> bool:
    return (
        len(stem) >= 2
        and stem[-1] == stem[-2]
        and _is_consonant(stem, len(stem) - 1)
    )


def _ends_cvc(stem: str) -> bool:
    """consonant-vowel-consonant where the final consonant is not w, x or y."""
    if len(stem) < 3:
        return False
    if not (
        _is_consonant(stem, len(stem) - 3)
        and not _is_consonant(stem, len(stem) - 2)
        and _is_consonant(stem, len(stem) - 1)
    ):
        return False
    return stem[-1] not in "wxy"


def _replace(word: str, suffix: str, replacement: str, min_measure: int = -1) -> str | None:
    """Swap `suffix` for `replacement` if the remaining stem passes the gate."""
    if not word.endswith(suffix):
        return None
    stem = word[: len(word) - len(suffix)]
    if min_measure >= 0 and _measure(stem) <= min_measure:
        return None
    return stem + replacement


_STEP2 = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
]

_STEP3 = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
]

_STEP4 = [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment",
    "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
]


def stem(word: str) -> str:
    """Stem a single lowercase word. Words of two letters or fewer pass through."""
    if len(word) <= 2:
        return word

    # Step 1a — plurals.
    for suffix, replacement in (("sses", "ss"), ("ies", "i"), ("ss", "ss"), ("s", "")):
        if word.endswith(suffix):
            word = word[: len(word) - len(suffix)] + replacement
            break

    # Step 1b — past tense and gerunds.
    grew = False
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    else:
        for suffix in ("ed", "ing"):
            if word.endswith(suffix) and _has_vowel(word[: len(word) - len(suffix)]):
                word = word[: len(word) - len(suffix)]
                grew = True
                break

    if grew:
        # Restore the shape the suffix removal broke: hop-ing -> hop, not hopp;
        # troubl-ed -> trouble, not troubl.
        if word.endswith(("at", "bl", "iz")):
            word += "e"
        elif _ends_double_consonant(word) and word[-1] not in "lsz":
            word = word[:-1]
        elif _measure(word) == 1 and _ends_cvc(word):
            word += "e"

    # Step 1c — terminal y to i.
    if word.endswith("y") and _has_vowel(word[:-1]):
        word = word[:-1] + "i"

    # Step 2, 3 — collapse derivational suffixes onto a common stem.
    for suffix, replacement in _STEP2:
        result = _replace(word, suffix, replacement, min_measure=0)
        if result is not None:
            word = result
            break

    for suffix, replacement in _STEP3:
        result = _replace(word, suffix, replacement, min_measure=0)
        if result is not None:
            word = result
            break

    # Step 4 — strip the suffix entirely when the stem is long enough to survive.
    for suffix in _STEP4:
        if not word.endswith(suffix):
            continue
        candidate = word[: len(word) - len(suffix)]
        if _measure(candidate) > 1:
            word = candidate
            break
    else:
        if word.endswith("ion"):
            candidate = word[:-3]
            if _measure(candidate) > 1 and candidate.endswith(("s", "t")):
                word = candidate

    # Step 5a, 5b — tidy the tail.
    if word.endswith("e"):
        measure = _measure(word[:-1])
        if measure > 1 or (measure == 1 and not _ends_cvc(word[:-1])):
            word = word[:-1]

    if word.endswith("ll") and _measure(word) > 1:
        word = word[:-1]

    return word
