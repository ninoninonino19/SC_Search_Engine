"""One test per domain rule in `indexer.tokenizer`.

These are the tests that pay for themselves immediately: every rule here exists
because the generic behaviour is wrong for case law, and a regression in any of
them is invisible in the search box until precision quietly drops.
"""

from __future__ import annotations

import pytest

from indexer.tokenizer import PROTECTED, STOPWORDS, scan, terms, tokenize


class TestCitations:
    """Citations must survive as single tokens."""

    def test_gr_number_is_one_token(self) -> None:
        assert terms("G.R. No. 192393") == ["gr:192393"]

    def test_gr_number_does_not_shatter(self) -> None:
        shattered = {"g", "r", "no", "192393"}
        assert not shattered & set(terms("See G.R. No. 192393."))

    def test_spelling_variants_collide(self) -> None:
        for text in ("G.R. No. 192393", "G.R. no. 192393", "GR No 192393", "G. R. No. 192393"):
            assert terms(text) == ["gr:192393"], text

    def test_consolidated_range_stays_whole(self) -> None:
        assert terms("G.R. Nos. 192393-95") == ["gr:192393-95"]

    def test_old_style_docket_keeps_its_letter(self) -> None:
        assert terms("G.R. No. L-45081") == ["gr:l-45081"]

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("A.M. No. RTJ-19-2552", "am:rtj-19-2552"),
            ("A.C. No. 9218", "ac:9218"),
            ("B.M. No. 3288", "bm:3288"),
        ],
    )
    def test_other_docket_kinds(self, text: str, expected: str) -> None:
        assert terms(text) == [expected]


class TestSectionReferences:
    """All spellings of one reference must land on one token."""

    def test_section_spellings_collide(self) -> None:
        variants = ["Sec. 5", "Section 5", "sec 5", "SECS. 5"]
        assert len({terms(v)[0] for v in variants}) == 1
        assert terms("Sec. 5") == ["sec:5"]

    def test_article_keeps_subsections(self) -> None:
        assert terms("Art. 315(2)(a)") == ["art:315(2)(a)"]

    def test_subsection_whitespace_is_normalised(self) -> None:
        assert terms("Article 315 (2) (a)") == terms("Art. 315(2)(a)")

    def test_rule_reference(self) -> None:
        assert terms("Rule 65") == ["rule:65"]

    def test_section_and_rule_in_one_phrase(self) -> None:
        assert terms("Sec. 5, Rule 65") == ["sec:5", "rule:65"]


class TestCaptions:
    """`People v. Santos` is a caption, not three unrelated words."""

    def test_versus_survives(self) -> None:
        assert "vs" in terms("People v. Santos")

    def test_versus_spellings_collide(self) -> None:
        assert terms("People v. Santos") == terms("People vs. Santos")


class TestStopwords:
    """The list is ours, and the exclusions are the point."""

    @pytest.mark.parametrize("word", ["no", "not", "party", "may", "shall", "under", "against"])
    def test_legally_meaningful_words_are_kept(self, word: str) -> None:
        assert word not in STOPWORDS

    def test_function_words_are_dropped(self) -> None:
        assert terms("the of and a an") == []


class TestLatin:
    """Stemming Latin is a silent precision loss."""

    @pytest.mark.parametrize("word", ["certiorari", "mandamus", "habeas", "corpus", "res"])
    def test_protected_terms_are_not_stemmed(self, word: str) -> None:
        assert terms(word) == [word]

    def test_res_judicata_survives_intact(self) -> None:
        assert terms("res judicata") == ["res", "judicata"]

    def test_protected_terms_are_not_also_stopwords(self) -> None:
        assert not PROTECTED & STOPWORDS


class TestNumbers:
    """Bare numerals bloat the vocabulary; numerals inside citations carry it."""

    def test_bare_numerals_are_dropped(self) -> None:
        assert terms("P1,500,000.00 paid on 12 March") == ["paid", "march"]

    def test_numerals_inside_citations_survive(self) -> None:
        assert terms("G.R. No. 192393") == ["gr:192393"]


class TestPositions:
    """Positions are gap-preserving, which is what makes phrase search work."""

    def test_dropped_stopwords_leave_a_gap(self) -> None:
        tokens = tokenize("grave abuse of discretion")
        assert [t.text for t in tokens] == ["grave", "abus", "discret"]
        # "of" is position 2 and is dropped, so "discretion" stays at 3.
        assert [t.position for t in tokens] == [0, 1, 3]

    def test_scan_counts_every_token(self) -> None:
        assert len(scan("grave abuse of discretion")) == 4

    def test_positions_are_strictly_increasing(self) -> None:
        tokens = tokenize("The petitioner filed a petition under Rule 65 of the Rules of Court.")
        positions = [t.position for t in tokens]
        assert positions == sorted(positions)
        assert len(set(positions)) == len(positions)


class TestPurity:
    """String in, list of tokens out, no I/O and no hidden state."""

    def test_repeated_calls_agree(self) -> None:
        text = "Grave abuse of discretion under Rule 65"
        assert terms(text) == terms(text)

    def test_normalisation_is_case_and_unicode_insensitive(self) -> None:
        assert terms("CERTIORARI") == terms("certiorari")
        # NFD "n" + combining tilde must match the NFC "ñ" in Las Piñas.
        assert terms("Piñas") == terms("Piñas")

    def test_stemming_can_be_turned_off(self) -> None:
        assert terms("petitions", stemming=False) == ["petitions"]
        assert terms("petitions") != ["petitions"]
