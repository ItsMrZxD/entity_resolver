"""Unit tests for the entity resolver.

Run from the project root with:
    python -m unittest
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

from main import clean_name, load_dataset, match_datasets


class CleanNameTests(unittest.TestCase):
    """Behavior of the name-normalization step."""

    def test_lowercases_and_replaces_punctuation(self) -> None:
        self.assertEqual(clean_name("Coca-Cola!"), "coca cola")

    def test_collapses_whitespace(self) -> None:
        self.assertEqual(clean_name("  Meta   Platforms  "), "meta platforms")

    def test_strips_single_trailing_suffix(self) -> None:
        self.assertEqual(clean_name("Apple Inc."), "apple")

    def test_strips_stacked_trailing_suffixes(self) -> None:
        self.assertEqual(clean_name("Sony Co., Ltd."), "sony")

    def test_keeps_suffix_word_mid_name(self) -> None:
        self.assertEqual(clean_name("Limited Brands Design"), "limited brands design")

    def test_suffix_only_name_cleans_to_empty(self) -> None:
        self.assertEqual(clean_name("Inc."), "")


class LoadDatasetTests(unittest.TestCase):
    """CSV loading and missing-value handling."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_dir = Path(self._tmp.name)

    def _write_csv(self, text: str) -> Path:
        path = self.tmp_dir / "test.csv"
        path.write_text(text, encoding="utf-8")
        return path

    def test_drops_missing_and_blank_names(self) -> None:
        path = self._write_csv('name\nApple\n""\n"   "\nTesla\n')
        df = load_dataset(path)
        self.assertEqual(df["name"].tolist(), ["Apple", "Tesla"])

    def test_rejects_csv_without_name_column(self) -> None:
        path = self._write_csv("company\nApple\n")
        with self.assertRaises(ValueError):
            load_dataset(path)


class MatchDatasetsTests(unittest.TestCase):
    """End-to-end matching behavior on small frames."""

    @staticmethod
    def _frame(names: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"name": names})

    def _match(self, names_a: list[str], names_b: list[str]) -> pd.DataFrame:
        return match_datasets(
            self._frame(names_a), self._frame(names_b),
            fuzz.WRatio, show_progress=False,
        )

    def test_matches_despite_suffix_case_and_punctuation(self) -> None:
        row = self._match(["Apple Inc."], ["Oracle", "APPLE"]).iloc[0]
        self.assertEqual(row["matched_name_B"], "APPLE")
        self.assertEqual(row["confidence_score"], 100.0)

    def test_keeps_original_spellings_in_output(self) -> None:
        row = self._match(["Tesla Motors"], ["Tesla, Inc."]).iloc[0]
        self.assertEqual(row["original_name_A"], "Tesla Motors")
        self.assertEqual(row["matched_name_B"], "Tesla, Inc.")

    def test_suffix_only_name_gets_zero_confidence(self) -> None:
        row = self._match(["Inc."], ["Apple"]).iloc[0]
        self.assertEqual(row["matched_name_B"], "")
        self.assertEqual(row["confidence_score"], 0.0)

    def test_results_sorted_by_confidence_descending(self) -> None:
        matches = self._match(["Zebra Holdings", "Apple"], ["Apple"])
        scores = matches["confidence_score"].tolist()
        self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
