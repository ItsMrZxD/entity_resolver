"""Fuzzy entity resolution between two CSV datasets of company names.

Matches every record in dataset A to its most similar record in dataset B
using RapidFuzz string similarity, tolerating typos, abbreviations, legal
suffixes (Inc., LLC, ...), punctuation differences, and word-order changes.
Each match is reported with a 0-100 confidence score.

Usage:
    python main.py                        # run with defaults
    python main.py --scorer token_sort    # pick a similarity metric
    python main.py --threshold 90         # stricter confidence threshold
    python main.py --compare              # benchmark all metrics
    python main.py --only-above-threshold # export confident matches only
"""

from __future__ import annotations

import argparse
import csv
import string
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

try:
    from tqdm import tqdm
except ImportError:  # progress bars are optional; matching works without them
    tqdm = None  # type: ignore[assignment]

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output"

# Matches scoring at or above this value count as high confidence.
THRESHOLD = 85.0

# Legal-entity suffixes stripped from the END of a name during cleaning.
# Only trailing tokens are removed, so a word like "Limited" in the middle
# of a name ("Limited Brands Design") survives.
LEGAL_SUFFIXES: frozenset[str] = frozenset({
    "co", "company", "corp", "corporation", "inc", "incorporated",
    "llc", "llp", "ltd", "limited", "plc",
})

# Maps every punctuation character to a space so "Coca-Cola" and
# "Coca Cola" clean to the same string.
_PUNCT_TO_SPACE = str.maketrans({ch: " " for ch in string.punctuation})

# Similarity metrics available on the command line (--scorer).
SCORERS: dict[str, Callable[..., float]] = {
    "token_sort": fuzz.token_sort_ratio,
    "token_set": fuzz.token_set_ratio,
    "wratio": fuzz.WRatio,
}
DEFAULT_SCORER = "wratio"

# --------------------------------------------------------------------------
# Sample data
# --------------------------------------------------------------------------

# Intentionally messy: legal suffixes, punctuation, abbreviations, typos,
# word-order noise, one blank entry, an acronym (IBM), and one company that
# exists only in B (Oracle) so unmatched noise shows up as low confidence.
SAMPLE_A = [
    "Apple Inc.",
    "Microsoft Corporation",
    "Alphabet Inc.",
    "Tesla Motors",
    "Meta Platforms",
    "Amazon.com, Inc.",
    "   ",                              # blank: exercises missing-value handling
    "Johnson & Johnson",
    "The Coca-Cola Company",
    "Nvidia Corporaton",                # typo: missing "i"
    "Berkshire Hathaway Inc.",
    "Proctor & Gamble",                 # common misspelling of "Procter"
    "International Business Machines",  # acronym in B (IBM)
]

SAMPLE_B = [
    "Apple",
    "Microsoft Corp",
    "Alphabet",
    "Tesla Inc",
    "Meta Platforms Incorporated",
    "Amazon",
    "Jonson and Johnson",               # typo + "&" spelled out
    "Coca Cola Co.",
    "NVIDIA Corporation",
    "Berkshire-Hathaway",
    "Procter & Gamble Co",
    "IBM",
    "Oracle Corporation",               # no counterpart in A
]

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def generate_sample_data(data_dir: Path) -> None:
    """Create small example CSVs when no input data exists yet.

    Existing files are never overwritten, so users can drop in their own
    ``dataset_a.csv`` / ``dataset_b.csv`` and this becomes a no-op.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, names in (("dataset_a.csv", SAMPLE_A), ("dataset_b.csv", SAMPLE_B)):
        path = data_dir / filename
        if not path.exists():
            # Quote every field so the intentionally blank entry survives
            # parsing (pandas skips unquoted whitespace-only lines entirely).
            pd.DataFrame({"name": names}).to_csv(
                path, index=False, quoting=csv.QUOTE_ALL
            )
            print(f"Generated sample data: {path}")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV containing a ``name`` column.

    Rows with missing or blank names are dropped (and reported) rather than
    crashing the matcher downstream.
    """
    df = pd.read_csv(path)
    if "name" not in df.columns:
        raise ValueError(f"{path} must contain a 'name' column")

    total = len(df)
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df = df[df["name"] != ""].reset_index(drop=True)

    dropped = total - len(df)
    if dropped:
        print(f"  note: dropped {dropped} row(s) with missing names from {path.name}")
    return df


# --------------------------------------------------------------------------
# Name cleaning
# --------------------------------------------------------------------------


def clean_name(name: str) -> str:
    """Normalize a raw name so cosmetic differences don't hurt matching.

    Steps: lowercase, replace punctuation with spaces, collapse whitespace,
    then strip trailing legal suffixes. "Coca-Cola Company, Inc." and
    "coca cola" clean to the same string.
    """
    tokens = name.lower().translate(_PUNCT_TO_SPACE).split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


# --------------------------------------------------------------------------
# Fuzzy matching
# --------------------------------------------------------------------------


def match_datasets(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    scorer: Callable[..., float],
    show_progress: bool = True,
) -> pd.DataFrame:
    """Match every name in dataset A to its most similar name in dataset B.

    Names are compared in cleaned form (see :func:`clean_name`) so that
    suffixes and punctuation don't dominate the score, but the returned
    DataFrame keeps the original spellings so results stay traceable to the
    input files. ``process.extractOne`` scans all of B for each A record and
    returns the single best match with its 0-100 similarity score.
    """
    choices = [clean_name(name) for name in df_b["name"]]
    names_a = df_a["name"].tolist()

    iterable = names_a
    if show_progress and tqdm is not None:
        iterable = tqdm(names_a, desc="Matching", unit="name")

    rows: list[tuple[str, str, float]] = []
    for original in iterable:
        query = clean_name(original)
        if not query:
            # Nothing left after cleaning (e.g. the name was just "Inc.").
            rows.append((original, "", 0.0))
            continue
        _, score, index = process.extractOne(query, choices, scorer=scorer)
        rows.append((original, df_b["name"].iloc[index], round(float(score), 1)))

    matches = pd.DataFrame(
        rows, columns=["original_name_A", "matched_name_B", "confidence_score"]
    )
    # Stable sort keeps input order within equal scores, so output is
    # deterministic across runs and platforms.
    return matches.sort_values(
        "confidence_score", ascending=False, kind="stable"
    ).reset_index(drop=True)


def compare_scorers(
    df_a: pd.DataFrame, df_b: pd.DataFrame, threshold: float
) -> None:
    """Run every available scorer and print a side-by-side comparison."""
    print(f"\nScorer comparison (threshold = {threshold:g}):\n")
    print(f"{'scorer':<12} {'avg confidence':>15} {'high (>= thr)':>14}"
          f" {'low (< thr)':>12}")
    for name, scorer in SCORERS.items():
        matches = match_datasets(df_a, df_b, scorer, show_progress=False)
        avg = matches["confidence_score"].mean()
        high = int((matches["confidence_score"] >= threshold).sum())
        print(f"{name:<12} {avg:>15.1f} {high:>14} {len(matches) - high:>12}")
    print()


# --------------------------------------------------------------------------
# Output & reporting
# --------------------------------------------------------------------------


def write_outputs(
    matches: pd.DataFrame,
    output_dir: Path,
    threshold: float,
    only_above_threshold: bool,
) -> None:
    """Write the match results plus high/low confidence splits to CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    high = matches[matches["confidence_score"] >= threshold]
    low = matches[matches["confidence_score"] < threshold]

    exported = high if only_above_threshold else matches
    exported.to_csv(output_dir / "matches.csv", index=False)
    high.to_csv(output_dir / "matches_high_confidence.csv", index=False)
    low.to_csv(output_dir / "matches_low_confidence.csv", index=False)
    print(f"\nResults written to {output_dir}")


def print_summary(matches: pd.DataFrame, threshold: float) -> None:
    """Print summary statistics for a completed matching run."""
    total = len(matches)
    above = int((matches["confidence_score"] >= threshold).sum())
    avg = matches["confidence_score"].mean() if total else 0.0

    print("\nSummary")
    print(f"  Total records matched : {total}")
    print(f"  Average confidence    : {avg:.1f}")
    print(f"  Above threshold ({threshold:g}) : {above}")
    print(f"  Below threshold ({threshold:g}) : {total - above}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Fuzzy-match company names between two CSV datasets."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="directory containing dataset_a.csv and dataset_b.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="directory where result CSVs are written",
    )
    parser.add_argument(
        "--threshold", type=float, default=THRESHOLD,
        help=f"confidence cutoff between high and low matches (default {THRESHOLD:g})",
    )
    parser.add_argument(
        "--scorer", choices=SCORERS, default=DEFAULT_SCORER,
        help=f"similarity metric to use (default {DEFAULT_SCORER})",
    )
    parser.add_argument(
        "--only-above-threshold", action="store_true",
        help="export only matches scoring at or above the threshold",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="benchmark all scorers side by side instead of exporting matches",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="disable the tqdm progress bar",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 100:
        parser.error("--threshold must be between 0 and 100")
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point: load data, match, and report."""
    args = parse_args(argv)

    generate_sample_data(args.data_dir)
    try:
        df_a = load_dataset(args.data_dir / "dataset_a.csv")
        df_b = load_dataset(args.data_dir / "dataset_b.csv")
    except ValueError as exc:  # includes pandas parser errors, which subclass it
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if df_a.empty or df_b.empty:
        print("error: no usable records to match", file=sys.stderr)
        return 1

    if args.compare:
        compare_scorers(df_a, df_b, args.threshold)
        return 0

    matches = match_datasets(
        df_a, df_b, SCORERS[args.scorer], show_progress=not args.no_progress
    )
    write_outputs(matches, args.output_dir, args.threshold, args.only_above_threshold)
    print_summary(matches, args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
