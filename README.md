# Entity Resolver

[![CI](https://github.com/ItsMrZxD/entity_resolver/actions/workflows/ci.yml/badge.svg)](https://github.com/ItsMrZxD/entity_resolver/actions/workflows/ci.yml)

Fuzzy entity resolution between two CSV datasets that contain the same
companies under slightly different names. Every record in dataset A is
matched to its most similar record in dataset B with a 0–100 confidence
score, tolerating typos, abbreviations, legal suffixes (Inc., LLC, Ltd., …),
punctuation differences, and word-order changes.

Built with **Python 3.11+**, **pandas**, and **RapidFuzz**.

## Project structure

```
entity_resolver/
├── .github/
│   └── workflows/
│       └── ci.yml       # CI: Ruff lint + unit tests on every push
├── data/
│   ├── dataset_a.csv    # input names (auto-generated sample if missing)
│   └── dataset_b.csv    # candidate names to match against
├── output/              # created at runtime, not committed
│   ├── matches.csv                   # all matches, sorted by confidence
│   ├── matches_high_confidence.csv   # matches >= threshold
│   └── matches_low_confidence.csv    # matches < threshold (need review)
├── main.py
├── test_main.py
├── pyproject.toml       # lint configuration
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
cd entity_resolver
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows PowerShell
# .venv\Scripts\activate.bat   # Windows cmd
# source .venv/bin/activate    # Linux / macOS
pip install -r requirements.txt
```

## How to run

```bash
python main.py
```

On the first run, if `data/dataset_a.csv` and `data/dataset_b.csv` don't
exist, small sample datasets with intentionally messy company names are
generated automatically. Drop in your own CSVs (one column named `name`)
to match real data — existing files are never overwritten.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--scorer {token_sort,token_set,wratio}` | `wratio` | similarity metric to use |
| `--threshold N` | `85` | confidence cutoff between high and low matches |
| `--compare` | off | benchmark all three metrics side by side |
| `--only-above-threshold` | off | export only matches scoring >= threshold |
| `--data-dir PATH` / `--output-dir PATH` | `data/` / `output/` | input/output locations |
| `--no-progress` | off | disable the tqdm progress bar |

### Running the tests

```bash
python -m unittest
```

Linting (same check CI runs, configured in `pyproject.toml`):

```bash
pip install ruff
ruff check .
```

## How the fuzzy matching works

1. **Cleaning** — every name is lowercased, punctuation is replaced with
   spaces, whitespace is collapsed, and trailing legal suffixes (`Co`,
   `Company`, `Corp`, `Corporation`, `Inc`, `Incorporated`, `LLC`, `LLP`,
   `Ltd`, `Limited`, `PLC`) are stripped. `"Coca-Cola Company, Inc."` and
   `"coca cola"` clean to the same string.
2. **Matching** — for each cleaned name in A, RapidFuzz's
   `process.extractOne` scans *every* cleaned name in B and returns the
   single best match with a 0–100 edit-distance-based similarity score.
3. **Reporting** — results keep the *original* spellings from both files,
   are sorted by confidence descending, and are split into high/low
   confidence groups around the threshold.

### Similarity metrics

| Metric | How it scores | Best for |
|--------|--------------|----------|
| `token_sort_ratio` | Sorts the words in both names, then compares. Fixes word-order differences, but every word still has to be present. | Names with shuffled word order and few extra words. |
| `token_set_ratio` | Compares the *intersection* of words to each name. Extra words on one side barely hurt the score. | Names where one side has extra words (`"Tesla Motors"` vs `"Tesla"` → 100). Riskier: a short name inside a longer unrelated one also scores 100. |
| `WRatio` | RapidFuzz's weighted combination of several strategies with length penalties. | General-purpose default — robust without token_set's false-positive risk. |

Benchmarked on the sample data with `python main.py --compare`:

```
scorer        avg confidence  high (>= thr)  low (< thr)
token_sort              81.9              6            6
token_set               94.1             11            1
wratio                  92.2             11            1
```

`token_sort_ratio` (the spec's starting point) punishes name pairs where one
side has extra words, so it misses obvious matches like *Tesla Motors →
Tesla Inc*. `token_set_ratio` and `WRatio` both recover all 11 true matches;
**`WRatio` is the default** because it avoids `token_set_ratio`'s known
failure mode of scoring any substring-name pair at 100. Switch metrics any
time with `--scorer`.

## Changing the threshold

Two ways:

- Per run: `python main.py --threshold 90`
- Permanently: edit the `THRESHOLD` constant near the top of `main.py`

Raising it trades recall for precision. On the sample data, raising the
threshold from 85 to 95 moves borderline-but-true matches like
*Proctor & Gamble → Procter & Gamble Co* (92.9) into the low-confidence
file — exactly the group a human should review before trusting.

## Example output

`output/matches.csv` after a default run on the sample data:

```
original_name_A,matched_name_B,confidence_score
Apple Inc.,Apple,100.0
Microsoft Corporation,Microsoft Corp,100.0
Alphabet Inc.,Alphabet,100.0
Meta Platforms,Meta Platforms Incorporated,100.0
Berkshire Hathaway Inc.,Berkshire-Hathaway,100.0
Johnson & Johnson,Jonson and Johnson,95.0
The Coca-Cola Company,Coca Cola Co.,95.0
Proctor & Gamble,Procter & Gamble Co,92.9
Tesla Motors,Tesla Inc,90.0
"Amazon.com, Inc.",Amazon,90.0
Nvidia Corporaton,NVIDIA Corporation,90.0
International Business Machines,Tesla Inc,54.0
```

Console summary:

```
Summary
  Total records matched : 12
  Average confidence    : 92.2
  Above threshold (85) : 11
  Below threshold (85) : 1
```

## Assumptions & known limitations

- Each input CSV has one column named `name`; rows with missing or blank
  names are dropped with a console note instead of crashing.
- Legal suffixes are only stripped from the **end** of a name, so a word
  like *Limited* in the middle of a name survives cleaning.
- Every A record gets its best B match even when nothing is truly similar —
  that's what the confidence score and threshold are for. The sample's
  *Oracle Corporation* (only in B) is never claimed by a good match.
- Character-based fuzzy matching cannot resolve acronyms: *International
  Business Machines* vs *IBM* shares almost no characters, so it lands at
  54.0 in the low-confidence file. Solving that would need an alias
  dictionary or embedding-based matching, both out of scope here.
- Matching is O(len(A) × len(B)). Fine for thousands of records; for
  millions you'd want blocking/indexing first.

## License

MIT — see [LICENSE](LICENSE).
