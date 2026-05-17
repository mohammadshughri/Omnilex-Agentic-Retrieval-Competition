"""
Language distribution analysis for the competition corpora.
Samples rows from each CSV and reports detected language percentages.

Usage:
    python scripts/analyze_languages.py
    python scripts/analyze_languages.py --sample-size 5000 --seed 42
    python scripts/analyze_languages.py --full-court
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm

# Fix langdetect randomness for reproducibility
DetectorFactory.seed = 0

DATA_DIR = Path(__file__).parent.parent / "data"

LANG_LABELS = {
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "en": "English",
    "ro": "Romanian",
    "la": "Latin",
}


def detect_lang(text: str) -> str:
    try:
        if not isinstance(text, str) or len(text.strip()) < 20:
            return "unknown"
        return detect(text)
    except LangDetectException:
        return "unknown"


def print_distribution(counts: Counter, total: int, label: str) -> None:
    print(f"\nLanguage distribution — {label} ({total:,} rows sampled):")
    for lang, count in counts.most_common():
        name = LANG_LABELS.get(lang, lang)
        pct = 100 * count / total
        bar = "#" * int(pct / 2)
        print(f"  {name:<12} ({lang})  {count:>6,}  {pct:5.1f}%  {bar}")


def analyze_small_file(path: Path, text_col: str, sample_size: int, seed: int) -> dict:
    print(f"\n{'='*60}")
    print(f"File: {path.name}  |  column: '{text_col}'")
    print(f"{'='*60}")

    df = pd.read_csv(path, usecols=[text_col])
    total_rows = len(df)
    print(f"Total rows: {total_rows:,}")

    sample = df[text_col].dropna().sample(n=min(sample_size, total_rows), random_state=seed)
    print(f"Sampling: {len(sample):,} rows")

    counts: Counter = Counter()
    for text in tqdm(sample, desc="Detecting", unit="row"):
        counts[detect_lang(str(text))] += 1

    total = sum(counts.values())
    print_distribution(counts, total, path.name)
    return {"file": path.name, "total_rows": total_rows, "sample": total, "counts": dict(counts)}


def analyze_court_file(path: Path, sample_size: int, seed: int, full: bool) -> dict:
    print(f"\n{'='*60}")
    print(f"File: {path.name}  |  chunked sampling (large file)")
    print(f"{'='*60}")

    chunk_size = 50_000
    target_per_chunk = max(1, sample_size // 20)
    sampled: list[str] = []
    chunks_read = 0

    for chunk in tqdm(
        pd.read_csv(path, usecols=["text"], chunksize=chunk_size),
        desc="Reading chunks",
        unit="chunk",
    ):
        drawn = chunk["text"].dropna().sample(
            n=min(target_per_chunk, len(chunk)), random_state=seed
        )
        sampled.extend(drawn.tolist())
        chunks_read += 1
        if not full and len(sampled) >= sample_size:
            break

    sampled = sampled[:sample_size]
    print(f"Sampled {len(sampled):,} rows across {chunks_read} chunks")

    counts: Counter = Counter()
    for text in tqdm(sampled, desc="Detecting", unit="row"):
        counts[detect_lang(str(text))] += 1

    total = sum(counts.values())
    print_distribution(counts, total, path.name)
    return {"file": path.name, "sample": total, "counts": dict(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect language distribution in corpora")
    parser.add_argument(
        "--sample-size", type=int, default=3000,
        help="Rows to sample per file (default: 3000)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--full-court", action="store_true",
        help="Read all chunks of court_considerations.csv (slower, more representative)"
    )
    args = parser.parse_args()

    small_files = [
        (DATA_DIR / "laws_de.csv",  "text"),
        (DATA_DIR / "train.csv",    "query"),
        (DATA_DIR / "val.csv",      "query"),
        (DATA_DIR / "test.csv",     "query"),
    ]

    results = []
    for path, col in small_files:
        if path.exists():
            results.append(analyze_small_file(path, col, args.sample_size, args.seed))
        else:
            print(f"\nSKIPPED (not found): {path.name}")

    court_path = DATA_DIR / "court_considerations.csv"
    if court_path.exists():
        results.append(analyze_court_file(court_path, args.sample_size, args.seed, args.full_court))
    else:
        print(f"\nSKIPPED (not found): {court_path.name}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        top = sorted(r["counts"].items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(
            f"{LANG_LABELS.get(lang, lang)} {100 * c / r['sample']:.0f}%"
            for lang, c in top
        )
        print(f"  {r['file']:<42}  {top_str}")


if __name__ == "__main__":
    main()
