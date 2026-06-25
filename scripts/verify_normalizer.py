"""Verify CitationNormalizer handles all three citation formats in val.csv.

Run: python scripts/verify_normalizer.py
Pass criteria: 0 unparsed citations from val.csv gold labels.
"""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from omnilex.citations.normalizer import CitationNormalizer

normalizer = CitationNormalizer()
data_dir = Path(__file__).resolve().parent.parent / "data"

# Test 1: Known docket citations must parse
DOCKET_CASES = [
    ("1B_210/2023 E. 4.1", "1B_210/2023 E. 4.1"),
    ("7B_496/2025 E. 3.2", "7B_496/2025 E. 3.2"),
    ("8C_160/2016 E. 4.1", "8C_160/2016 E. 4.1"),
    ("5A_561/2020 E. 5.1.1", "5A_561/2020 E. 5.1.1"),
    ("6B_1233/2016 E. 1", "6B_1233/2016 E. 1"),
]

print("=== Test 1: Docket citation parsing ===")
docket_pass = True
for raw, expected_canonical in DOCKET_CASES:
    result = normalizer.normalize(raw)
    if result is None:
        print(f"  FAIL: '{raw}' -> None (expected '{expected_canonical}')")
        docket_pass = False
    elif result.canonical_id != expected_canonical:
        print(f"  FAIL: '{raw}' -> '{result.canonical_id}' (expected '{expected_canonical}')")
        docket_pass = False
    else:
        print(f"  OK:   '{raw}' -> '{result.canonical_id}'")
print(f"  Result: {'PASS' if docket_pass else 'FAIL'}\n")

# Test 2: Existing Art. and BGE patterns still work
ART_BGE_CASES = [
    ("Art. 221 Abs. 1 StPO", "Art. 221 Abs. 1 StPO"),
    ("BGE 137 IV 122 E. 6.2", "BGE 137 IV 122 E. 6.2"),
    ("Art. 100 Abs. 1 BGG", "Art. 100 Abs. 1 BGG"),
]

print("=== Test 2: Existing patterns still work ===")
existing_pass = True
for raw, expected_canonical in ART_BGE_CASES:
    result = normalizer.normalize(raw)
    if result is None:
        print(f"  FAIL: '{raw}' -> None")
        existing_pass = False
    elif result.canonical_id != expected_canonical:
        print(f"  FAIL: '{raw}' -> '{result.canonical_id}' (expected '{expected_canonical}')")
        existing_pass = False
    else:
        print(f"  OK:   '{raw}' -> '{result.canonical_id}'")
print(f"  Result: {'PASS' if existing_pass else 'FAIL'}\n")

# Test 3: Zero unparsed citations in val.csv
print("=== Test 3: val.csv full coverage ===")
val_path = data_dir / "val.csv"
total_raw = 0
unparsed = []
with open(val_path, encoding="utf-8") as f:
    for q in csv.DictReader(f):
        for c in q["gold_citations"].split(";"):
            c = c.strip()
            if c:
                total_raw += 1
                if normalizer.normalize(c) is None:
                    unparsed.append(c)

print(f"  Total raw citations: {total_raw}")
print(f"  Unparsed: {len(unparsed)}")
if unparsed:
    for c in sorted(set(unparsed)):
        print(f"    MISS: {c}")
val_pass = len(unparsed) == 0
print(f"  Result: {'PASS' if val_pass else 'FAIL'}\n")

# Final verdict
all_pass = docket_pass and existing_pass and val_pass
print(f"=== OVERALL: {'PASS' if all_pass else 'FAIL'} ===")
sys.exit(0 if all_pass else 1)
