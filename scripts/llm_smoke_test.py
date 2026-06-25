"""Single-query LLM smoke test on val_001.

Run: python scripts/llm_smoke_test.py
Gate: >=5 gold citation hits -> pursue LLM reasoning; <3 -> focus on retrieval.
"""
import csv
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from omnilex.citations.normalizer import CitationNormalizer

api_key = os.environ.get("GWDG_API_KEY") or os.environ.get("KISSKI_API_KEY")
base_url = (
    os.environ.get("GWDG_BASE_URL")
    or os.environ.get("KISSKI_BASE_URL")
    or "https://chat-ai.academiccloud.de/v1"
)
# Use the strongest available chat model for best recall
model = os.environ.get("KISSKI_MODEL", "qwen3.5-122b-a10b")

client = OpenAI(api_key=api_key, base_url=base_url)
normalizer = CitationNormalizer()

with open(Path(__file__).resolve().parent.parent / "data" / "val.csv", encoding="utf-8") as f:
    val_001 = next(q for q in csv.DictReader(f) if q["query_id"] == "val_001")

print(f"Query: {val_001['query_id']}")
print(f"Using model: {model}\n")

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a Swiss legal expert. Given a legal question, list all "
                "applicable Swiss law citations (Art. X [Abs. Y] BOOK format, "
                "BGE VOL SECTION PAGE [E. X] format, and docket-style case numbers "
                "like 1B_210/2023 E. 4.1). "
                "Output ONLY a semicolon-separated list of citations, nothing else."
            ),
        },
        {
            "role": "user",
            "content": val_001["query"],
        },
    ],
    max_tokens=8000,  # thinking models need room to reason before producing output
    temperature=0,
)

msg = response.choices[0].message
# Qwen3.5 thinking models put final answer in content; reasoning goes to msg.reasoning.
# If content is still None (model ran out of tokens thinking), fall back to reasoning.
llm_output = (msg.content or getattr(msg, "reasoning", None) or "").strip()
print(f"LLM raw output:\n{llm_output}\n")

llm_cites = [c.strip() for c in llm_output.split(";") if c.strip()]
llm_canonical = normalizer.canonicalize_list(llm_cites)
gold_raw = [c.strip() for c in val_001["gold_citations"].split(";") if c.strip()]
gold_canonical = normalizer.canonicalize_list(gold_raw)

gold_set = set(gold_canonical)
llm_set = set(llm_canonical)
hits = gold_set & llm_set

print(f"LLM predicted : {len(llm_canonical)} citations ({len(llm_cites)} raw, "
      f"{len(llm_cites) - len(llm_canonical)} unparseable)")
print(f"Gold          : {len(gold_canonical)} citations")
print(f"Hits          : {len(hits)}")
if llm_set:
    print(f"Precision     : {len(hits)/len(llm_set):.3f}")
if gold_set:
    print(f"Recall        : {len(hits)/len(gold_set):.3f}")

print(f"\nHits      : {sorted(hits)}")
print(f"Missed    : {sorted(gold_set - llm_set)}")
print(f"Hallucin. : {sorted(llm_set - gold_set)}")

print("\n--- Gate evaluation ---")
if len(hits) >= 5:
    print(f"RESULT: {len(hits)} hits >= 5 -> PURSUE LLM reasoning pipeline")
elif len(hits) >= 3:
    print(f"RESULT: {len(hits)} hits (borderline) -> investigate further before committing")
else:
    print(f"RESULT: {len(hits)} hits < 3 -> DEPRIORITIZE LLM, focus on retrieval improvements")
