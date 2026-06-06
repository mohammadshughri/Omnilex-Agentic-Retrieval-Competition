"""Verify GWDG Academic Cloud API access.

Run: python scripts/test_gwdg_api.py
"""
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.environ.get("GWDG_API_KEY") or os.environ.get("KISSKI_API_KEY")
base_url = (
    os.environ.get("GWDG_BASE_URL")
    or os.environ.get("KISSKI_BASE_URL")
    or "https://chat-ai.academiccloud.de/v1"
)

if not api_key:
    print("ERROR: neither GWDG_API_KEY nor KISSKI_API_KEY is set in .env")
    sys.exit(1)

client = OpenAI(api_key=api_key, base_url=base_url)

# Test 1: List models
print("=== Available Models ===")
models = client.models.list().data
for m in models:
    print(f"  {m.id}")

# Test 2: Chat
print("\n=== Chat Test ===")
r = client.chat.completions.create(
    model="meta-llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "Say 'API works' and nothing else."}],
    max_tokens=10,
    temperature=0,
)
print(f"  Response: {r.choices[0].message.content}")

# Test 3: Embeddings — try known candidates, report what works
print("\n=== Embedding Test ===")
EMBEDDING_CANDIDATES = [
    "multilingual-e5-large",
    "e5-mistral-7b-instruct",
    "qwen3-embedding-4b",
    "text-embedding-ada-002",
]
embedding_ok = False
for model_id in EMBEDDING_CANDIDATES:
    try:
        emb = client.embeddings.create(
            model=model_id,
            input=["Swiss federal law article about property rights"],
        )
        dim = len(emb.data[0].embedding)
        print(f"  OK: {model_id} — dim={dim}, first5={emb.data[0].embedding[:5]}")
        embedding_ok = True
        break
    except Exception as e:
        print(f"  SKIP: {model_id} — {e}")

if not embedding_ok:
    print("  WARNING: No embedding model found on this endpoint.")
    print("  BGE-M3 re-evaluation will need to use sentence-transformers locally.")

print("\nTests complete (chat API confirmed working).")
