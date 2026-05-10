"""
Pre-build embedding cache.
Usage:
  $env:GEMINI_API_KEY = "your_key_here"
  python build_embeddings.py
"""

import json, os, time
import numpy as np
from pathlib import Path
import requests

CATALOG_PATH = Path("data/catalog.json")
EMBED_CACHE  = Path("data/embeddings.npy")
API_KEY      = os.environ["GEMINI_API_KEY"]

EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-embedding-001:embedContent?key={API_KEY}"
)

def embed_text(text: str) -> list[float]:
    resp = requests.post(
        EMBED_URL,
        json={
            "model": "models/gemini-embedding-001",
            "content": {"parts": [{"text": text}]},
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]

with open(CATALOG_PATH) as f:
    catalog = json.load(f)

def build_doc_text(item):
    types_str = ", ".join(item.get("test_types", []))
    return (
        f"Name: {item['name']}. "
        f"Test types: {types_str}. "
        f"Description: {item.get('description', '')}. "
        f"Duration: {item.get('duration', '')}. "
        f"Remote: {item.get('remote_testing', False)}. "
        f"Adaptive: {item.get('adaptive_irt', False)}."
    )

texts = [build_doc_text(item) for item in catalog]
vecs  = []

print(f"Embedding {len(texts)} assessments...")
for i, text in enumerate(texts):
    vec = embed_text(text)
    vecs.append(vec)
    print(f"  {i+1}/{len(texts)}: {catalog[i]['name']}")
    time.sleep(0.1)

arr = np.array(vecs, dtype=np.float32)
np.save(str(EMBED_CACHE), arr)
print(f"\nSaved {arr.shape} embeddings to {EMBED_CACHE}")