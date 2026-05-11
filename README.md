# SHL Assessment Recommender

Conversational agent that helps hiring managers find the right SHL assessments through natural dialogue.

**Live API:** `https://shl-recommender-m7uo.onrender.com`

## Stack
- **FastAPI** — stateless `/chat` + `/health` endpoints
- **Groq (Llama-3.3-70b-versatile)** — LLM reasoning (free tier, no per-minute throttle)
- **Google gemini-embedding-001** — semantic retrieval via direct REST calls
- **NumPy cosine similarity** — lightweight vector search (49-item catalog fits in RAM)
- **Render** — free-tier deployment

---

## Quick start (local)

```bash
# 1. Clone and enter project
git clone <your-repo>
cd shl_recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API keys (PowerShell)
$env:GEMINI_API_KEY = "your_gemini_key"   # for embeddings
$env:GROQ_API_KEY   = "your_groq_key"     # for chat (get free key at console.groq.com)

# 4. Pre-build embeddings (run once, saves data/embeddings.npy)
python build_embeddings.py

# 5. Run the service
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Test health
curl http://localhost:8000/health
# {"status":"ok"}

# 7. Test chat
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"I need to hire a Java developer"}]}'

# 8. Run evaluation harness
python evaluate.py --url http://localhost:8000
```

---

## Deploying to Render (free tier)

1. Push this repo to GitHub (include `data/embeddings.npy` — do NOT gitignore it)
2. Go to https://render.com → New → Web Service → connect your repo
3. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`
4. Add environment variables:
   - `GEMINI_API_KEY = <your gemini key>`
   - `GROQ_API_KEY = <your groq key>`
5. Deploy. `/health` allows up to 2 minutes for cold start.

---

## API reference

### GET /health
Returns `{"status": "ok"}` with HTTP 200.

### POST /chat

**Request**
```json
{
  "messages": [
    {"role": "user",      "content": "Hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "What seniority level?"},
    {"role": "user",      "content": "Mid-level, 4 years"}
  ]
}
```

**Response**
```json
{
  "reply": "Here are assessments for a mid-level Java dev with stakeholder needs.",
  "recommendations": [
    {"name": "Java (New)",    "url": "https://www.shl.com/solutions/products/product-catalog/view/java-new/",    "test_type": "K"},
    {"name": "OPQ32r",        "url": "https://www.shl.com/solutions/products/product-catalog/view/opq32r/",      "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` is `[]` when clarifying or refusing.
- `end_of_conversation` is `true` when the agent considers the task complete.
- Max 8 turns per conversation, 30 second timeout per call.

---

## Project structure

```
shl_recommender/
├── app/
│   └── main.py              # FastAPI service (embeddings + chat + validation)
├── data/
│   ├── catalog.json         # 49 SHL Individual Test Solutions (seed catalog)
│   └── embeddings.npy       # Pre-built gemini-embedding-001 cache (3072-dim)
├── scrape_catalog.py        # Scraper — run locally to refresh catalog from shl.com
├── build_embeddings.py      # One-time embedding builder (uses Gemini REST API)
├── evaluate.py              # Local eval harness (8 synthetic traces)
├── requirements.txt
├── Dockerfile
├── render.yaml
└── README.md
```

---

## How the agent works

1. **Retrieval** — last 4 user turns are embedded via `gemini-embedding-001` REST API and top-15 catalog matches retrieved by cosine similarity against the cached `.npy` matrix.
2. **LLM call** — Groq receives: system prompt (full catalog + behavior rules) + conversation history + retrieved top-15 candidates injected into the last user turn.
3. **Validation** — every URL in the LLM output is checked against the catalog URL set. Non-catalog URLs are silently dropped before the response is returned. Name-only matches are resolved to the canonical catalog URL.
4. **Schema enforcement** — Pydantic validates the exact response shape (`reply`, `recommendations`, `end_of_conversation`) before returning.

## Agent behavior rules (encoded in system prompt)

| Situation | Behavior |
|---|---|
| Any first message (even if role is clear) | Ask at least one clarifying question (seniority, key competencies) |
| After one round of clarification | Commit to shortlist immediately |
| User provides full job description | Recommend immediately without clarifying |
| User changes constraints mid-conversation | Update shortlist, do not restart |
| Comparison question (e.g. OPQ32 vs OPQ32r) | Answer from catalog data only |
| Off-topic / legal / salary question | Refuse politely, redirect to assessments |
| Prompt injection attempt | Refuse |
| Turn 7-8 reached | Commit to shortlist regardless |