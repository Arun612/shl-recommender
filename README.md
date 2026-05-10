# SHL Assessment Recommender

Conversational agent that helps hiring managers find the right SHL assessments through natural dialogue.

## Stack
- **FastAPI** — stateless `/chat` + `/health` endpoints
- **Gemini 1.5 Flash** — LLM reasoning (free tier)
- **Google text-embedding-004** — semantic retrieval
- **NumPy cosine similarity** — lightweight vector search
- **Render** — free-tier deployment

---

## Quick start (local)

```bash
# 1. Clone and enter project
git clone <your-repo>
cd shl_recommender

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Gemini API key
cp .env.example .env
# edit .env → GEMINI_API_KEY=your_key_here

# 4. (Optional) Scrape fresh catalog data
python scrape_catalog.py          # requires network access to shl.com
# The repo ships with data/catalog.json as a fallback seed

# 5. Pre-build embeddings (saves time on startup)
export $(cat .env | xargs)
python build_embeddings.py        # saves data/embeddings.npy

# 6. Run the service
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 7. Test health
curl http://localhost:8000/health
# {"status":"ok"}

# 8. Test chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"I need to hire a Java developer"}]}'

# 9. Run evaluation harness
python evaluate.py --url http://localhost:8000
```

---

## Deploying to Render (free tier)

1. Push this repo to GitHub
2. Go to https://render.com → New → Web Service → connect your repo
3. Set:
   - **Build command:** `pip install -r requirements.txt && python build_embeddings.py`
   - **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 1`
4. Add environment variable: `GEMINI_API_KEY = <your key>`
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
  "reply": "Here are 5 assessments for a mid-level Java dev...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r",       "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

- `recommendations` is `[]` when clarifying or refusing.
- `end_of_conversation` is `true` when the agent considers the task complete.
- Max 8 turns per conversation.
- Max 30 second response time per call.

---

## Project structure

```
shl_recommender/
├── app/
│   └── main.py              # FastAPI service
├── data/
│   ├── catalog.json         # Scraped SHL catalog (Individual Test Solutions)
│   └── embeddings.npy       # Pre-built embedding cache (generated)
├── scrape_catalog.py        # Scraper — run locally to refresh catalog
├── build_embeddings.py      # One-time embedding builder
├── evaluate.py              # Local eval harness
├── requirements.txt
├── Dockerfile
├── render.yaml
└── README.md
```

---

## How the agent works

1. **Retrieval** — the last 4 user turns are embedded and top-15 catalog matches retrieved by cosine similarity.
2. **LLM call** — Gemini receives: system prompt (full catalog + behavior rules) + conversation history + retrieved candidates.
3. **Validation** — every URL in the LLM output is checked against the catalog URL set. Non-catalog URLs are dropped, preventing hallucination from reaching the caller.
4. **Schema enforcement** — Pydantic validates the response shape before returning.

## Agent behavior rules (encoded in system prompt)

| Situation | Behavior |
|---|---|
| Vague first message | Ask for role and key requirements |
| Enough context | Recommend 1–10 assessments from catalog |
| User changes constraints | Update shortlist, do not restart |
| Comparison question | Answer from catalog data only |
| Off-topic / legal / salary | Refuse politely |
| Prompt injection | Refuse |
| Turn 7–8 | Commit to shortlist regardless |
