# Law Saarthi

**OOSC Hackathon: IIIT Allahabad | Problem Statement 3: AI for Civic and Legal Empowerment**

Live demo: https://civic-and-legal-empowerment.vercel.app/

---

## What it does

Rights Navigator is a grounded RAG chatbot that answers plain-language questions about tenant, consumer, and right-to-information rights under Indian law. Every answer in the chat is cited to the actual section of the source Act it came from. It also includes an **RTI Drafting Agent** that turns a plain-language request into a formatted, ready-to-file RTI application.

**Core feature: Rights Navigator (Q&A):**

- Answers grounded strictly in retrieved legal text (no hallucinated law)
- Cites the Act and Section number for every claim
- Declines gracefully, "I don't have information on this", when a question falls outside its knowledge base, instead of guessing
- Every response carries a "not legal advice" disclaimer

**Stretch feature: RTI Drafting Agent:**

- Takes a plain-language description of what information the user wants
- Produces a formatted RTI application draft

## Why this matters

Most people don't know their rights as tenants or consumers, or how to exercise their right to information — and the actual law is long, dense, and hard to search. Rights Navigator makes three specific Acts (Model Tenancy Act 2021, Consumer Protection Act 2019, RTI Act 2005) queryable in plain language, with citations so answers can be verified rather than trusted blindly.

## Architecture

```
┌─────────────┐      ┌──────────────┐       ┌─────────────────┐       ┌────────────┐
│  Frontend   │ ───▶ │  FastAPI     │ ───▶ │  ChromaDB       │       │ Gemini API │
│  (Vercel)   │      │  /ask        │       │  (vector store) │       │  (Flash)   │
│             │ ◀─── │  /draft-rti  │ ◀─── │  138 chunks     │ ◀─── │            │
└─────────────┘      └──────────────┘       └─────────────────┘       └────────────┘
```

1. Source Acts (Model Tenancy Act, Consumer Protection Act, RTI Act) are cleaned and split into per-section chunks (`chunk_docs.py`)
2. Chunks are embedded with `all-MiniLM-L6-v2` (sentence-transformers) and stored in a persistent ChromaDB collection (`build_vector_store.py`)
3. On each question, the top-K most similar chunks are retrieved; a **distance threshold (0.7)** decides whether there's real relevant context — below threshold, the question is answered by Gemini with a strict grounded system prompt; above threshold, the API returns a fallback message without calling the LLM at all
4. Every answer is returned with its source citations

## Tech stack

- **Backend:** FastAPI, ChromaDB, sentence-transformers, Google Gemini API (`gemini-3.6-flash`)
- **Frontend:** [fill in — React/Next.js/plain HTML, whatever you used]
- **Deployment:** Backend on Render, frontend on Vercel

## Setup instructions (local)

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt

# Add your Gemini API key to backend/.env:
# GEMINI_API_KEY=your_key_here

# Build the knowledge base (run once)
python chunk_docs.py
python build_vector_store.py

# Run the API
uvicorn main:app --reload
```

Test the API directly:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"My landlord wont return my security deposit, what can I do?\"}"
```

```bash
# Frontend
cd frontend
npm install
npm run dev
```

## Known limitations

- Knowledge base currently covers three Acts only (Model Tenancy Act, Consumer Protection Act, RTI Act),  questions outside these areas (e.g. labor law) are correctly declined rather than answered
- The Model Tenancy Act is a model/template Act, actual tenancy rules vary by state, since most states have their own Rent Control Acts; this is flagged in the disclaimer but worth being explicit about
- Uses Gemini's free-tier Flash model, which may have lower reasoning depth than larger models on edge-case legal nuance
- RTI Drafting Agent produces a draft for review, not a legally verified filing, users should confirm department/PIO details before submission

## Evaluation criteria mapping

- **Innovation** — grounded retrieval-first architecture that refuses to answer without real legal-text backing (via a distance threshold gate before the LLM is even called), rather than a generic legal chatbot that relies on the model's own (unverifiable, potentially wrong) knowledge of Indian law.
- **Technical implementation** — full RAG pipeline: section-aware chunking of three source Acts → sentence-transformer embeddings → ChromaDB retrieval → threshold-gated grounded generation via Gemini → cited response. Verified against adversarial prompt-injection and out-of-scope queries, both handled correctly (see Known limitations and testing below).
- **Feasibility** — built and deployed end-to-end within the hackathon window on entirely free-tier infrastructure (Gemini API free tier, Render, Vercel), so it's usable today with zero ongoing cost at small scale.
- **Scalability** — the same chunking/embedding/retrieval pipeline generalizes to any additional Act or legal domain by dropping a new cleaned source document into `data/` — no architecture changes needed. RTI Drafting Agent reuses the identical retrieval pipeline rather than a separate system.
- **Code quality** — modular pipeline (`chunk_docs.py` → `build_vector_store.py` → `main.py`), each stage independently testable and sanity-checked before the next was built (see Step 8 retrieval smoke tests during development).
- **Documentation** — this README, plus in-repo comments explaining the grounding/threshold logic and system prompt design choices.
- **Presentation** — live demo covering a real grounded Q&A, the RTI drafting flow, and an adversarial-robustness edge case (see demo video).
