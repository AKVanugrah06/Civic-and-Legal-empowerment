"""
Step 7: Generate embeddings and build the vector store.
Step 8: Sanity-check retrieval quality.

Reads data/chunks.json (written by chunk_docs.py), embeds every chunk with
a local sentence-transformers model, and stores them in a persistent
ChromaDB collection. Then runs a handful of smoke-test queries across all
three source acts (plus one deliberately out-of-scope query) so you can
eyeball whether retrieval is actually grounded before wiring up the /ask
endpoint.

Run from the backend/ folder:
    python build_vector_store.py
"""

import json
import os

import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = "data"
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
COLLECTION_NAME = "legal_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Smoke-test queries — one per act, plus one deliberately out-of-scope
# question to check that retrieval doesn't confidently return irrelevant
# chunks when there's no real source for the answer.
TEST_QUERIES = [
    "landlord not returning security deposit",       # Model Tenancy Act
    "how do I file a consumer complaint",             # Consumer Protection Act
    "what information can I request under RTI",       # RTI Act
    "employer not paying overtime",                   # out-of-scope: no labor-law source loaded
    "landlord entering my apartment without notice",  # Model Tenancy Act
]

TOP_K = 3  # how many chunks to show per test query


def load_chunks(path: str):
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


def build_vector_store(chunks):
    print(f"Using default semantic embedding model ({EMBEDDING_MODEL}).")
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # start clean each run so re-running the script doesn't duplicate chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "source": c["source"],
            "chapter": c["chapter"] or "",
            "section_number": c["section_number"],
        }
        for c in chunks
    ]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"Embedded and stored {len(chunks)} chunks in '{COLLECTION_NAME}'")
    print(f"Vector store written to: {CHROMA_DIR}")
    return collection


def run_smoke_tests(collection):
    print("\n--- Retrieval smoke test ---")
    for query in TEST_QUERIES:
        print(f"\nQuery: '{query}'")
        results = collection.query(query_texts=[query], n_results=TOP_K)

        ids = results["ids"][0]
        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[None] * len(ids)])[0]

        if not ids:
            print("  -> no results returned")
            continue

        for chunk_id, meta, distance in zip(ids, metadatas, distances):
            chapter = f" ({meta['chapter']})" if meta.get("chapter") else ""
            score = f"  [distance: {distance:.3f}]" if distance is not None else ""
            print(
                f"  -> {chunk_id}: {meta['source']}, "
                f"Section {meta['section_number']}{chapter}{score}"
            )


def main():
    chunks = load_chunks(CHUNKS_PATH)
    collection = build_vector_store(chunks)
    run_smoke_tests(collection)


if __name__ == "__main__":
    main()
