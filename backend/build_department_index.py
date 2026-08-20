"""
Step 14 helper: embeds data/department_mapping.md into its own ChromaDB
collection ("department_mapping"), kept separate from "legal_docs" so that
department-routing queries don't compete with Act-section retrieval for
top-K slots.

Run once (and again any time department_mapping.md changes):
    python build_department_index.py

Chunking strategy: one chunk per "## <Topic>" section in the markdown file.
Each chunk keeps its example requests and department/PIO line together,
since that's the unit retrieval should match against.
"""

import os
import re

import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = "data"
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
SOURCE_FILE = os.path.join(DATA_DIR, "department_mapping.md")
COLLECTION_NAME = "department_mapping"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def parse_sections(md_text: str):
    """Split the markdown into one chunk per '## Topic' section, skipping
    the intro and the trailing 'Notes for the drafting agent' section
    (that's guidance for the LLM prompt, not a routable topic)."""
    # Split on level-2 headers, keep the header text with its body.
    parts = re.split(r"\n## ", md_text)
    sections = []
    for part in parts[1:]:  # parts[0] is the intro before the first ##
        title, _, body = part.partition("\n")
        title = title.strip()
        if title.lower().startswith("notes for the drafting agent"):
            continue
        sections.append({"title": title, "text": f"## {title}\n{body.strip()}"})
    return sections


def main():
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        md_text = f.read()

    sections = parse_sections(md_text)
    print(f"Parsed {len(sections)} department sections from {SOURCE_FILE}")

    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Recreate the collection fresh each run so stale entries never linger.
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)

    ids = [f"dept_{i}" for i in range(len(sections))]
    documents = [s["text"] for s in sections]
    metadatas = [{"topic": s["title"]} for s in sections]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Indexed {len(sections)} sections into collection '{COLLECTION_NAME}' at {CHROMA_DIR}")


if __name__ == "__main__":
    main()
