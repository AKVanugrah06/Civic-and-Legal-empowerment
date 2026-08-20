"""
Step 5: Clean and chunk the legal knowledge base documents.

Splits each markdown act file into per-section chunks and writes:
  - data/chunks.json      -> list of {chunk_id, source, chapter, section_number, text}
  - data/chunk_map.json   -> simple chunk_id -> "Act, Section N" citation mapping

Handles two input formats:
  1. "raw"       -> PDF-extracted text with margin notes / em-dashes
                    (Consumer_Protection_Act_2019.md, Right_to_Information_Act_2005.md)
  2. "markdown"  -> clean hand-structured markdown with headers like
                    "### Section 11. Security deposit" and "## Chapter IV — ..."
                    (Model_Tenancy_Act.md)

Run from the backend/ folder:
    python chunk_docs.py
"""

import re
import json
import os

# ---- Config: map each markdown file to (citation name, format) ----
SOURCES = {
    "Consumer_Protection_Act_2019.md": ("Consumer Protection Act, 2019", "raw"),
    "Right_to_Information_Act_2005.md": ("Right to Information Act, 2005", "raw"),
    "Model_Tenancy_Act.md": ("Model Tenancy Act, 2021 (Summary)", "markdown"),
}

INPUT_DIR = "data"      # where the .md files live
OUTPUT_DIR = "data"     # where chunks.json / chunk_map.json get written

# ---------------------------------------------------------------------------
# Format 1: "raw" PDF-extracted text (margin notes, em-dashes)
# ---------------------------------------------------------------------------

CHAPTER_RE = re.compile(r'^\s*##\s+CHAPTER\s+([IVXLC]+)', re.IGNORECASE)
# The PDF's two-column layout sometimes extracts a margin note (e.g. "Definitions.")
# BEFORE the section number on the same line, e.g. "Definitions.       2. In this
# Act...". Look for "N. " either at the very start, or after a run of 3+ spaces
# (which marks the boundary between a margin note and the real column of text).
SECTION_START_RE = re.compile(r'(?:^|\s{3,})(\d{1,3})\.\s+(?=[A-Z(])')


def is_real_section_start(number: str, line: str, already_seen: set) -> bool:
    """Distinguish an actual section body (e.g. '3. Right to information.—Subject
    to...') from a table-of-contents entry, a footnote, or a schedule list item
    (e.g. '1. Intelligence Bureau.'), which all also start with 'N. ' but aren't
    the section itself. Real section text is long and contains an em-dash or an
    early '(1)' introducing the first sub-section."""
    if len(line) < 35:
        return False
    has_marker = ("\u2014" in line) or bool(re.search(r'\(1\)', line[:150])) or line.rstrip().endswith(('—', ':'))
    if not has_marker:
        return False
    # once we've captured the real body for a section number, later "N. ..." lines
    # (footnotes, schedule items reusing 1,2,3...) are not new sections
    if number in already_seen:
        return False
    return True


def chunk_file_raw(path: str, source_name: str):
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")

    chunks = []
    seen_sections = set()
    current_chapter = None
    current_section_num = None
    current_lines = []

    def flush():
        if current_section_num is not None and current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                chunks.append({
                    "source": source_name,
                    "chapter": current_chapter,
                    "section_number": current_section_num,
                    "text": text,
                })

    for raw_line in lines:
        line = raw_line.rstrip()

        chapter_match = CHAPTER_RE.match(line)
        if chapter_match:
            current_chapter = f"Chapter {chapter_match.group(1)}"
            continue

        start_match = SECTION_START_RE.search(line)
        if start_match and is_real_section_start(start_match.group(1), line, seen_sections):
            flush()  # new section starts -> flush the previous one
            current_section_num = start_match.group(1)
            seen_sections.add(current_section_num)
            # drop any margin-note text that appeared before the section number
            current_lines = [line[start_match.start():]]
            continue

        # skip markdown headers / TOC / footnotes / blank noise before the first
        # real section is found, or once we've moved past a section that's over
        if current_section_num is None:
            continue

        current_lines.append(line)

    flush()  # flush the final section
    return chunks


# ---------------------------------------------------------------------------
# Format 2: clean hand-structured markdown
#   "## Chapter IV — Rights and Obligations of Landlord and Tenant"
#   "### Section 15. Repair and maintenance of property"
# ---------------------------------------------------------------------------

MD_CHAPTER_RE = re.compile(r'^\s*##\s+Chapter\s+([IVXLC]+)\b(?:\s*[—\-:]\s*(.*))?', re.IGNORECASE)
MD_SECTION_RE = re.compile(r'^\s*###\s+Section\s+(\d{1,3})\.\s*(.*)')


def chunk_file_markdown(path: str, source_name: str):
    with open(path, encoding="utf-8") as f:
        lines = f.read().split("\n")

    chunks = []
    current_chapter = None
    current_section_num = None
    current_lines = []

    def flush():
        if current_section_num is not None and current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                chunks.append({
                    "source": source_name,
                    "chapter": current_chapter,
                    "section_number": current_section_num,
                    "text": text,
                })

    for raw_line in lines:
        line = raw_line.rstrip()

        chapter_match = MD_CHAPTER_RE.match(line)
        if chapter_match:
            title = chapter_match.group(2)
            current_chapter = f"Chapter {chapter_match.group(1)}" + (f" — {title}" if title else "")
            continue

        section_match = MD_SECTION_RE.match(line)
        if section_match:
            flush()  # new section starts -> flush the previous one
            current_section_num = section_match.group(1)
            current_lines = [line.lstrip("#").strip()]
            continue

        # Stop collecting once we hit the Schedules (reference/form content,
        # not substantive rights text) so they don't get glued onto the last section.
        if re.match(r'^\s*##\s+(First|Second|Third)\s+Schedule', line, re.IGNORECASE):
            flush()
            current_section_num = None
            continue

        if current_section_num is None:
            continue

        current_lines.append(line)

    flush()
    return chunks


def main():
    all_chunks = []
    for filename, (source_name, fmt) in SOURCES.items():
        path = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(path):
            print(f"  [skip] {filename} not found at {path}")
            continue
        if fmt == "markdown":
            file_chunks = chunk_file_markdown(path, source_name)
        else:
            file_chunks = chunk_file_raw(path, source_name)
        print(f"  {filename}: {len(file_chunks)} sections")
        all_chunks.extend(file_chunks)

    # assign stable IDs + build the citation map
    chunk_map = {}
    for i, c in enumerate(all_chunks):
        chunk_id = f"chunk_{i:04d}"
        c["chunk_id"] = chunk_id
        citation = f"{c['source']}, Section {c['section_number']}"
        if c["chapter"]:
            citation += f" ({c['chapter']})"
        chunk_map[chunk_id] = citation

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_DIR, "chunk_map.json"), "w", encoding="utf-8") as f:
        json.dump(chunk_map, f, indent=2, ensure_ascii=False)

    print(f"\nTotal chunks: {len(all_chunks)}")
    print(f"Wrote {OUTPUT_DIR}/chunks.json and {OUTPUT_DIR}/chunk_map.json")


if __name__ == "__main__":
    main()
