#splitting the files into  8 section based pn the subjects
import json
import re
from pathlib import Path

from pydantic import BaseModel

from schema import Section

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_FILE = Path(__file__).resolve().parent.parent / "data" / "processed" / "chunks.json"


# STEP 1: Define a Chunk model (pydantic BaseModel) with these fields:
#         source_file, stakeholder, section (Section | None), text, warning (str | None).
#         Reuse the Section literal from schema.py.
class Chunk(BaseModel):
    source_file: str
    stakeholder: str | None = None
    section: Section | None = None
    text: str
    warning: str | None = None


# STEP 2: Write a map from the raw header titles to your Section literals.
#         Several headers are variants of the same section, so normalize them:
#         strip the "(...)" part and lowercase the title first.
TITLE_TO_SECTION: dict[str, Section] = {
    "political position": "political_position",
    "family and ancestry": "family_ancestry",
    "personal ambitions": "personal_ambitions",
    "political alliances": "political_alliances",
    "rivalries": "rivalries",
    "military influence": "military_influence",
    "relationships with other people": "relationships",
    "historical actions": "historical_actions",  # no matching header in the raw files yet
}


def normalize_title(title: str) -> str:
    """'FAMILY AND ANCESTRY (COMPOSITION)' -> 'family and ancestry'."""
    return re.sub(r"\(.*?\)", "", title).strip().lower()


# STEP 5 pattern: a line of dashes, the TITLE, a line of dashes.
HEADER_RE = re.compile(r"^-{10,}\n([^\n]+)\n-{10,}\n", re.MULTILINE)
STAKEHOLDER_RE = re.compile(r"^STAKEHOLDER:\s*(.+)$", re.MULTILINE)
WARNING_RE = re.compile(r"^DATA WARNING\n(.+?)(?:\n\s*\n|\Z)", re.MULTILINE | re.DOTALL)


# STEP 3: read one file as a string, with encoding="utf-8".
def chunk_file(path: str | Path) -> list[Chunk]:
    """Read one stakeholder file and return a list of Chunks."""
    path = Path(path)
    source_file = path.name  # e.g. "01_cassius.txt"
    raw = path.read_text(encoding="utf-8")

    # STEP 4: pull out the stakeholder name.
    m = STAKEHOLDER_RE.search(raw)
    stakeholder = m.group(1).strip() if m else None

    # STEP 6: DATA WARNING is not a chunk; attach it to every chunk of this file.
    m = WARNING_RE.search(raw)
    warning = m.group(1).strip() if m else None

    # STEP 5: split into sections, keeping each title with the body that follows it.
    headers = list(HEADER_RE.finditer(raw))
    chunks: list[Chunk] = []
    for i, h in enumerate(headers):
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(raw)
        body = raw[h.end():body_end]  # STEP 7: kept verbatim, so quote-in-text checks work
        chunks.append(
            Chunk(
                source_file=source_file,
                stakeholder=stakeholder,
                section=TITLE_TO_SECTION.get(normalize_title(h.group(1))),
                text=body,
                warning=warning,
            )
        )
    return chunks


# STEP 8: loop over data/raw/*.txt in sorted order and return one flat list.
def chunk_all(raw_dir: str | Path = RAW_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(raw_dir).glob("*.txt")):
        chunks.extend(chunk_file(path))
    return chunks


# STEP 9: save the chunks as JSON so extract.py can read them without re-chunking.
def save_chunks(chunks: list[Chunk], out_file: str | Path = OUT_FILE) -> None:
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps([c.model_dump() for c in chunks], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# STEP 10: sanity checks.
def run_checks(raw_dir: str | Path = RAW_DIR) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for path in sorted(Path(raw_dir).glob("*.txt")):
        chunks = chunk_file(path)
        assert chunks, f"{path.name}: no chunks"  # at least 1 chunk per file
        for c in chunks:
            assert c.section is not None, f"{path.name}: unmapped section title"
        before = len(re.findall(r"^- ", path.read_text(encoding="utf-8"), re.MULTILINE))
        after = sum(len(re.findall(r"^- ", c.text, re.MULTILINE)) for c in chunks)
        assert before == after, f"{path.name}: bullets {before} != {after}"
        if "casca" in path.name:
            assert all(c.warning for c in chunks), "Casca chunks missing warning"
        print(f"{path.name}: {len(chunks)} chunks, {after} bullets")
        all_chunks.extend(chunks)
    return all_chunks


if __name__ == "__main__":
    save_chunks(run_checks())
    print(f"saved to {OUT_FILE}")
