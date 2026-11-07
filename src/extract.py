#this file will implement the code for llm to call and turn the chunk into schema
import json
import re
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from schema import Node, proof, Graph, Hyperedge, Rivalry, CUTOFF_YEAR
from chunker import Chunk, OUT_FILE as CHUNKS_FILE

# OVERALL FLOW
#   chunks.json  ->  one LLM call per chunk  ->  validate + verify quotes  ->  merge  ->  Graph  ->  graph.json
# The LLM only PROPOSES nodes/edges. Our code decides what is accepted.


# def load_chunks(path) -> list[Chunk]
#   - read data/processed/chunks.json (made by chunker.py) and rebuild Chunk objects.
def load_chunks(path: str | Path = CHUNKS_FILE) -> list[Chunk]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, run `python src/chunker.py` first")
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk.model_validate(r) for r in rows]


# def seed_stakeholders(chunks) -> list[Node]
#   - make one Node(kind="stakeholder") per source file (11 of them) straight from the
#     "STAKEHOLDER:" header, WITHOUT the LLM. Pick a clean id (e.g. "cassius", "the_senate").
#   - fill aliases by hand or from the header (e.g. "Caesar", "Julius Caesar").
#   - these ids are the anchor: every edge the LLM writes must point at an existing id.
def seed_stakeholders(chunks: list[Chunk]) -> list["Node"]:
    seen_ids = set()
    nodes = []
    for c in chunks:
        if not c.stakeholder:
            continue
        # make a clean id from the stakeholder name: lowercase, letters/digits/underscores only.
        node_id = re.sub(r"^\d+_", "", Path(c.source_file).stem)  # "01_cassius.txt" -> "cassius"
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)
        nodes.append(Node(
            id=node_id,
            name=c.stakeholder.title(),
            kind="stakeholder",
            aliases=[c.stakeholder],
            res=proof(source_file=c.source_file, supporting_quote=f"STAKEHOLDER: {c.stakeholder}", confidence="high"),
        ))
    return nodes    

# def build_prompt(chunk, known_node_ids) -> str
#   - SYSTEM rules (what the model must obey):
#       * use ONLY the chunk text, no outside/historical knowledge
#       * knowledge cutoff is 45 BCE: never mention or imply Caesar's death or anything after
#       * every node/edge needs a supporting_quote copied word for word (8-200 chars)
#       * ids are lowercase letters, digits, underscore only
#       * reuse an id from known_node_ids when the entity already exists, don't invent a duplicate
#       * prefer fewer, correct items over many guesses; if unsure, return nothing
#       * if chunk.warning is set, stay sparse and low confidence, do NOT fill gaps
#   - USER message: stakeholder name + section + chunk.text + the list of known ids.
#   - section tells the model what to look for:
#       family_ancestry -> family_tie edges, political_alliances -> alliance edges,
#       rivalries -> rivalry edges (sides, grievance), military_influence/political_position
#       -> power/office nodes, personal_ambitions -> goal/value nodes (these become `drivers`).


SYSTEM_RULES = """You extract a knowledge graph from a short history excerpt.

RULES
- Use ONLY the excerpt below. Do not use outside or historical knowledge.
- The knowledge cutoff is 45 BCE. Never mention or imply Caesar's death, or any event after the cutoff.
- Every node and edge needs a supporting_quote copied WORD FOR WORD from the excerpt (8-200 characters). No paraphrasing.
- Ids use only lowercase letters, digits and underscores.
- If an entity already exists in KNOWN IDS, reuse that id. Do not invent a duplicate.
- Prefer fewer, correct items over many guesses. If unsure, return nothing.
"""

WARNING_RULE = """
DATA WARNING for this excerpt: {warning}
Stay sparse and low confidence. Do NOT fill gaps with guesses.
"""

SECTION_HINTS = {
    "family_ancestry": "Look for family_tie edges (blood or marriage links between people).",
    "political_alliances": "Look for alliance edges (who works with whom, and in what role, work relation).",
    "rivalries": "Look for rivalry edges: the sides, each side's grievance, and what is said publicly vs what is really happening.",
    "military_influence": "Look for power nodes (commands, armies, loyalty of troops) and career_event edges.",
    "political_position": "Look for power nodes (offices, titles, wealth, legal protection) and career_event edges.",
    "personal_ambitions": "Look for goal and value nodes. These can be used as the drivers of other edges.",
    "relationships": "Look for alliance or rivalry edges between the people and groups named.",
    "historical_actions": "Look for career_event edges and revalries and its associate cause (things this stakeholder did) with dates.",
}


def build_prompt(chunk: Chunk, known_node_ids: list[str]) -> str:
    parts = [SYSTEM_RULES]
    if chunk.warning:
        parts.append(WARNING_RULE.format(warning=chunk.warning))
    hint = SECTION_HINTS.get(chunk.section or "", "")
    parts.append(
        f"STAKEHOLDER: {chunk.stakeholder}\n"
        f"SECTION: {chunk.section}\n"
        f"SOURCE FILE: {chunk.source_file}\n"
        f"{hint}\n\n"
        f"KNOWN IDS: {', '.join(sorted(known_node_ids))}\n\n"
        f"EXCERPT:\n{chunk.text}"
    )
    return "\n".join(parts)


# class ChunkExtraction(BaseModel)
#   - small output model for ONE call: nodes: list[Node], edges: list[Hyperedge].
#   - not the full Graph, because Graph rejects dangling ids and one chunk can't see everything yet.
#   - hand this model's JSON schema to the LLM (structured output / tool definition) so the
#     field descriptions in schema.py act as instructions.

class ChunkExtraction(BaseModel):
    nodes: list[Node] = Field(default_factory=list, description="Nodes proposed by the LLM for this chunk.")
    edges: list[Hyperedge] = Field(default_factory=list, description="Edges proposed by the LLM for this chunk.")
    


# def call_llm(prompt) -> ChunkExtraction
#   - call the model with structured output, parse the reply with ChunkExtraction.model_validate.
#   - on pydantic ValidationError: send the error text back and retry (max 2 retries),
#     then give up on that chunk and log it.



load_dotenv()  # read .env file for ANTHROPIC_API_KEY
MODEL = "claude-sonnet-5-5"
MAX_RETRIES = 2
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment (loaded from .env above)


def call_llm(prompt: str) -> ChunkExtraction | None:
    schema = json.dumps(ChunkExtraction.model_json_schema())
    messages = [{
        "role": "user",
        "content": prompt + "\n\nReply with ONLY a JSON object matching this JSON schema, no other text:\n" + schema,
    }]
    for attempt in range(MAX_RETRIES + 1):
        resp = client.messages.create(model=MODEL, max_tokens=5096, messages=messages)
        raw = "".join(b.text for b in resp.content if b.type == "text").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)  # strip code fences if present
        try:
            return ChunkExtraction.model_validate_json(raw)
        except ValidationError as e:
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"That failed validation. Fix it and reply with ONLY the JSON.\n{e}"},
            ]
    print("gave up on chunk after retries")
    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()
# def verify_quotes(extraction, chunk) -> ChunkExtraction
#   - for every node.res / edge.prov: normalize whitespace on both sides and check
#     supporting_quote is a substring of chunk.text.
#   - not found = hallucination -> drop that item and log it.
#   - overwrite source_file and from_section from the chunk (code knows these, don't trust the LLM).


def verify_quotes(extraction: ChunkExtraction, chunk: Chunk) -> ChunkExtraction:
    text = _norm(chunk.text)

    kept_nodes = []
    for n in extraction.nodes:
        if _norm(n.res.supporting_quote) in text:
            n.res.source_file = chunk.source_file  # code knows these, don't trust the LLM
            n.res.from_section = chunk.section
            kept_nodes.append(n)
        else:
            print(f"dropped node {n.id}: quote not in chunk")

    kept_edges = []
    for e in extraction.edges:
        if _norm(e.prov.supporting_quote) in text:
            e.prov.source_file = chunk.source_file
            e.prov.from_section = chunk.section
            kept_edges.append(e)
        else:
            print(f"dropped edge {e.id}: quote not in chunk")
    return ChunkExtraction(nodes=kept_nodes, edges=kept_edges)


LEAK_WORDS = re.compile(r"assassinat|ides of march|philippi|stabbed|murder of caesar|death of caesar", re.I)



# def leak_check(extraction) -> ChunkExtraction
#   - belt and braces on top of Graph's year check: drop items whose text/note mention
#     post-cutoff things (assassination, Ides of March, 44 BCE, Philippi, ...).

def leak_check(extraction: ChunkExtraction) -> ChunkExtraction:
    nodes = [n for n in extraction.nodes if not LEAK_WORDS.search(n.model_dump_json())]
    edges = [e for e in extraction.edges if not LEAK_WORDS.search(e.model_dump_json())]
    dropped = len(extraction.nodes) - len(nodes) + len(extraction.edges) - len(edges)
    if dropped:
        print(f"leak check dropped {dropped} item(s)")
    return ChunkExtraction(nodes=nodes, edges=edges)


GRAPH_FILE = CHUNKS_FILE.parent / "graph.json"
CACHE_DIR = CHUNKS_FILE.parent / "llm_cache"


def merge(all_extractions: list[ChunkExtraction], seeds: list[Node]) -> tuple[list[Node], list[dict]]:
    nodes = {n.id: n for n in seeds}
    alias_to_id = {a.lower(): n.id for n in seeds for a in [n.name, *n.aliases]}
    remap: dict[str, str] = {}  # id the LLM used -> id we kept

    for ex in all_extractions:
        for n in ex.nodes:
            names = [n.name, *n.aliases]
            target = n.id if n.id in nodes else next(
                (alias_to_id[a.lower()] for a in names if a.lower() in alias_to_id), None)
            if target:  # same entity: keep the first one, merge aliases
                remap[n.id] = target
                keep = nodes[target]
                keep.aliases = sorted((set(keep.aliases) | set(names)) - {keep.name})
            else:
                nodes[n.id] = n
                for a in names:
                    alias_to_id[a.lower()] = n.id

    def fix(i: str) -> str:
        return remap.get(i, i)

    edges, seen = [], set()
    for ex in all_extractions:
        for e in ex.edges:
            e = e.model_dump()
            e["participants"] = list(dict.fromkeys(fix(p) for p in e.get("participants", [])))
            e["roles"] = {fix(k): v for k, v in e.get("roles", {}).items()}
            e["drivers"] = [fix(d) for d in e.get("drivers", [])]
            for key in ("cause_node", "cause_nodes"):
                if key in e and e[key]:
                    e[key] = [fix(c) for c in e[key]]
            for key in ("side", "sides"):
                for s in e.get(key) or []:
                    s["members"] = [fix(m) for m in s.get("members", [])]
            # rule: drop edges that point at nodes that don't exist
            missing = [i for i in e["participants"] + e["drivers"] if i not in nodes]
            if missing:
                print(f"dropped edge {e.get('id')}: unknown ids {missing}")
                continue
            base, k = e.get("id", "edge"), 2  # make edge ids unique
            while e.get("id") in seen:
                e["id"] = f"{base}_{k}"
                k += 1
            seen.add(e["id"])
            edges.append(e)
    return list(nodes.values()), edges


# def merge(all_extractions, seeds) -> tuple[list[Node], list[Hyperedge]]
#   - start from the seed stakeholders, add new nodes; same id (or matching alias) = same node,
#     keep the first one and merge aliases.
#   - make edge ids unique (add a counter or hash if two chunks produce the same id).
#   - drop edges whose participants/drivers/cause nodes don't exist (or create stub nodes, pick one rule).


# def build_graph(nodes, edges) -> Graph
#   - Graph(nodes=..., edges=...) runs every cross-check (duplicates, dangling ids,
#     drivers must be goal/value, cutoff year). If it raises, print which item failed.


# def main()
#   - load_chunks -> seed_stakeholders -> for each chunk: build_prompt, call_llm, verify_quotes,
#     leak_check -> merge -> build_graph -> save to data/processed/graph.json.
#   - print a summary: nodes, edges, dropped items with reasons (this is how you tune the prompt).
#   - tip: cache each chunk's raw LLM reply on disk so re-runs don't cost money.
def build_graph(nodes: list[Node], edges: list[dict]) -> Graph:
    adapter = TypeAdapter(Hyperedge)

    def too_late(w) -> bool:
        return w is not None and w.year is not None and w.year > CUTOFF_YEAR

    # clean up what Graph would reject, and log it, so one bad item doesn't block everything
    nodes = [n for n in nodes if not too_late(n.when) or print(f"dropped node {n.id}: after cutoff")]
    by_id = {n.id: n for n in nodes}

    good = []
    for e in edges:
        try:
            e = adapter.validate_python(e)
        except ValidationError as err:
            print(f"dropped edge {e.get('id')}: {err.errors()[0]['msg']}")
            continue
        if too_late(e.when):
            print(f"dropped edge {e.id}: after cutoff")
            continue
        if any(p not in by_id for p in e.participants):
            print(f"dropped edge {e.id}: participant missing")
            continue
        # drivers must exist and be goal/value nodes; drop just the bad driver, keep the edge
        bad = [d for d in e.drivers if d not in by_id or by_id[d].kind not in ("goal", "value")]
        if bad:
            print(f"edge {e.id}: removed invalid drivers {bad}")
            e.drivers = [d for d in e.drivers if d not in bad]
        if isinstance(e, Rivalry):
            e.cause_nodes = [c for c in e.cause_nodes if c in by_id]
        good.append(e)
    return Graph(nodes=nodes, edges=good)  # final safety net


def main() -> None:
    chunks = load_chunks()
    seeds = seed_stakeholders(chunks)
    known = [n.id for n in seeds]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    extractions = []
    for c in chunks:
        cache = CACHE_DIR / f"{Path(c.source_file).stem}__{c.section}.json"
        if cache.exists():  # re-runs cost nothing
            ex = ChunkExtraction.model_validate_json(cache.read_text(encoding="utf-8"))
        else:
            ex = call_llm(build_prompt(c, known))
            if ex is None:
                continue
            cache.write_text(ex.model_dump_json(indent=2), encoding="utf-8")
        ex = leak_check(verify_quotes(ex, c))
        extractions.append(ex)
        known = sorted(set(known) | {n.id for n in ex.nodes})
        print(f"{c.source_file} / {c.section}: {len(ex.nodes)} nodes, {len(ex.edges)} edges")

    nodes, edges = merge(extractions, seeds)
    graph = build_graph(nodes, edges)
    GRAPH_FILE.write_text(graph.model_dump_json(indent=2), encoding="utf-8")
    print(f"saved {len(graph.nodes)} nodes, {len(graph.edges)} edges to {GRAPH_FILE}")


if __name__ == "__main__":
    main()
