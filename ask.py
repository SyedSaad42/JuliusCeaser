"""
ask.py: answer questions using only the hypergraph.
usage:  python ask.py "if Brutus was to go against Caesar, what can stop him?"
        python ask.py            (interactive)
"""
import re
import sys
from pathlib import Path

import anthropic
import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "graph"))
from graph.build_graph import build, edges_of

load_dotenv()
MODEL = "claude-haiku-5-5"  # cheaper; switch back to "claude-sonnet-5-5" for harder questions
MAX_EDGES = 35
# the default connect timeout is only 5s, which a slow network's TLS handshake can exceed
client = anthropic.Anthropic(timeout=httpx.Timeout(60.0, connect=30.0), max_retries=4)

# words in a question -> stakeholder node id
KEYWORDS = {
    "julius_caesar": r"caesar|ceaser|cesar|julius",
    "marcus_brutus": r"\bbrutus\b|marcus",
    "decimus_brutus": r"decimus",
    "cassius": r"cassius",
    "mark_antony": r"antony|antonius",
    "cicero": r"cicero|tully",
    "casca": r"casca",
    "the_senate": r"senate|senator",
    "the_roman_people": r"people|plebs|citizens",
    "caesars_veterans_and_army": r"veteran|army|soldier|legion",
    "former_pompey_supporters": r"pompey|pompeian|ligarius",
}

SYSTEM = """You answer questions about the world of Rome using ONLY the graph facts provided.

RULES
- Use only the facts below. No outside or historical knowledge.
- The knowledge cutoff is 45 BCE. Never mention or imply Caesar's death or anything after the cutoff. Treat the future as unknown.
- Reason about motives (drivers), conditions, stance, roles and rivalries/alliances to reach a judgement.
- Cite the edge ids you relied on in square brackets, e.g. [cassius_sided_with_pompey_senate].
- If the facts are too thin to answer, say "not enough information" and say what is missing. Do not guess.
"""


def find_nodes(question: str, H) -> list[str]:
    q = question.lower()
    found = [nid for nid, pat in KEYWORDS.items() if re.search(pat, q) and nid in H.node_props]
    if "decimus" in q and "marcus_brutus" in found:  # "Decimus Brutus" is not Marcus
        found.remove("marcus_brutus")
    return found


def name(H, nid: str) -> str:
    return H.node_props.get(nid, {}).get("name", nid)


def edge_line(H, eid: str) -> str:
    p = H.edge_props[eid]
    members = H.edges.elements[eid]
    who = "; ".join(f"{name(H, m)} ({p['roles'].get(m, 'participant')})" for m in members)
    parts = [f"[{eid}] {p['type']} | {who}"]
    if p["stance"]:
        parts.append(f"stance: {p['stance']}")
    if p["drivers"]:
        parts.append("drivers: " + ", ".join(name(H, d) for d in p["drivers"]))
    if p["conditions"]:
        parts.append("conditions: " + "; ".join(p["conditions"]))
    if p["sides"]:
        parts.append("sides: " + " | ".join(
            f"{', '.join(name(H, m) for m in s['members'])} - {s['grievance']}"
            + (f" (publicly vs really: {s['stated_vs_revealed']})" if s.get("stated_vs_revealed") else "")
            for s in p["sides"]))
    if p["intensity"] and p["intensity"] != "unknown":
        parts.append(f"intensity: {p['intensity']}, status: {p['status']}")
    if p["when"] and p["when"].get("text"):
        parts.append(f"when: {p['when']['text']}")
    parts.append(f"quote: \"{p['quote']}\"")
    return " | ".join(parts)


def build_context(H, node_ids: list[str]) -> tuple[str, list[str]]:
    scored = {}
    for nid in node_ids:
        for eid in edges_of(H, nid):
            members = set(H.edges.elements[eid])
            hits = len(members & set(node_ids))
            kind_rank = {"rivalry": 0, "alliance": 1, "family_tie": 2, "career_event": 3}[H.edge_props[eid]["type"]]
            scored[eid] = (-hits, kind_rank)  # edges touching more of the asked nodes come first
    chosen = sorted(scored, key=scored.get)[:MAX_EDGES]
    lines = [edge_line(H, e) for e in chosen]
    heads = [f"- {name(H, n)}: {H.node_props[n]['note'] or ''}".rstrip(": ") for n in node_ids]
    return "PEOPLE/GROUPS ASKED ABOUT:\n" + "\n".join(heads) + "\n\nFACTS (graph edges):\n" + "\n".join(lines), chosen


def ask(question: str, H) -> str:
    node_ids = find_nodes(question, H)
    if not node_ids:
        return "I couldn't tell which stakeholder you mean. Mention one by name (e.g. Brutus, Cassius, Antony, the Senate)."
    context, used = build_context(H, node_ids)
    print(f"[using {len(used)} edges about: {', '.join(node_ids)}]")
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM,
            messages=[{"role": "user", "content": f"{context}\n\nQUESTION: {question}"}],
        )
    except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
        return f"Couldn't reach the API ({type(e).__name__}). Check your internet/VPN and try again."
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def main() -> None:
    H = build()
    if len(sys.argv) > 1:
        print("\n" + ask(" ".join(sys.argv[1:]), H))
        return
    while True:
        q = input("\nquestion (blank to quit): ").strip()
        if not q:
            break
        print("\n" + ask(q, H))


if __name__ == "__main__":
    main()
