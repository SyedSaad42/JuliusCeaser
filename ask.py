"""
ask.py: answer questions using only the hypergraph.
usage:  python ask.py "if Brutus was to go against Caesar, what can stop him?"
        python ask.py            (interactive)
        python ask.py --simple "..."   (old one-shot mode: code picks the edges, no tool calls)

Default mode is an agent: the LLM calls tools (find_node, get_edges, get_edge, neighbours,
connection) to search and walk the graph itself, so it can follow links across several hops.
"""
import json
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


def ask_simple(question: str, H) -> str:  # old one-shot mode, kept for comparison (--simple)
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


# ----------------------------------------------------------------------------
# AGENT MODE: the LLM calls these tools to search and walk the graph (multi-hop).
# ----------------------------------------------------------------------------
MAX_STEPS = 8            # most tool-calling rounds per question (keeps cost bounded)
MAX_RESULT_CHARS = 6000  # each tool result is cut to this size

AGENT_SYSTEM = """You answer questions about the world of Rome by exploring a knowledge graph with tools.

HOW TO WORK
- Start with find_node to get node ids (names are not ids). If a search finds nothing, try another spelling or one word.
- Use get_edges / get_edge / neighbours to read facts, and connection to see how two people are linked. Follow links across several hops when the question needs it.
- Make only the calls you need (at most about 8), then answer.

RULES
- Use ONLY facts returned by the tools. No outside or historical knowledge.
- The knowledge cutoff is 45 BCE. Never mention or imply Caesar's death or anything after the cutoff. Treat the future as unknown.
- Reason about motives (drivers), conditions, stance, roles and rivalries/alliances to reach a judgement.
- Cite the edge ids you relied on in square brackets, e.g. [cassius_sided_with_pompey_senate].
- If the tools show the facts are missing, say "not enough information" and say what you searched for. Do not guess.
"""

STYLE_RULES = """
NO OUTSIDE KNOWLEDGE (most important rule)
- You may know how this story ends. Pretend you do not. Use nothing but facts returned by the tools.
- The world is the end of 45 BCE and Caesar is alive and in power. Never mention, hint at or foreshadow any plot against him, his death, the Ides of March, Philippi, or anything after 45 BCE.
- Do not use words like assassination, conspirators or "before his death".
- If a question assumes something the graph does not record (e.g. "why did X kill Y"), say the graph records no such thing. For "what if" questions, reason only from the graph's motives and conditions, as a hypothetical.

ANSWER FORMAT (use exactly this layout, markdown)
**Answer:** one or two sentences.

**Key points**
- One statement per bullet. Put its sources at the very END of the bullet, after the final full stop, in one bracket: "... statement. [edge_id_1, edge_id_2]". Never put a citation in the middle of a sentence.

**What the graph does not say**
- Bullets for gaps or things you could not find. Write "none" if there are no gaps.
"""
SYSTEM += STYLE_RULES
AGENT_SYSTEM += STYLE_RULES

# last line of defence: if the answer still contains outside knowledge, it is rewritten or withheld
LEAK = re.compile(
    r"assassinat|ides of march|philippi|stabb|death of caesar|caesar'?s (death|murder|killing)|"
    r"(kill|murder|slay|slew|slain|stab)\w*\s+(julius\s+)?caesar|caesar\s+(was|is)\s+(killed|murdered|slain)|"
    r"brutus\s+(killed|murdered|stabbed)",
    re.I)


def guard(messages: list, text: str) -> str:
    for _ in range(2):
        if not LEAK.search(text):
            return text
        print("  !! answer used outside knowledge, asking for a rewrite")
        messages += [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "Your answer used knowledge from outside the graph (events after 45 BCE). "
             "Rewrite it using ONLY facts from the tool results, in the same format. Treat Caesar as alive and in power."},
        ]
        resp = client.messages.create(model=MODEL, max_tokens=2000, system=AGENT_SYSTEM, messages=messages,
                                      tools=TOOLS, tool_choice={"type": "none"})
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return text if not LEAK.search(text) else "Answer withheld: it relied on knowledge outside the graph. Try rephrasing the question."


TOOLS = [
    {"name": "find_node",
     "description": "Search nodes (people, groups, powers, goals, events) by name, alias or id. Returns node ids you can use in other tools.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "get_edges",
     "description": "List the edges (alliances, rivalries, family ties, career events) a node takes part in, with a short summary each.",
     "input_schema": {"type": "object", "properties": {
         "node_id": {"type": "string"},
         "type": {"type": "string", "enum": ["rivalry", "alliance", "family_tie", "career_event"], "description": "optional filter"},
         "limit": {"type": "integer", "description": "default 15"}}, "required": ["node_id"]}},
    {"name": "get_edge",
     "description": "Full detail of one edge: roles, stance, drivers, conditions, rivalry sides, date and the source quote.",
     "input_schema": {"type": "object", "properties": {"edge_id": {"type": "string"}}, "required": ["edge_id"]}},
    {"name": "neighbours",
     "description": "Nodes that share at least one edge with this node, with how many edges they share.",
     "input_schema": {"type": "object", "properties": {"node_id": {"type": "string"}}, "required": ["node_id"]}},
    {"name": "connection",
     "description": "Shortest chain of shared edges linking two nodes (multi-hop).",
     "input_schema": {"type": "object", "properties": {"node_a": {"type": "string"}, "node_b": {"type": "string"}}, "required": ["node_a", "node_b"]}},
]

TYPE_RANK = {"rivalry": 0, "alliance": 1, "family_tie": 2, "career_event": 3}


def adjacency(H) -> dict:
    if not hasattr(H, "adj"):  # node -> {other node -> [shared edge ids]}
        H.adj = {}
        for eid, members in H.edges.elements.items():
            for a in members:
                for b in members:
                    if a != b:
                        H.adj.setdefault(a, {}).setdefault(b, []).append(eid)
    return H.adj


def tool_find_node(H, query: str) -> str:
    words = re.findall(r"[a-z0-9]+", query.lower())
    hits = []
    for nid, p in H.node_props.items():
        hay = " ".join([nid.replace("_", " "), p["name"].lower(), *[a.lower() for a in p["aliases"]]])
        score = sum(w in hay for w in words)
        if score:
            hits.append((-score, nid))
    if not hits:
        return f"no node matches '{query}'"
    return "\n".join(f"{nid} | {H.node_props[nid]['name']} | {H.node_props[nid]['kind']} | {(H.node_props[nid]['note'] or '')[:100]}"
                     for _, nid in sorted(hits)[:8])


def tool_get_edges(H, node_id: str, type: str | None = None, limit: int = 15) -> str:
    if node_id not in H.node_props:
        return f"unknown node id '{node_id}' (use find_node first)"
    eids = [e for e in edges_of(H, node_id) if not type or H.edge_props[e]["type"] == type]
    eids.sort(key=lambda e: TYPE_RANK[H.edge_props[e]["type"]])
    lines = [f"[{e}] {H.edge_props[e]['type']} | {', '.join(name(H, m) for m in H.edges.elements[e])} | {(H.edge_props[e]['stance'] or '')[:140]}"
             for e in eids[:limit]]
    return f"{len(eids)} edges, showing {len(lines)}:\n" + "\n".join(lines) if lines else "no edges"


def tool_get_edge(H, edge_id: str) -> str:
    return edge_line(H, edge_id) if edge_id in H.edge_props else f"unknown edge id '{edge_id}'"


def tool_neighbours(H, node_id: str) -> str:
    near = adjacency(H).get(node_id)
    if not near:
        return "no neighbours (unknown node or it is in no edge)"
    ranked = sorted(near.items(), key=lambda kv: -len(kv[1]))[:25]
    return "\n".join(f"{n} | {name(H, n)} | {H.node_props[n]['kind']} | shared edges: {len(es)}" for n, es in ranked)


def tool_connection(H, node_a: str, node_b: str) -> str:
    adj = adjacency(H)
    if node_a not in adj or node_b not in adj:
        return "one of the nodes is unknown or in no edge"
    prev, frontier = {node_a: None}, [node_a]
    while frontier and node_b not in prev:
        nxt = []
        for cur in frontier:
            for other, es in adj[cur].items():
                if other not in prev:
                    prev[other] = (cur, es[0])
                    nxt.append(other)
        frontier = nxt
    if node_b not in prev:
        return "no connection"
    path, cur = [], node_b
    while prev[cur]:
        cur_prev, eid = prev[cur]
        path.append(f"{name(H, cur_prev)} -[{eid}]-> {name(H, cur)}")
        cur = cur_prev
    return f"{len(path)} hop(s):\n" + "\n".join(reversed(path))


TOOL_FUNCS = {"find_node": tool_find_node, "get_edges": tool_get_edges, "get_edge": tool_get_edge,
              "neighbours": tool_neighbours, "connection": tool_connection}


def run_tool(H, tool: str, args: dict) -> str:
    try:
        return TOOL_FUNCS[tool](H, **args)[:MAX_RESULT_CHARS]
    except Exception as e:  # a bad call goes back to the model so it can correct itself
        return f"tool error: {type(e).__name__}: {e}"


def ask(question: str, H) -> str:
    messages = [{"role": "user", "content": question}]
    for step in range(MAX_STEPS + 1):
        last = step == MAX_STEPS  # out of rounds: force a final text answer
        try:
            resp = client.messages.create(
                model=MODEL, max_tokens=2000, system=AGENT_SYSTEM, messages=messages, tools=TOOLS,
                **({"tool_choice": {"type": "none"}} if last else {}),
            )
        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            return f"Couldn't reach the API ({type(e).__name__}). Check your internet/VPN and try again."
        if resp.stop_reason != "tool_use":
            return guard(messages, "".join(b.text for b in resp.content if b.type == "text").strip())
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in resp.content:
            if b.type == "tool_use":
                print(f"  -> {b.name}({json.dumps(b.input)})")
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": run_tool(H, b.name, b.input)})
        messages.append({"role": "user", "content": results})
    return "stopped: too many steps"


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--simple"]
    answer = ask_simple if "--simple" in sys.argv else ask
    H = build()
    if args:
        print("\n" + answer(" ".join(args), H))
        return
    while True:
        q = input("\nquestion (blank to quit): ").strip()
        if not q:
            break
        print("\n" + answer(q, H))


if __name__ == "__main__":
    main()
