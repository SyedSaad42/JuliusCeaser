"""
visualize.py: draw the hypergraph as an interactive HTML page.
Every hyperedge becomes a small coloured hub node linked to its participants.

usage:  python graph/visualize.py                      (edges linking 2+ main stakeholders)
        python graph/visualize.py marcus_brutus        (everything touching Brutus)
        python graph/visualize.py cassius mark_antony  (several people)
        python graph/visualize.py --all                (whole graph, will be crowded)
Hover a node or hub to see the quote, stance and roles. Drag to rearrange.
"""
import sys
import webbrowser
from html import escape
from pathlib import Path

from pyvis.network import Network

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_graph import build  # noqa: E402

OUT_FILE = Path(__file__).resolve().parent.parent / "data" / "processed" / "graph.html"

EDGE_COLOR = {"rivalry": "#e5484d", "alliance": "#30a46c", "family_tie": "#3e63dd", "career_event": "#8d8d8d"}
NODE_COLOR = {"stakeholder": "#f5a623", "person": "#7fb3ff", "group": "#b794f4", "event": "#a0a0a0",
              "power": "#4fd1c5", "goal": "#f6ad55", "value": "#fc8181"}


def pick_edges(H, focus: list[str], show_all: bool) -> list[str]:
    stakeholders = {n for n, p in H.node_props.items() if p["kind"] == "stakeholder"}
    out = []
    for eid, members in H.edges.elements.items():
        members = set(members)
        if show_all:
            out.append(eid)
        elif focus:
            if members & set(focus):
                out.append(eid)
        elif len(members & stakeholders) >= 2:
            out.append(eid)
    return out


def draw(H, edge_ids: list[str], out_file: Path = OUT_FILE) -> None:
    net = Network(height="900px", width="100%", bgcolor="#111111", font_color="white", cdn_resources="remote")
    net.barnes_hut(gravity=-6000, spring_length=140)

    used = {m for eid in edge_ids for m in H.edges.elements[eid]}
    for nid in used:  # pyvis needs nodes to exist before edges point at them
        p = H.node_props[nid]
        big = p["kind"] == "stakeholder"
        tip = f"<b>{escape(p['name'])}</b> ({p['kind']})" + (f"<br>{escape(p['note'])}" if p["note"] else "")
        net.add_node(nid, label=p["name"].split(" (")[0], shape="dot", size=28 if big else 12,
                     color=NODE_COLOR.get(p["kind"], "#cccccc"), title=tip)

    for eid in edge_ids:
        p = H.edge_props[eid]
        members = list(H.edges.elements[eid])
        tip = (f"<b>{escape(p['type'])}</b>: {escape(eid)}<br>"
               + (f"stance: {escape(p['stance'])}<br>" if p["stance"] else "")
               + (f"when: {escape(p['when']['text'])}<br>" if p["when"] and p["when"].get("text") else "")
               + f"<i>\"{escape(p['quote'])}\"</i>")
        net.add_node(f"edge::{eid}", label=" ", shape="dot", size=7, color=EDGE_COLOR[p["type"]], title=tip)
        for m in members:
            role = p["roles"].get(m, "")
            net.add_edge(f"edge::{eid}", m, color=EDGE_COLOR[p["type"]], title=escape(role))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out_file))
    print(f"{len(used)} nodes, {len(edge_ids)} hyperedges -> {out_file}")


def main() -> None:
    args = sys.argv[1:]
    show_all = "--all" in args
    focus = [a for a in args if not a.startswith("--")]
    H = build()
    bad = [f for f in focus if f not in H.node_props]
    if bad:
        sys.exit(f"unknown node id(s): {bad}. Try one of: {[n for n, p in H.node_props.items() if p['kind'] == 'stakeholder']}")
    edge_ids = pick_edges(H, focus, show_all)
    if not edge_ids:
        sys.exit("no edges matched")
    draw(H, edge_ids)
    webbrowser.open(OUT_FILE.as_uri())


if __name__ == "__main__":
    main()
