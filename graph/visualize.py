"""
visualize.py: draw the hypergraph as an interactive HTML page.
Every hyperedge becomes a small coloured hub node linked to its participants.

usage:  python graph/visualize.py                      (edges linking 2+ main stakeholders)
        python graph/visualize.py marcus_brutus        (everything touching Brutus)
        python graph/visualize.py cassius mark_antony  (several people)
        python graph/visualize.py --all                (whole graph, will be crowded)
Hover a node or hub to see the quote, stance and roles. Drag to rearrange.
"""
import json
import re
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


OPTIONS = {
    "nodes": {"font": {"face": "Parkinsans, system-ui, sans-serif", "strokeWidth": 3, "strokeColor": "#0b1020"}},
    "interaction": {"hover": True, "tooltipDelay": 0},  # scroll to zoom, drag to move (no on-screen buttons)
    "physics": {"solver": "barnesHut",
                "barnesHut": {"gravitationalConstant": -6000, "springLength": 140, "damping": 0.35},
                "stabilization": {"iterations": 200}},
}

# page tweaks so the graph fills whatever frame it is shown in (the web app embeds it in an iframe)
HEAD_INJECT = """
<link href="https://fonts.googleapis.com/css2?family=Parkinsans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  html, body { height: 100%; margin: 0; background: #0b1020; }
  .card { border: 0 !important; background: #0b1020 !important; height: 100%; }
  #mynetwork { height: 100vh !important; border: 0 !important; float: none !important; }
  #loadingBar { display: none !important; }
  div.vis-tooltip { background: #0a0e1c; color: #e8ecf6; border: 1px solid #2a3358; border-radius: 10px;
    padding: 10px 12px; max-width: 340px; white-space: normal; font: 13px/1.45 Parkinsans, system-ui, sans-serif; }
</style>
"""

# vis-network 9 shows string tooltips as plain text, so turn our HTML strings into elements.
TAIL_SCRIPT = """
<script>
(function () {
  function toEl(h) { var d = document.createElement("div"); d.innerHTML = h; return d; }
  nodes.get().forEach(function (n) { if (typeof n.title === "string") nodes.update({id: n.id, title: toEl(n.title)}); });
  edges.get().forEach(function (e) { if (typeof e.title === "string" && e.title) edges.update({id: e.id, title: toEl(e.title)}); });
  network.once("stabilizationIterationsDone", function () {
    network.setOptions({physics: false});
    network.fit({animation: true});
    // tell the page that embeds this one that the graph is finished
    try { parent.postMessage({type: "graph-ready"}, "*"); } catch (e) {}
  });
})();
</script>
"""


def clean_name(name: str) -> str:
    # seeded names went through str.title(), which gives "Caesar'S Veterans And Army"
    name = name.split(" (")[0].replace("'S ", "'s ")
    return re.sub(r" (And|Of|The)\b", lambda m: m.group(0).lower(), name)


def build_net(H, edge_ids: list[str], cited: list[str] | None = None, nodes: list[str] | None = None) -> Network:
    """cited = edge ids used in an answer (ringed in white); nodes = nodes the agent looked at (blue ring).
    Edges shown but not cited were examined. cited=None means a normal view with no emphasis."""
    net = Network(height="900px", width="100%", bgcolor="#0b1020", font_color="white", cdn_resources="remote")
    net.set_options(json.dumps(OPTIONS))

    emphasis = cited is not None
    cited_set = set(cited or [])
    cited_people = {m for e in cited_set if e in H.edge_props for m in H.edges.elements[e]}
    looked_at = [n for n in (nodes or []) if n in H.node_props]
    used = {m for eid in edge_ids for m in H.edges.elements[eid]} | set(looked_at)
    for nid in used:  # pyvis needs nodes to exist before edges point at them
        p = H.node_props[nid]
        big = p["kind"] == "stakeholder"
        base = NODE_COLOR.get(p["kind"], "#cccccc")
        tip = f"<b>{escape(p['name'])}</b> ({p['kind']})" + (f"<br>{escape(p['note'])}" if p["note"] else "")
        style = {"color": base}
        if emphasis and nid in cited_people:
            style = {"borderWidth": 4, "color": {"background": base, "border": "#ffffff"}}
        elif emphasis and nid in looked_at:
            style = {"borderWidth": 4, "color": {"background": base, "border": "#7c9cff"}}
        net.add_node(nid, label=clean_name(p["name"]), shape="dot", size=(28 if big else 12) + (4 if len(style) > 1 else 0),
                     title=tip, **style)

    for eid in edge_ids:
        p = H.edge_props[eid]
        members = list(H.edges.elements[eid])
        tip = (f"<b>{escape(p['type'])}</b>: {escape(eid)}<br>"
               + (f"stance: {escape(p['stance'])}<br>" if p["stance"] else "")
               + (f"when: {escape(p['when']['text'])}<br>" if p["when"] and p["when"].get("text") else "")
               + f"<i>\"{escape(p['quote'])}\"</i>")
        col = EDGE_COLOR[p["type"]]
        used_in_answer = emphasis and eid in cited_set
        net.add_node(f"edge::{eid}", label=" ", shape="dot", size=11 if used_in_answer else 7, title=tip,
                     borderWidth=3 if used_in_answer else 1,
                     color={"background": col, "border": "#ffffff" if used_in_answer else col})
        for m in members:
            role = p["roles"].get(m, "")
            net.add_edge(f"edge::{eid}", m, color=col, title=escape(role),
                         width=3 if used_in_answer else (1.6 if emphasis else 1))

    return net


def render_html(H, edge_ids: list[str], select: list[str] | None = None, nodes: list[str] | None = None) -> str:
    """The finished page as a string. `select` = edge ids used in an answer (ringed), `nodes` = nodes the agent visited."""
    html = build_net(H, edge_ids, cited=select, nodes=nodes).generate_html()
    html = html.replace("</head>", HEAD_INJECT + "</head>", 1)
    return html.replace("</body>", TAIL_SCRIPT + "</body>", 1)


def draw(H, edge_ids: list[str], out_file: Path = OUT_FILE) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(render_html(H, edge_ids), encoding="utf-8")
    nodes = {m for e in edge_ids for m in H.edges.elements[e]}
    print(f"{len(nodes)} nodes, {len(edge_ids)} hyperedges -> {out_file}")


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
