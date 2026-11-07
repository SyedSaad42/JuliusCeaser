"""
build_graph.py: turn data/processed/graph.json into a HyperNetX hypergraph.
Each node becomes a hypergraph node, each edge (alliance, rivalry, ...) becomes a hyperedge
over its participants, and all the extra fields are kept as properties.
"""
import json
from pathlib import Path

import hypernetx as hnx

GRAPH_FILE = Path(__file__).resolve().parent.parent / "data" / "processed" / "graph.json"


def load_graph_json(path: str | Path = GRAPH_FILE) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found, run `python src/extract.py` first")
    return json.loads(path.read_text(encoding="utf-8"))


def build_hypergraph(data: dict) -> hnx.Hypergraph:
    # every edge: id -> list of participant node ids
    incidence = {e["id"]: e["participants"] for e in data["edges"]}

    # properties kept on the nodes and hyperedges, so questions can use them later
    node_props = {
        n["id"]: {
            "name": n["name"],
            "kind": n["kind"],
            "aliases": n.get("aliases", []),
            "note": n.get("note"),
            "when": n.get("when"),
            "quote": n["res"]["supporting_quote"],
            "source_file": n["res"]["source_file"],
        }
        for n in data["nodes"]
    }
    edge_props = {
        e["id"]: {
            "type": e["type"],
            "roles": e.get("roles", {}),
            "stance": e.get("stance"),
            "drivers": e.get("drivers", []),
            "conditions": e.get("conditions", []),
            "when": e.get("when"),
            "quote": e["prov"]["supporting_quote"],
            "source_file": e["prov"]["source_file"],
            # rivalry only
            "sides": e.get("side") or e.get("sides"),
            "intensity": e.get("intensity"),
            "status": e.get("status"),
        }
        for e in data["edges"]
    }

    H = hnx.Hypergraph(incidence)
    # attach the properties; nodes with no edge are not in H.nodes, but stay in node_props
    H.node_props = node_props
    H.edge_props = edge_props
    return H


def edges_of(H: hnx.Hypergraph, node_id: str) -> list[str]:
    """Ids of every hyperedge this node takes part in."""
    return [eid for eid, members in H.edges.elements.items() if node_id in members]


def neighbours(H: hnx.Hypergraph, node_id: str) -> set[str]:
    """Every node that shares at least one hyperedge with node_id."""
    out: set[str] = set()
    for eid in edges_of(H, node_id):
        out |= set(H.edges.elements[eid])
    out.discard(node_id)
    return out


def build(path: str | Path = GRAPH_FILE) -> hnx.Hypergraph:
    H = build_hypergraph(load_graph_json(path))
    print(f"hypergraph: {len(H.node_props)} nodes ({len(H.nodes)} in an edge), {len(H.edges)} hyperedges")
    return H


if __name__ == "__main__":
    H = build()
    for nid in ("marcus_brutus", "cassius"):
        if nid in H.node_props:
            print(nid, "->", sorted(neighbours(H, nid)))
