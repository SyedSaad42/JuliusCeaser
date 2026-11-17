"""The data pipeline: chunker output and the schema rules that keep the graph honest."""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import chunker
import schema

ROOT = Path(__file__).resolve().parent.parent
P = {"source_file": "x.txt", "supporting_quote": "a long enough quote"}


def test_chunker_splits_every_raw_file_into_known_sections():
    for path in sorted((ROOT / "data" / "raw").glob("*.txt")):
        chunks = chunker.chunk_file(path)
        assert chunks, path.name
        assert all(c.section is not None for c in chunks), f"{path.name}: unmapped section title"


def test_chunker_keeps_the_casca_warning():
    casca = chunker.chunk_file(ROOT / "data" / "raw" / "05_casca.txt")
    assert casca and all(c.warning for c in casca)


def test_dates_without_a_year_must_be_vague():
    with pytest.raises(ValidationError):
        schema.When(month=3, precision="day")
    assert schema.When(precision="relative").year is None


def test_edges_reject_bad_participants_and_roles():
    base = dict(id="e1", when={}, prov=P)
    with pytest.raises(ValidationError):
        schema.Alliance(participants=["a", "a"], **base)
    with pytest.raises(ValidationError):
        schema.Alliance(participants=["a", "b"], roles={"z": "role"}, **base)


def test_rivalry_sides_must_be_participants():
    sides = [{"members": ["a"], "grievance": "g"}, {"members": ["zzz"], "grievance": "h"}]
    with pytest.raises(ValidationError):
        schema.Rivalry(id="r1", participants=["a", "b"], sides=sides, when={}, prov=P)


def _node(i, kind="person"):
    return schema.Node(id=i, name=i, kind=kind, res=P)


def _edge(**kw):
    return schema.Alliance(id="e1", participants=["a", "b"], when=kw.pop("when", {}), prov=P, **kw)


def test_graph_rejects_dangling_links_wrong_drivers_and_the_future():
    with pytest.raises(ValidationError):
        schema.Graph(nodes=[_node("a")], edges=[_edge()])                              # b does not exist
    with pytest.raises(ValidationError):
        schema.Graph(nodes=[_node("a"), _node("b")], edges=[_edge(drivers=["a"])])     # driver is not a goal/value
    late = _edge(when={"year": -44, "precision": "year"})
    with pytest.raises(ValidationError):
        schema.Graph(nodes=[_node("a"), _node("b")], edges=[late])                     # after the 45 BCE cutoff
    ok = schema.Graph(nodes=[_node("a"), _node("b"), _node("g", "goal")], edges=[_edge(drivers=["g"])])
    assert len(ok.edges) == 1


def test_the_shipped_graph_is_valid_and_has_no_future():
    data = json.loads((ROOT / "data" / "processed" / "graph.json").read_text(encoding="utf-8"))
    graph = schema.Graph.model_validate(data)        # runs every cross-check on the real graph
    assert len(graph.nodes) > 100 and len(graph.edges) > 100
