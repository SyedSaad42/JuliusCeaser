import types

import pytest
from fastapi.testclient import TestClient

import api


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app, raise_server_exceptions=False) as c:
        yield c


def test_health(client):
    d = client.get("/health").json()
    assert d["status"] == "ok" and d["nodes"] > 100 and d["edges"] > 100
    assert d["assistant_ready"] is True


def test_pages(client):
    home = client.get("/")
    assert home.status_code == 200 and "Caesar's Rome" in home.text
    view = client.get("/graph-view")
    assert view.status_code == 200 and "graph-ready" in view.text
    assert client.get("/assets/me.png").status_code == 200


def test_graph_view_variants(client):
    assert client.get("/graph-view?focus=casca").status_code == 200
    assert client.get("/graph-view?ids=alliance_antony_caesar").status_code == 200
    assert client.get("/graph-view?ids=does_not_exist").status_code == 404


def test_nodes_and_edges(client):
    stake = client.get("/nodes", params={"kind": "stakeholder"}).json()
    assert len(stake) == 11
    node = client.get("/nodes/marcus_brutus").json()
    assert node["edges"], "Brutus should be in some edges"
    edge = client.get("/edges/" + node["edges"][0]["id"]).json()
    assert edge["quote"] and edge["participants"]
    assert client.get("/nodes/nobody").status_code == 404
    assert client.get("/edges/nothing").status_code == 404


def test_connection(client):
    d = client.get("/connection", params={"a": "cassius", "b": "mark_antony"}).json()
    assert "hop" in d["path"]
    assert client.get("/connection", params={"a": "x", "b": "cassius"}).status_code == 404


def test_ask_validation(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422          # too short
    assert client.post("/ask", json={"question": "x" * 600}).status_code == 422     # too long
    assert client.post("/ask", json={"question": "who is cassius?", "mode": "bad"}).status_code == 422


def test_ask_returns_trace_and_citations(client, monkeypatch):
    def fake(question, H, trace=None):
        trace.append({"tool": "get_edge", "args": {"edge_id": "alliance_antony_caesar"},
                      "edges": ["alliance_antony_caesar"]})
        trace.append({"tool": "connection", "args": {"node_a": "cassius", "node_b": "mark_antony"},
                      "edges": ["alliance_cassius_caesar"]})
        return "**Answer:** x\n- A point. [alliance_antony_caesar, made_up_id]"

    monkeypatch.setattr(api.agent, "ask", fake)
    d = client.post("/ask", json={"question": "who is antony?"}).json()
    assert d["cited_edges"] == ["alliance_antony_caesar"]           # made-up ids are dropped
    assert d["visited_edges"] == ["alliance_cassius_caesar"]        # examined but not cited
    assert d["visited_nodes"] == ["cassius", "mark_antony"]


def test_ask_out_of_credits_sends_one_alert(client, monkeypatch):
    sent = []
    monkeypatch.setenv("GITHUB_DISPATCH_TOKEN", "ghp_fake")
    monkeypatch.setattr(api, "_last_alert", 0.0)
    monkeypatch.setattr(api.httpx, "post",
                        lambda url, **kw: sent.append(url) or types.SimpleNamespace(status_code=204))

    def broke(question, H, trace=None):
        raise api.agent.OutOfCredits("credit balance is too low")

    monkeypatch.setattr(api.agent, "ask", broke)
    for _ in range(2):
        r = client.post("/ask", json={"question": "who is cassius?"})
        assert r.status_code == 402 and "broke" in r.json()["detail"]
    assert len(sent) == 1 and sent[0].endswith("/dispatches")       # the second call is inside the cooldown


def test_ask_without_key_is_a_clean_503(client, monkeypatch):
    def unconfigured(question, H, trace=None):
        raise api.agent.NotConfigured("no key")

    monkeypatch.setattr(api.agent, "ask", unconfigured)
    r = client.post("/ask", json={"question": "who is cassius?"})
    assert r.status_code == 503 and "not set up" in r.json()["detail"]
