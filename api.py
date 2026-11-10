"""
api.py: FastAPI interface over the hypergraph and the question-answering agent.

run:   uvicorn api:app --reload
open:  http://127.0.0.1:8000        (one-page app: chat + interactive graph)
       http://127.0.0.1:8000/docs   (interactive API docs)

NOTE: POST /ask calls the Anthropic API and costs money. There is no login, so keep it
bound to localhost (the default) and don't expose it publicly as is.
"""
import re
import sys
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import ask as agent  # noqa: E402  (ask.py: tools, agent loop, edge formatting)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.H = agent.build()  # load graph.json once at startup
    app.state.degree = Counter(m for members in app.state.H.edges.elements.values() for m in members)
    yield


app = FastAPI(title="Julius Caesar world graph", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: Literal["agent", "simple"] = "agent"


class ToolCall(BaseModel):
    tool: str
    args: dict


class AskResponse(BaseModel):
    answer: str
    mode: str
    tool_calls: list[ToolCall] = []


@app.get("/health")
def health():
    H = app.state.H
    return {"status": "ok", "nodes": len(H.node_props), "edges": len(H.edge_props)}


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    """Answer a question. agent = the LLM explores the graph with tools (multi-hop); simple = one-shot."""
    H = app.state.H
    if req.mode == "simple":
        return AskResponse(answer=agent.ask_simple(req.question, H), mode="simple")
    trace: list[dict] = []
    answer = agent.ask(req.question, H, trace=trace)
    return AskResponse(answer=answer, mode="agent", tool_calls=trace)


@app.get("/nodes")
def list_nodes(
    kind: str | None = Query(None, description="stakeholder, person, group, event, power, goal, value"),
    q: str | None = Query(None, description="text to search in name, alias or id"),
    limit: int = Query(50, ge=1, le=300),
):
    H = app.state.H
    out = []
    for nid, p in H.node_props.items():
        if kind and p["kind"] != kind:
            continue
        if q and q.lower() not in " ".join([nid, p["name"], *p["aliases"]]).lower():
            continue
        out.append({"id": nid, "name": p["name"], "kind": p["kind"]})
    return out[:limit]


@app.get("/nodes/{node_id}")
def get_node(node_id: str):
    H = app.state.H
    if node_id not in H.node_props:
        raise HTTPException(404, f"unknown node '{node_id}'")
    edges = [
        {"id": e, "type": H.edge_props[e]["type"], "participants": list(H.edges.elements[e]),
         "stance": H.edge_props[e]["stance"]}
        for e in agent.edges_of(H, node_id)
    ]
    return {"id": node_id, **H.node_props[node_id], "edges": edges}


@app.get("/edges/{edge_id}")
def get_edge(edge_id: str):
    H = app.state.H
    if edge_id not in H.edge_props:
        raise HTTPException(404, f"unknown edge '{edge_id}'")
    return {"id": edge_id, "participants": list(H.edges.elements[edge_id]), **H.edge_props[edge_id]}


@app.get("/connection")
def connection(a: str, b: str):
    """Shortest chain of shared edges between two nodes."""
    H = app.state.H
    for n in (a, b):
        if n not in H.node_props:
            raise HTTPException(404, f"unknown node '{n}'")
    return {"a": a, "b": b, "path": agent.tool_connection(H, a, b)}


@app.get("/graph")
def graph(
    ids: str | None = Query(None, description="comma-separated edge ids (e.g. the ones an answer cited)"),
    focus: str | None = Query(None, description="comma-separated node ids: every edge touching them"),
    all: bool = Query(False, description="the whole graph (crowded)"),
):
    """Nodes and edges for the graph view. Default: edges that link 2+ of the main stakeholders."""
    H = app.state.H
    if ids:
        wanted = [i.strip() for i in ids.split(",")]
        edge_ids = [i for i in wanted if i in H.edge_props]
    elif focus:
        want = {i.strip() for i in focus.split(",")}
        edge_ids = [e for e, ms in H.edges.elements.items() if want & set(ms)]
    elif all:
        edge_ids = list(H.edges.elements)
    else:
        main = {n for n, p in H.node_props.items() if p["kind"] == "stakeholder"}
        edge_ids = [e for e, ms in H.edges.elements.items() if len(main & set(ms)) >= 2]

    def nm(n: str) -> str:
        # seeded names went through str.title(), which gives "Caesar'S Veterans And Army"
        name = H.node_props[n]["name"].split(" (")[0].replace("'S ", "'s ")
        return re.sub(r" (And|Of|The)\b", lambda m: m.group(0).lower(), name)

    nodes, edges = {}, []
    for eid in edge_ids:
        p, members = H.edge_props[eid], list(H.edges.elements[eid])
        for m in members:
            if m not in nodes:
                np_ = H.node_props[m]
                nodes[m] = {"id": m, "name": nm(m), "kind": np_["kind"], "note": np_["note"],
                            "edges": app.state.degree[m]}
        edges.append({
            "id": eid, "type": p["type"], "stance": p["stance"], "quote": p["quote"],
            "source": p["source_file"], "when": (p["when"] or {}).get("text"),
            "participants": [{"id": m, "role": p["roles"].get(m, "")} for m in members],
            "drivers": [nm(d) for d in p["drivers"] if d in H.node_props],
            "sides": [{"members": [nm(m) for m in s["members"]], "grievance": s["grievance"],
                       "revealed": s.get("stated_vs_revealed")} for s in (p["sides"] or [])],
            "intensity": p["intensity"] if p["intensity"] != "unknown" else None,
        })
    return {"nodes": list(nodes.values()), "edges": edges}


@app.get("/")
def home():
    return FileResponse(ROOT / "web" / "index.html")
