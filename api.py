"""
api.py: FastAPI interface over the hypergraph and the question-answering agent.

run:   uvicorn api:app --reload
open:  http://127.0.0.1:8000        (one-page app: chat + interactive graph)
       http://127.0.0.1:8000/docs   (interactive API docs)

NOTE: POST /ask calls the Anthropic API and costs money. There is no login, so keep it
bound to localhost (the default) and don't expose it publicly as is.

If the Anthropic account runs out of credits, /ask answers 402 with a friendly message, and
(once per hour) a GitHub `repository_dispatch` fires, which the workflow
.github/workflows/out-of-credits.yml turns into an issue that notifies the repo owner.
Needs GITHUB_DISPATCH_TOKEN (and optionally GITHUB_REPO) in the environment / .env.
"""
import os
import re
import sys
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "graph"))
import ask as agent  # noqa: E402  (ask.py: tools, agent loop, edge formatting)
import visualize as viz  # noqa: E402  (the pyvis graph page)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.H = agent.build()  # load graph.json once at startup
    app.state.degree = Counter(m for members in app.state.H.edges.elements.values() for m in members)
    yield


app = FastAPI(title="Julius Caesar world graph", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=ROOT / "web" / "assets"), name="assets")  # your photo: web/assets/me.png

NOTIFY_COOLDOWN = 3600  # seconds between GitHub alerts
_last_alert = 0.0


def alert_out_of_credits() -> None:
    """Tell GitHub, so the workflow opens an issue (which notifies the owner). Never raises."""
    global _last_alert
    if time.time() - _last_alert < NOTIFY_COOLDOWN:
        return
    _last_alert = time.time()
    token, repo = os.getenv("GITHUB_DISPATCH_TOKEN"), os.getenv("GITHUB_REPO", "SyedSaad42/JuliusCeaser")
    if not token:
        print("out of API credits, but GITHUB_DISPATCH_TOKEN is not set, so no GitHub alert was sent")
        return
    try:
        r = httpx.post(
            f"https://api.github.com/repos/{repo}/dispatches",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28"},
            json={"event_type": "out_of_credits"}, timeout=10)
        print(f"GitHub alert sent (status {r.status_code})")
    except httpx.HTTPError as e:
        print(f"could not reach GitHub: {e}")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: Literal["agent", "simple"] = "agent"


class ToolCall(BaseModel):
    tool: str
    args: dict
    edges: list[str] = []  # edges this call actually opened or walked


class AskResponse(BaseModel):
    answer: str
    mode: str
    tool_calls: list[ToolCall] = []
    cited_edges: list[str] = []    # edge ids named in the answer: what it is based on
    visited_edges: list[str] = []  # other edges the agent examined on the way
    visited_nodes: list[str] = []  # nodes the agent looked up


def cited_ids(H, answer: str) -> list[str]:
    """Edge ids in the answer's [bracketed] source lists."""
    ids = [i.strip() for group in re.findall(r"\[([^\[\]]+)\]", answer) for i in re.split(r"[,;]", group)]
    return [i for i in dict.fromkeys(ids) if i in H.edge_props]


@app.get("/health")
def health():
    H = app.state.H
    return {"status": "ok", "nodes": len(H.node_props), "edges": len(H.edge_props)}


@app.post("/ask", response_model=AskResponse, responses={402: {"description": "Out of API credits"}})
def ask_question(req: AskRequest):
    """Answer a question. agent = the LLM explores the graph with tools (multi-hop); simple = one-shot."""
    H = app.state.H
    try:
        if req.mode == "simple":
            answer = agent.ask_simple(req.question, H)
            return AskResponse(answer=answer, mode="simple", cited_edges=cited_ids(H, answer))
        trace: list[dict] = []
        answer = agent.ask(req.question, H, trace=trace)
        cited = cited_ids(H, answer)
        examined = [e for t in trace for e in t["edges"] if e not in cited]
        looked = [a for t in trace for k in ("node_id", "node_a", "node_b")
                  for a in [t["args"].get(k)] if a in H.node_props]
        return AskResponse(answer=answer, mode="agent", tool_calls=trace, cited_edges=cited,
                           visited_edges=list(dict.fromkeys(examined)), visited_nodes=list(dict.fromkeys(looked)))
    except agent.OutOfCredits:
        # returned (not raised) so the alert runs after the response is sent
        return JSONResponse({"detail": agent.BROKE_MESSAGE}, status_code=402,
                            background=BackgroundTask(alert_out_of_credits))


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


def select_edges(ids: str | None, focus: str | None, everything: bool) -> list[str]:
    """Which edges to show. Default: edges that link 2+ of the main stakeholders."""
    H = app.state.H
    if ids:
        return [i for i in (s.strip() for s in ids.split(",")) if i in H.edge_props]
    if focus:
        want = {i.strip() for i in focus.split(",")}
        return [e for e, ms in H.edges.elements.items() if want & set(ms)]
    if everything:
        return list(H.edges.elements)
    return viz.pick_edges(H, [], False)


@app.get("/graph")
def graph(
    ids: str | None = Query(None, description="comma-separated edge ids (e.g. the ones an answer cited)"),
    focus: str | None = Query(None, description="comma-separated node ids: every edge touching them"),
    all: bool = Query(False, description="the whole graph (crowded)"),
):
    """The same data as JSON: nodes and edges."""
    H = app.state.H

    def nm(n: str) -> str:
        return H.node_props[n]["name"].split(" (")[0]

    nodes, edges = {}, []
    for eid in select_edges(ids, focus, all):
        p, members = H.edge_props[eid], list(H.edges.elements[eid])
        for m in members:
            if m not in nodes:
                nodes[m] = {"id": m, "name": nm(m), "kind": H.node_props[m]["kind"], "note": H.node_props[m]["note"],
                            "edges": app.state.degree[m]}
        edges.append({
            "id": eid, "type": p["type"], "stance": p["stance"], "quote": p["quote"],
            "source": p["source_file"], "when": (p["when"] or {}).get("text"),
            "participants": [{"id": m, "role": p["roles"].get(m, "")} for m in members],
            "drivers": [nm(d) for d in p["drivers"] if d in H.node_props],
        })
    return {"nodes": list(nodes.values()), "edges": edges}


@app.get("/graph-view", response_class=HTMLResponse)
def graph_view(
    ids: str | None = Query(None, description="edge ids to show and highlight"),
    focus: str | None = Query(None, description="node ids: every edge touching them"),
    all: bool = Query(False, description="the whole graph (crowded)"),
    cited: str | None = Query(None, description="edge ids used in an answer: ringed in white (defaults to ids)"),
    nodes: str | None = Query(None, description="node ids the agent visited: ringed in blue"),
):
    """The interactive graph page (same as data/processed/graph.html). The web page embeds this."""
    edge_ids = select_edges(ids, focus, all)
    extra = [n.strip() for n in nodes.split(",") if n.strip()] if nodes else None
    if not edge_ids and not extra:
        raise HTTPException(404, "no matching edges")
    if cited is not None:
        select = [i.strip() for i in cited.split(",") if i.strip()]
    else:
        select = [i.strip() for i in ids.split(",")] if ids else None
    return viz.render_html(app.state.H, edge_ids, select=select, nodes=extra)


@app.get("/")
def home():
    return FileResponse(ROOT / "web" / "index.html")
