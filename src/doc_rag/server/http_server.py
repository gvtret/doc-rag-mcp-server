from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from doc_rag.server.retrieval import doc_search

app = FastAPI(title="doc-rag-http")


class Req(BaseModel):
    query: str
    top_k: int | None = None


@app.get("/health")
def h():
    return {"ok": True}


@app.post("/search")
def s(r: Req):
    k = r.top_k if r.top_k is not None else 6
    k = max(1, min(50, int(k)))
    return {"results": doc_search(r.query, k)}


def run(h, p):
    import uvicorn

    uvicorn.run(app, host=h, port=p)
