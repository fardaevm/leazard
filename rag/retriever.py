# rag/retriever.py
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from rag.indexer import ensure_index, STORE_DIR, INDEX_NAME, EMBED_MODEL  # changed

DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "6"))

def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBED_MODEL)

@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    source: str
    page: Optional[int]
    score: Optional[float]
    text: str
    section: Optional[str] = None
    url: Optional[str] = None

def display_source_name(source: str) -> str:
    """'ca_civil_code-1950.5.pdf' -> 'ca civil code 1950.5'."""
    stem = Path(source).stem if source.lower().endswith(".pdf") else source
    name = " ".join(stem.replace("_", " ").replace("-", " ").split())
    return name or "unknown"

class LawRetriever:
    def __init__(self, top_k: int = DEFAULT_TOP_K):
        self.top_k = top_k
        self.index_dir = ensure_index()  # changed: docling indexer manages dirs/env
        self.vs = FAISS.load_local(
            str(self.index_dir),
            _embeddings(),
            allow_dangerous_deserialization=True,
        )

    def search(self, query: str, top_k: Optional[int] = None) -> List[RetrievedChunk]:
        k = self.top_k if top_k is None else top_k
        docs_scores = self.vs.similarity_search_with_score(query, k=k)
        out: List[RetrievedChunk] = []
        for doc, score in docs_scores:
            md = doc.metadata or {}
            out.append(RetrievedChunk(
                chunk_id=int(md.get("chunk_id", -1)),
                source=str(md.get("source", "unknown")),
                page=md.get("page", None),
                score=float(score) if score is not None else None,
                text=(doc.page_content or "").strip(),
                section=md.get("section") or None,
                url=md.get("url") or None,
            ))
        return out

    def context_with_chunks(
        self, query: str, top_k: Optional[int] = None, max_chars: int = 9000,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Return (context, refs). refs[i-1] describes the chunk labelled [i] in context.
        Only chunks whose label survives the max_chars cut are included."""
        chunks = self.search(query, top_k=top_k)
        parts: List[str] = []
        headers: List[str] = []
        for i, c in enumerate(chunks, start=1):
            page = "" if c.page is None else f" page={c.page}"
            headers.append(f"[{i}] chunk_id={c.chunk_id} source={c.source}{page}\n")
            parts.append(headers[-1] + c.text)
        ctx = "\n\n".join(parts).strip()[:max_chars]

        refs: List[Dict[str, Any]] = []
        pos = 0
        for c, header, part in zip(chunks, headers, parts):
            if pos + len(header) > len(ctx):
                break
            refs.append({
                "chunk_id":    c.chunk_id,
                "source":      c.source,
                "source_name": display_source_name(c.source),
                "page":        c.page,
                "section":     c.section,
                "url":         c.url,
            })
            pos += len(part) + 2
        return ctx, refs

    def context(self, query: str, top_k: Optional[int] = None, max_chars: int = 9000) -> str:
        return self.context_with_chunks(query, top_k=top_k, max_chars=max_chars)[0]

    def as_dicts(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        return [
            {"chunk_id": c.chunk_id, "source": c.source, "page": c.page, "score": c.score, "text": c.text}
            for c in self.search(query, top_k=top_k)
        ]
