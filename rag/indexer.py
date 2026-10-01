# rag/indexer.py  (Docling option)
import os, hashlib
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

import pdfplumber
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

LAW_DIR = Path(os.getenv("LAW_DIR", "data/law"))
STORE_DIR = Path(os.getenv("RAG_STORE_DIR", "rag_store"))
INDEX_NAME = os.getenv("RAG_INDEX_NAME", "sf_law_faiss")
EMBED_MODEL = os.getenv("OPENAI_EMBEDDINGS_MODEL") or "text-embedding-3-small"
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

def _iter_pdfs(law_dir: Path) -> List[Path]:
    if not law_dir.exists(): 
        return []
    return sorted([p for p in law_dir.glob("*.pdf") if p.is_file()])

def _fingerprint(pdfs: List[Path]) -> str:
    h = hashlib.sha256()
    h.update(f"{EMBED_MODEL}|{CHUNK_SIZE}|{CHUNK_OVERLAP}".encode())
    for p in pdfs:
        st = p.stat()
        h.update(f"{p.name}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()

def _meta_path(index_dir: Path) -> Path:
    return index_dir / "meta.txt"

def _read_meta(index_dir: Path) -> Optional[str]:
    p = _meta_path(index_dir)
    return p.read_text(encoding="utf-8").strip() if p.exists() else None

def _write_meta(index_dir: Path, fp: str) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    _meta_path(index_dir).write_text(fp, encoding="utf-8")

def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBED_MODEL)

def _pdf_to_docs(pdf_path: Path) -> List[Document]:
    out: List[Document] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = (page.extract_text() or "").strip()
            if not txt:
                continue
            out.append(Document(
                page_content=txt,
                metadata={"source": pdf_path.name, "page": i},
            ))
    return out

def _split(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, d in enumerate(chunks):
        md = d.metadata or {}
        md["chunk_id"] = i
        d.metadata = md
    return chunks

def ensure_index() -> Path:
    pdfs = _iter_pdfs(LAW_DIR)
    if not pdfs:
        raise FileNotFoundError(f"No PDFs in {LAW_DIR}")
    out_dir = STORE_DIR / INDEX_NAME
    fp = _fingerprint(pdfs)
    if (out_dir / "index.faiss").exists() and (out_dir / "index.pkl").exists() and _read_meta(out_dir) == fp:
        return out_dir

    docs: List[Document] = []
    for p in pdfs:
        docs.extend(_pdf_to_docs(p))
    if not docs:
        raise ValueError(f"Could not extract any text from PDFs in {LAW_DIR}")

    chunks = _split(docs)
    vs = FAISS.from_documents(chunks, _embeddings())
    out_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(out_dir))
    _write_meta(out_dir, fp)
    return out_dir