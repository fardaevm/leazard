import os
import re
from pathlib import Path
import pdfplumber

PDF_MAGIC = b"%PDF-"
MIN_TEXT_CHARS = int(os.getenv("MIN_PDF_TEXT_CHARS", "200"))

def extract_text_from_pdf(pdf_path: Path) -> str:
    parts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            t = (page.extract_text() or "").strip()
            if t:
                parts.append(f"\n\n--- Page {i} ---\n{t}")
    return "".join(parts).strip()


def looks_scanned(text: str, min_chars: int = MIN_TEXT_CHARS) -> bool:
    """True if extracted text (minus page markers and whitespace) is too short to be a real lease."""
    body = re.sub(r"--- Page \d+ ---", "", text or "")
    return len(re.sub(r"\s+", "", body)) < min_chars
