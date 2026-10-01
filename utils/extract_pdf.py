from pathlib import Path
import pdfplumber

def extract_text_from_pdf(pdf_path: Path) -> str:
    parts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            t = (page.extract_text() or "").strip()
            if t:
                parts.append(f"\n\n--- Page {i} ---\n{t}")
    return "".join(parts).strip()


