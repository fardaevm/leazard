import os

from rag import indexer


def _write(path, data: bytes):
    path.write_bytes(data)
    return path


def test_fingerprint_ignores_mtime(tmp_path):
    p = _write(tmp_path / "a.pdf", b"%PDF-1.4 law text")
    before = indexer._fingerprint([p])
    os.utime(p, (1_000_000, 1_000_000))
    assert indexer._fingerprint([p]) == before


def test_fingerprint_changes_with_content(tmp_path):
    p = _write(tmp_path / "a.pdf", b"%PDF-1.4 law text")
    before = indexer._fingerprint([p])
    p.write_bytes(b"%PDF-1.4 amended law text")
    assert indexer._fingerprint([p]) != before


def test_fingerprint_changes_with_name(tmp_path):
    a = _write(tmp_path / "a.pdf", b"%PDF-1.4 same")
    b = _write(tmp_path / "b.pdf", b"%PDF-1.4 same")
    assert indexer._fingerprint([a]) != indexer._fingerprint([b])
