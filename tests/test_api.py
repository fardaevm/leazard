import asyncio
import uuid

import pytest

from conftest import make_pdf

LEASE_LINES = [
    "RESIDENTIAL LEASE AGREEMENT between Landlord and Tenant for the premises.",
    "Tenant shall pay monthly rent of 3000 dollars on the first day of each month.",
    "A late fee of 200 dollars applies if rent is received after the third day.",
    "Security deposit of 6000 dollars is due at signing and is non-refundable.",
    "Tenant may not sublet the premises without prior written consent of Landlord.",
]


def _auth(client) -> dict:
    username, password = f"u_{uuid.uuid4().hex[:10]}", "s3cret-pass"
    assert client.post("/auth/register", json={"username": username, "password": password}).status_code == 201
    r = client.post("/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _analyze(client, headers, data: bytes, filename="lease.pdf", zip_code="94110"):
    return client.post(
        "/analyze",
        headers=headers,
        data={"zip_code": zip_code},
        files={"lease": (filename, data, "application/pdf")},
    )


def _files(upload_dir):
    return set(upload_dir.glob("*.pdf")) if upload_dir.exists() else set()


# ── File ownership ────────────────────────────────────────────────────────────

def test_owner_can_fetch_pdf_and_other_user_gets_404(client):
    owner, other = _auth(client), _auth(client)
    r = _analyze(client, owner, make_pdf(LEASE_LINES))
    assert r.status_code == 200, r.text
    body = r.json()
    for key in ("id", "file_id", "zip_code", "status", "message", "lease_json", "risk_json", "letter_text"):
        assert key in body
    file_id = body["file_id"]

    ok = client.get(f"/uploads/{file_id}", headers=owner)
    assert ok.status_code == 200
    assert ok.content.startswith(b"%PDF-")

    assert client.get(f"/uploads/{file_id}", headers=other).status_code == 404


def test_pdf_without_record_is_404_even_if_file_exists(client, upload_dir):
    headers = _auth(client)
    upload_dir.mkdir(parents=True, exist_ok=True)
    orphan = uuid.uuid4().hex
    (upload_dir / f"{orphan}.pdf").write_bytes(make_pdf(LEASE_LINES))
    assert client.get(f"/uploads/{orphan}", headers=headers).status_code == 404


def test_pdf_requires_auth_and_valid_id(client):
    assert client.get(f"/uploads/{uuid.uuid4().hex}").status_code in (401, 403)
    headers = _auth(client)
    assert client.get("/uploads/..%2Fleaze", headers=headers).status_code in (400, 404)
    assert client.get("/uploads/not-hex", headers=headers).status_code == 400


def test_tampered_token_rejected(client):
    headers = _auth(client)
    bad = {"Authorization": headers["Authorization"][:-2] + "xx"}
    assert client.get("/history", headers=bad).status_code == 401


# ── Upload validation ─────────────────────────────────────────────────────────

def test_rejects_non_pdf_bytes_with_pdf_extension(client, upload_dir):
    headers = _auth(client)
    before = _files(upload_dir)
    r = _analyze(client, headers, b"MZ\x90\x00 definitely not a pdf" * 10)
    assert r.status_code == 400
    assert "not a valid PDF" in r.json()["detail"]
    assert _files(upload_dir) == before
    assert client.graph.calls == 0


def test_rejects_oversized_upload(client, upload_dir, main_module):
    headers = _auth(client)
    before = _files(upload_dir)
    big = b"%PDF-1.4\n" + b"0" * (main_module.MAX_UPLOAD_BYTES + 10)
    r = _analyze(client, headers, big)
    assert r.status_code == 413
    assert _files(upload_dir) == before


def test_size_limit_enforced_while_streaming(main_module, tmp_path, monkeypatch):
    """No Content-Length involved: _save_upload must stop as soon as the limit is crossed."""
    from fastapi import HTTPException

    monkeypatch.setattr(main_module, "MAX_UPLOAD_BYTES", 3 * 1024)
    monkeypatch.setattr(main_module, "UPLOAD_READ_CHUNK", 1024)

    class Stream:
        reads = 0
        async def read(self, n):
            self.reads += 1
            return (b"%PDF-" + b"x" * (n - 5)) if self.reads == 1 else b"x" * n

    s = Stream()
    with pytest.raises(HTTPException) as exc:
        asyncio.run(main_module._save_upload(s, tmp_path / "f.pdf"))
    assert exc.value.status_code == 413
    assert s.reads == 4


def test_rejects_scanned_pdf_with_clear_message(client, upload_dir):
    headers = _auth(client)
    before = _files(upload_dir)
    r = _analyze(client, headers, make_pdf([]))
    assert r.status_code == 422
    assert r.json()["detail"] == "This looks like a scanned PDF. Scanned leases aren't supported yet."
    assert _files(upload_dir) == before


def test_file_deleted_when_analysis_raises(client, upload_dir):
    headers = _auth(client)
    before = _files(upload_dir)
    client.graph.raise_exc = RuntimeError("llm down")
    r = _analyze(client, headers, make_pdf(LEASE_LINES))
    assert r.status_code == 502
    assert _files(upload_dir) == before


def test_file_deleted_when_analysis_returns_error(client, upload_dir):
    headers = _auth(client)
    before = _files(upload_dir)
    client.graph.status = "error"
    r = _analyze(client, headers, make_pdf(LEASE_LINES))
    assert r.status_code == 502
    assert _files(upload_dir) == before
    assert client.get("/history", headers=headers).json() == []


def test_rejects_wrong_extension(client):
    headers = _auth(client)
    assert _analyze(client, headers, make_pdf(LEASE_LINES), filename="lease.txt").status_code == 400
