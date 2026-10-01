"""Regenerate the README screenshots and demo GIF from the real, running app.

What it does
  1. Builds the Docker image and starts a throwaway container on port 8001 with its own
     data volume (`leazard-shots-data`), so your real data is never touched.
  2. Drives the UI with Playwright: registers a "demo" account (random password), uploads
     tests/golden/sample_sf_lease_agreement.pdf with ZIP 94110, waits for the real analysis.
  3. Saves WebP screenshots to docs/screenshots/ and docs/demo.gif, then removes the
     container and volume.

Requirements: Docker, ffmpeg, cwebp, and a .env with OPENAI_API_KEY and SECRET_KEY
(passed to the container only; this script never reads or prints them). One run makes
one real analysis with your OpenAI key.

Re-run (Playwright is a dev tool only; keep it out of the app's dependencies):
  python3 -m venv /tmp/pw && /tmp/pw/bin/pip install playwright && /tmp/pw/bin/playwright install chromium
  /tmp/pw/bin/python scripts/capture_screenshots.py
"""
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "docs" / "screenshots"
GIF = ROOT / "docs" / "demo.gif"
SAMPLE = ROOT / "tests" / "golden" / "sample_sf_lease_agreement.pdf"
IMAGE, CONTAINER, VOLUME, PORT = "leazard:shots", "leazard-shots", "leazard-shots-data", 8001
BASE = f"http://127.0.0.1:{PORT}"
DESKTOP = {"width": 1280, "height": 800}
MOBILE = {"width": 375, "height": 812}


def sh(*cmd: str, check: bool = True) -> None:
    subprocess.run(cmd, check=check, stdout=subprocess.DEVNULL, stderr=None if check else subprocess.DEVNULL)


def start_app() -> None:
    sh("docker", "build", "-q", "-t", IMAGE, str(ROOT))
    sh("docker", "rm", "-f", CONTAINER, check=False)
    sh("docker", "volume", "rm", "-f", VOLUME, check=False)
    sh("docker", "run", "-d", "--name", CONTAINER, "-p", f"127.0.0.1:{PORT}:8000",
       "--env-file", str(ROOT / ".env"),
       "-e", "DATABASE_URL=sqlite:////data/leaze.db", "-e", "UPLOAD_DIR=/data/uploads",
       "-e", "RAG_STORE_DIR=/data/rag_store", "-e", "LAW_DIR=/app/data/law",
       "--user", "10001:10001", "--read-only", "--tmpfs", "/tmp",
       "-v", f"{VOLUME}:/data", IMAGE)
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{BASE}/health", timeout=2)
            return
        except OSError:
            time.sleep(1)
    sys.exit("App did not become healthy; see `docker logs leazard-shots`.")


def stop_app() -> None:
    sh("docker", "rm", "-f", CONTAINER, check=False)
    sh("docker", "volume", "rm", "-f", VOLUME, check=False)


def save(page: Page, name: str, element=None) -> None:
    png = SHOTS / f"{name}.png"
    if element:
        element.screenshot(path=str(png))
    elif page.viewport_size["width"] == DESKTOP["width"]:
        # The app is a centered 672 px column; crop the empty side margins so it stays legible in the README.
        page.screenshot(path=str(png), clip={"x": 264, "y": 0, "width": 752, "height": DESKTOP["height"]})
    else:
        page.screenshot(path=str(png))
    sh("cwebp", "-quiet", "-q", "82", str(png), "-o", str(png.with_suffix(".webp")))
    png.unlink()
    print(f"  {name}.webp  {png.with_suffix('.webp').stat().st_size // 1024} KB")


def run_flow(page: Page, password: str) -> str:
    """Home → sign up → upload → analyzing → results. Returns the results URL."""
    page.goto(BASE)
    page.wait_for_selector("h1")
    page.set_input_files("#lease", str(SAMPLE))
    page.fill("#zip", "94110")
    save(page, "01-home")
    page.click("button[type=submit]:has-text('Analyze my lease')")
    page.click("#auth-toggle")                       # switch the dialog to "Create account"
    page.fill("#auth-user", "demo")
    page.fill("#auth-pass", password)
    page.wait_for_timeout(1200)                      # pauses only so the dialog is visible in the GIF
    page.click("#auth-submit")
    page.wait_for_url("**/#/analyzing/*")
    started = time.monotonic()
    page.wait_for_function("document.querySelectorAll('.step.done').length >= 2", timeout=120_000)
    save(page, "02-analyzing")
    page.wait_for_url("**/#/results/*", timeout=300_000)
    page.wait_for_selector(".score-num")
    print(f"  analysis took {time.monotonic() - started:.0f}s (job start to results)")
    page.wait_for_timeout(4000)                      # let the GIF linger on the result
    return page.url


def capture_results(page: Page) -> None:
    page.evaluate("window.scrollTo(0, 0)")
    save(page, "03-results")

    flag = page.locator("article.card.flag").first
    if flag.count():
        flag.locator("details").first.evaluate("d => d.open = true")
        flag.scroll_into_view_if_needed()
        save(page, "04-flag", element=flag)
    else:
        print("  (no flagged clauses for the sample lease; skipped 04-flag)")

    standard = page.locator("details.card:has(summary:has-text('Looks standard'))")
    if standard.count():
        standard.evaluate("d => d.open = true")
        standard.evaluate("d => window.scrollTo(0, d.getBoundingClientRect().top + scrollY - 16)")
        save(page, "05-standard")

    letter = page.locator("section[aria-labelledby=letter-h]")
    if letter.count():
        letter.scroll_into_view_if_needed()
        save(page, "06-email", element=letter)


def main() -> None:
    if not (ROOT / ".env").exists():
        sys.exit("Create .env with OPENAI_API_KEY and SECRET_KEY first (see .env.example).")
    for tool in ("docker", "ffmpeg", "cwebp"):
        if not shutil.which(tool):
            sys.exit(f"{tool} is required.")
    SHOTS.mkdir(parents=True, exist_ok=True)
    password = secrets.token_urlsafe(16)            # throwaway; never printed
    video_dir = Path(tempfile.mkdtemp(prefix="leazard-video-"))

    print("Starting app on", BASE)
    start_app()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()

            ctx = browser.new_context(viewport=DESKTOP, color_scheme="light",
                                      record_video_dir=str(video_dir), record_video_size=DESKTOP)
            page = ctx.new_page()
            results_url = run_flow(page, password)
            video = page.video.path()
            capture_results(page)
            storage = ctx.storage_state()            # reuse the session token for other views
            ctx.close()

            for scheme in ("light", "dark"):
                c = browser.new_context(viewport=MOBILE, device_scale_factor=2, is_mobile=True,
                                        has_touch=True, color_scheme=scheme, storage_state=storage)
                m = c.new_page()
                m.goto(results_url)
                m.wait_for_selector(".score-num")
                save(m, f"07-mobile-{scheme}")
                c.close()

            browser.close()

        # Video → GIF: 2x speed, 8 fps, 800 px wide, shared palette.
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vf",
             "setpts=0.5*PTS,fps=8,scale=800:-1:flags=lanczos,split[a][b];"
             "[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
             str(GIF)], check=True)
        print(f"  demo.gif  {GIF.stat().st_size // 1024} KB")
    finally:
        stop_app()
        shutil.rmtree(video_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
