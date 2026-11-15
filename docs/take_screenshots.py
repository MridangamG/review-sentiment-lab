"""Capture README screenshots of the running Streamlit app (one per tab).

    streamlit run app/streamlit_app.py --server.headless true &
    python docs/take_screenshots.py            # needs `pip install playwright` + Google Chrome

Uses the locally installed Google Chrome via Playwright (channel="chrome"),
so no browser download is required.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
OUT = Path(__file__).parent / "screenshots"
OUT.mkdir(exist_ok=True)
TABS = ["Live Analyzer", "Explain", "Truncation Visualizer", "Length Lab", "Review Dashboard"]


def wait_idle(page, extra_ms=1500):
    # Streamlit shows a "Running..." status widget while a script run is in progress
    page.wait_for_timeout(800)
    page.wait_for_function("() => !document.querySelector('[data-testid=\"stStatusWidget\"]')", timeout=180_000)
    page.wait_for_timeout(extra_ms)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome")
    page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1.5)
    page.goto(URL)
    page.wait_for_selector("text=Review Sentiment Lab", timeout=180_000)
    wait_idle(page, 3000)
    errors = []
    for i, tab in enumerate(TABS):
        page.get_by_role("tab", name=tab).click()
        wait_idle(page, 2500)
        if tab == "Review Dashboard":
            page.locator("[data-testid=stButtonGroup] button", has_text="Battery").first.click()
            wait_idle(page, 2500)
        exc = page.locator("[data-testid='stException']")
        if exc.count():
            errors.append((tab, exc.first.inner_text()[:2000]))
        f = OUT / f"{i + 1}_{tab.lower().replace(' ', '_')}.png"
        # Streamlit scrolls inside its own container, so grow the viewport to the content height
        h = page.evaluate("() => { const m = document.querySelector('[data-testid=\"stMain\"]') "
                          "|| document.body; return m.scrollHeight; }")
        page.set_viewport_size({"width": 1440, "height": min(max(int(h) + 40, 1000), 4200)})
        page.wait_for_timeout(1500)
        page.screenshot(path=str(f))
        page.set_viewport_size({"width": 1440, "height": 1000})
        print("saved", f)
    browser.close()
for tab, e in errors:
    print(f"\n!!! exception in tab {tab}:\n{e}")
sys.exit(1 if errors else 0)
