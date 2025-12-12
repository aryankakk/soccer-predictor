from __future__ import annotations

"""Browser-based HTML fetching (optional).

Use this when FBref is protected by Cloudflare and plain HTTP requests get 403.
Requires:
  - pip install playwright
  - python -m playwright install chromium
"""

def fetch_html_with_playwright(url: str, user_agent: str, timeout_s: float = 30.0) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Playwright is not installed. Install it with: pip install playwright"
        ) from e

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=user_agent,
            extra_http_headers={"Referer": "https://fbref.com/"},
            locale="en-US",
        )
        page = context.new_page()

        page.goto(url, wait_until="domcontentloaded", timeout=int(timeout_s * 1000))

        # Give Cloudflare/challenge JS time to complete (best-effort).
        # If a CAPTCHA/managed challenge is required, this will likely never clear.
        deadline_ms = int(timeout_s * 1000)
        elapsed = 0
        while elapsed < deadline_ms:
            try:
                title = (page.title() or "").strip().lower()
            except Exception:
                title = ""

            url_now = (page.url or "").lower()
            if ("just a moment" not in title) and ("checking your browser" not in title) and ("cdn-cgi" not in url_now):
                break
            page.wait_for_timeout(1000)
            elapsed += 1000

        try:
            page.wait_for_load_state("networkidle", timeout=min(10_000, deadline_ms))
        except Exception:
            # Non-fatal; some pages never reach networkidle.
            pass

        html = page.content()
        context.close()
        browser.close()
        return html
