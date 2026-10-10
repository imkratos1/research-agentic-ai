import threading
import time
from io import BytesIO
from urllib.parse import urlparse

import requests
import trafilatura
from pypdf import PdfReader

from agent import cache, config

_last_visit = {}
_visit_lock = threading.Lock()


def check_url(url):
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError(f"Only http and https links can be opened: {url}")


def _wait_for_site(site):
    # be polite: never hit the same website more than once per SAME_SITE_DELAY seconds, even from parallel tool calls
    with _visit_lock:
        wait = max(0.0, _last_visit.get(site, 0.0) + config.SAME_SITE_DELAY - time.time())
        _last_visit[site] = time.time() + wait
    if wait:
        time.sleep(wait)


def _download(url):
    _wait_for_site(urlparse(url).netloc)
    response = requests.get(url, headers={"User-Agent": config.user_agent()}, timeout=config.REQUEST_TIMEOUT, stream=True)
    response.raise_for_status()
    data = b""
    for chunk in response.iter_content(65536):
        data += chunk
        if len(data) > config.MAX_DOWNLOAD_BYTES:
            raise ValueError(f"File is bigger than {config.MAX_DOWNLOAD_BYTES // 1_000_000} MB, skipped: {url}")
    return data, response.headers.get("Content-Type", "").lower()


def read_html(html, url):
    doc = trafilatura.bare_extraction(html, url=url, with_metadata=True, include_comments=False, favor_precision=True)
    info = doc.as_dict() if doc else {}
    return {
        "url": url,
        "title": info.get("title") or "",
        "author": info.get("author") or "",
        "date": info.get("date") or "",
        "site_name": info.get("sitename") or "",
        "text": info.get("text") or "",
        "file_type": "html",
    }


def read_pdf(data, url):
    reader = PdfReader(BytesIO(data))
    pages = reader.pages[: config.MAX_PDF_PAGES]
    text = "\n".join(page.extract_text() or "" for page in pages)
    meta = reader.metadata
    return {
        "url": url,
        "title": str(meta.title or "") if meta else "",
        "author": str(meta.author or "") if meta else "",
        "date": "",
        "site_name": "",
        "text": text,
        "file_type": "pdf",
    }


def fetch_page(url):
    check_url(url)
    page = cache.load("pages", url)
    if page is not None:
        return page

    data, content_type = _download(url)
    if "pdf" in content_type or url.lower().endswith(".pdf"):
        page = read_pdf(data, url)
    else:
        page = read_html(data, url)

    page["text"] = page["text"][: config.MAX_PAGE_CHARS]
    if len(page["text"]) < config.MIN_PAGE_CHARS:
        raise ValueError(f"Too little readable text (maybe a paywall, login page or video): {url}")

    cache.save("pages", url, page)
    return page
