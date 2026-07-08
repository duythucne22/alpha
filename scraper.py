#!/usr/bin/env python3
"""Task 1 — Scrape Zendesk Help Center articles and convert them to clean Markdown.

Fetches articles from the OptiSigns support center (Zendesk API), converts the
HTML body to Markdown with html2text, and saves each article as contents/<slug>.md
with a metadata header. Returns structured records so main.py can run delta
detection on top.
"""

import logging
import os
import re
import time

import html2text
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

BASE_URL = "https://support.optisigns.com/api/v2/help_center/articles.json"
DETAIL_URL = "https://support.optisigns.com/api/v2/help_center/articles/{article_id}.json"
LOCALE = "en-us"
PER_PAGE = 50
REQUEST_DELAY = 0.3  # seconds between HTTP requests, avoids Zendesk rate limits
REQUEST_TIMEOUT = 30
OUTPUT_DIR = "contents"


def _markdown_converter() -> html2text.HTML2Text:
    h = html2text.HTML2Text()
    h.body_width = 0          # no hard wrapping — keeps lines intact for chunking
    h.ignore_images = False
    h.ignore_emphasis = False
    h.mark_code = False
    h.single_line_break = False
    return h


def clean_html(html: str) -> str:
    """Strip nav/script/style/ads-like elements before conversion."""
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "nav", "iframe", "noscript"]):
        tag.decompose()
    return str(soup)


def html_to_markdown(html: str) -> str:
    markdown = _markdown_converter().handle(clean_html(html))
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)  # collapse excessive blank lines
    return markdown.strip()


def extract_slug_from_url(html_url: str) -> str | None:
    """Fallback slug: last URL segment minus the leading numeric article id.

    https://.../articles/53216900345875-What-Access-Does-OptiSigns-Have
    -> what-access-does-optisigns-have
    """
    if not html_url:
        return None
    last_part = html_url.rstrip("/").split("/")[-1]
    dash_index = last_part.find("-")
    if dash_index == -1:
        return None
    slug = last_part[dash_index + 1:]
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[-\s]+", "-", slug)
    return slug.strip("-").lower() or None


def slug_for(article: dict) -> str | None:
    """Prefer Zendesk's own slug field; fall back to parsing html_url."""
    slug = article.get("slug")
    if slug:
        return slug.strip("-").lower()
    return extract_slug_from_url(article.get("html_url", ""))


def build_markdown(article: dict, body_markdown: str) -> str:
    """Metadata header + body. The literal "Article URL:" line lets the bot
    cite sources exactly as its system prompt asks for."""
    title = article.get("title", "Untitled")
    html_url = article.get("html_url", "")
    return (
        f"# {title}\n\n"
        f"**Source:** {html_url}\n\n"
        f"Article URL: {html_url}\n\n"
        "---\n\n"
        f"{body_markdown}\n"
    )


def fetch_article_detail(session: requests.Session, article_id: int) -> dict | None:
    """Fetch a single article via the detail API (used when the list payload
    is missing the body)."""
    url = DETAIL_URL.format(article_id=article_id)
    try:
        resp = session.get(url, params={"locale": LOCALE}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("article")
    except requests.RequestException as exc:
        logger.error("Failed to fetch detail for article %s: %s", article_id, exc)
        return None


def fetch_all_articles(session: requests.Session | None = None) -> list[dict]:
    """Fetch every article via the paginated list API (per_page=50)."""
    session = session or requests.Session()
    articles: list[dict] = []
    next_page: str | None = BASE_URL
    params = {"locale": LOCALE, "per_page": PER_PAGE}
    page = 1

    while next_page:
        logger.info("Fetching page %d ...", page)
        try:
            resp = session.get(next_page, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Failed to fetch page %d: %s", page, exc)
            break

        data = resp.json()
        batch = data.get("articles", [])
        articles.extend(batch)
        logger.info("  got %d articles (total %d)", len(batch), len(articles))

        next_page = data.get("next_page")
        params = None  # next_page URL already carries the query string
        page += 1
        if next_page:
            time.sleep(REQUEST_DELAY)

    return articles


def scrape_articles(output_dir: str = OUTPUT_DIR) -> list[dict]:
    """Scrape all articles, write contents/<slug>.md, return records for delta
    detection: [{id, slug, title, html_url, updated_at, path, content}]."""
    os.makedirs(output_dir, exist_ok=True)
    session = requests.Session()

    records: list[dict] = []
    failed = 0

    for article in fetch_all_articles(session):
        title = article.get("title", "Untitled")
        try:
            slug = slug_for(article)
            if not slug:
                logger.warning("Skipping article without usable slug: %s", title)
                failed += 1
                continue

            body_html = article.get("body")
            if body_html is None:
                # list payload lacked the body — fall back to the detail API
                time.sleep(REQUEST_DELAY)
                detail = fetch_article_detail(session, article["id"])
                body_html = (detail or {}).get("body") or ""

            content = build_markdown(article, html_to_markdown(body_html))
            path = os.path.join(output_dir, f"{slug}.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            records.append({
                "id": article["id"],
                "slug": slug,
                "title": title,
                "html_url": article.get("html_url", ""),
                "updated_at": article.get("updated_at", ""),
                "path": path,
                "content": content,
            })
            logger.info("  saved %s.md", slug)
        except Exception as exc:  # keep going — one bad article must not kill the run
            failed += 1
            logger.error("Error processing article '%s': %s", title, exc)

    logger.info("Scrape complete: %d saved, %d failed -> %s",
                len(records), failed, os.path.abspath(output_dir))
    return records


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scraped = scrape_articles()
    print(f"Scraped {len(scraped)} articles into ./{OUTPUT_DIR}/")
