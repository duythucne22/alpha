#!/usr/bin/env python
"""Task 3 — Daily job: re-scrape, detect deltas, upload only what changed.

Flow:
1. Load state from data/state.json (created on first run).
2. Scrape all articles from the Zendesk API (Task 1).
3. For each article decide added / updated / skipped using updated_at + MD5 hash.
4. Upload only new/changed articles; on update, remove the stale file first so
   the vector store never contains two versions of one article.
5. Create the assistant on first run, refresh it afterwards (Task 2).
6. Save state, print "Added: X, Updated: Y, Skipped: Z", run the sanity check.

Exit code 0 on success, 1 on fatal error — required for cron/scheduler health.
"""

import hashlib
import json
import logging
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from scraper import scrape_articles
from vector_store_uploader import (
    find_vector_store_by_name,
    get_or_create_assistant,
    get_or_create_vector_store,
    remove_file,
    run_sanity_check,
    smart_hybrid_chunking,
    store_file_count,
    upload_markdown_file,
    wipe_vector_store,
)

logger = logging.getLogger(__name__)

STATE_FILE = os.environ.get("STATE_FILE", os.path.join("data", "state.json"))

# Reconciliation safety: if this many known articles vanish from one scrape,
# assume a broken scrape (not a real mass-deletion) and refuse to delete.
DELETE_THRESHOLD_FRACTION = 0.20
DELETE_THRESHOLD_MIN = 5


# ==========================================================================
# State management
# ==========================================================================

def load_state(path: str = STATE_FILE) -> dict:
    """Load state; return a fresh skeleton if the file doesn't exist yet."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except FileNotFoundError:
        state = {}
    except json.JSONDecodeError:
        logger.warning("State file %s is corrupt, starting fresh", path)
        state = {}
    state.setdefault("vector_store_id", None)
    state.setdefault("assistant_id", None)
    state.setdefault("articles", {})
    return state


def save_state(state: dict, path: str = STATE_FILE) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def get_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ==========================================================================
# Delta detection
# ==========================================================================

def classify_article(entry: dict | None, updated_at: str, content_hash: str) -> str:
    """added   — never seen this article id before
    updated — Zendesk timestamp OR rendered-markdown hash changed
    skipped — identical to last run"""
    if entry is None:
        return "added"
    if entry.get("updated_at") != updated_at or entry.get("hash") != content_hash:
        return "updated"
    return "skipped"


def find_removed_ids(known_ids: set[str], scraped_ids: set[str]) -> list[str]:
    """Article ids we synced before that no longer exist on Zendesk."""
    return sorted(known_ids - scraped_ids)


def deletion_is_suspicious(n_missing: int, n_known: int) -> bool:
    """A large fraction of known articles vanishing in one run usually means a
    broken scrape, not a real mass-deletion — don't wipe the knowledge base."""
    if n_known == 0 or n_missing <= DELETE_THRESHOLD_MIN:
        return False
    return n_missing / n_known > DELETE_THRESHOLD_FRACTION


# ==========================================================================
# Orchestration
# ==========================================================================

def run() -> dict:
    state = load_state()
    client = OpenAI()

    logger.info("Step 1/4 — scraping articles from Zendesk ...")
    records = scrape_articles()
    if not records:
        raise RuntimeError("Scraper returned no articles — aborting without touching state")

    logger.info("Step 2/4 — delta detection + upload ...")

    # Startup guard: state lost (no articles tracked) but a populated store
    # already exists → re-uploading everything would double every document.
    # Refuse, unless FORCE_RESYNC=1 wipes the store for a clean rebuild.
    if not state["vector_store_id"]:
        adopted = find_vector_store_by_name(client)
        if adopted:
            if not state["articles"] and store_file_count(client, adopted) > 0:
                if os.environ.get("FORCE_RESYNC", "").lower() in ("1", "true", "yes"):
                    logger.warning("FORCE_RESYNC set — wiping store %s for clean rebuild", adopted)
                    wipe_vector_store(client, adopted)
                else:
                    raise RuntimeError(
                        f"State is empty but vector store {adopted} already holds files. "
                        "Re-uploading would duplicate every document. Restore data/state.json "
                        "(volume mount?) or set FORCE_RESYNC=1 to wipe and rebuild."
                    )
            state["vector_store_id"] = adopted
            logger.info("Adopted existing vector store %s", adopted)
    vector_store_id = get_or_create_vector_store(client, state)

    counts = {"added": 0, "updated": 0, "skipped": 0, "removed": 0, "failed": 0,
              "delete_aborted": False}
    chunk_total = 0

    for rec in records:
        article_key = str(rec["id"])
        content_hash = get_content_hash(rec["content"])
        entry = state["articles"].get(article_key)
        status = classify_article(entry, rec["updated_at"], content_hash)

        if status == "skipped":
            counts["skipped"] += 1
            continue

        try:
            if status == "updated" and entry and entry.get("file_id"):
                remove_file(client, vector_store_id, entry["file_id"])
            file_id = upload_markdown_file(client, vector_store_id, rec["path"])
        except OpenAIError as exc:
            counts["failed"] += 1
            logger.error("Upload failed for %s: %s", rec["slug"], exc)
            continue  # state not updated -> retried automatically on the next run

        chunks = len(smart_hybrid_chunking(rec["content"]))
        chunk_total += chunks
        counts[status] += 1
        state["articles"][article_key] = {
            "slug": rec["slug"],
            "html_url": rec["html_url"],
            "updated_at": rec["updated_at"],
            "hash": content_hash,
            "file_id": file_id,
        }
        save_state(state)  # crash-safe: progress survives an interrupted run
        logger.info("  %s: %s (%d chunks)", status.upper(), rec["slug"], chunks)

    # Reconciliation: articles deleted on Zendesk must leave the store too,
    # or the bot keeps citing zombie docs.
    removed_ids = find_removed_ids(set(state["articles"]),
                                   {str(r["id"]) for r in records})
    if removed_ids:
        if deletion_is_suspicious(len(removed_ids), len(state["articles"])):
            counts["delete_aborted"] = True
            logger.error(
                "%d of %d known articles missing from scrape — looks like a broken "
                "scrape, refusing to delete. Investigate before the next run.",
                len(removed_ids), len(state["articles"]))
        else:
            for article_key in removed_ids:
                entry = state["articles"][article_key]
                if entry.get("file_id"):
                    remove_file(client, vector_store_id, entry["file_id"])
                del state["articles"][article_key]
                counts["removed"] += 1
                logger.info("  REMOVED: %s (deleted on Zendesk)", entry.get("slug", article_key))
            save_state(state)

    logger.info("Step 3/4 — ensuring assistant ...")
    assistant_id = get_or_create_assistant(client, state, vector_store_id)
    save_state(state)

    uploaded = counts["added"] + counts["updated"]
    logger.info("Embedded %d files, ~%d chunks this run (local estimate)",
                uploaded, chunk_total)
    print(f"\nAdded: {counts['added']}, Updated: {counts['updated']}, "
          f"Skipped: {counts['skipped']}, Removed: {counts['removed']}" +
          (f", Failed: {counts['failed']}" if counts["failed"] else ""))

    if os.environ.get("SKIP_SANITY_CHECK", "").lower() in ("1", "true", "yes"):
        logger.info("Step 4/4 — sanity check skipped (SKIP_SANITY_CHECK set)")
    else:
        logger.info("Step 4/4 — sanity check ...")
        url_by_file_id = {e["file_id"]: e["html_url"]
                          for e in state["articles"].values()
                          if e.get("file_id") and e.get("html_url")}
        run_sanity_check(client, assistant_id, url_by_file_id=url_by_file_id)

    return counts


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY is not set (see .env.sample)")
        return 1
    try:
        counts = run()
    except Exception as exc:
        logger.exception("Job failed: %s", exc)
        return 1
    if counts["failed"] or counts["delete_aborted"]:
        # partial failure: uploads retry next run automatically, but the
        # scheduler must see red so a systemic issue can't rot silently
        logger.error("Run finished with problems: %d failed uploads%s",
                     counts["failed"],
                     ", deletion pass aborted" if counts["delete_aborted"] else "")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
