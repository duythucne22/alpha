# kb-sync-agent

Daily job that scrapes the OptiSigns support center, converts articles to Markdown, and syncs only the changed ones into an OpenAI vector store behind an **OptiBot** assistant.

## Setup

```bash
pip install -r requirements.txt
cp .env.sample .env          # then paste your real OPENAI_API_KEY
```

## Run locally

```bash
python main.py               # scrape → delta detect → upload delta → sanity check
python scraper.py            # Task 1 only: scrape to contents/*.md
python -m pytest             # unit tests (offline, no API key needed)
```

Docker:

```bash
docker build -t kb-sync .
docker run -e OPENAI_API_KEY=sk-... -v $(pwd)/data:/app/data kb-sync   # runs once, exits 0
```

## Chunking strategy — hybrid structure-aware

1. Extract title + source URL from the Markdown metadata header, strip the header.
2. Split on **H2/H3 headings** (`MarkdownHeaderTextSplitter`) so chunks follow the article's own structure — a "Step 2" chunk never bleeds into "Step 3".
3. Sections longer than **3000 chars** get a `RecursiveCharacterTextSplitter` pass (overlap **150**) so no chunk exceeds the limit.
4. Every chunk carries `article_title` + `source_url` metadata; each `.md` file also embeds a literal `Article URL:` line so the bot can cite exactly as its prompt requires.

Note: OpenAI `file_search` re-chunks server-side — we pass an explicit `chunking_strategy` (800 tokens, 200 overlap) so those numbers are a deliberate choice; the local splitter is validation + logging.

## Delta detection

`data/state.json` maps `article_id → {slug, html_url, updated_at, hash, file_id}`.

- **added** — id never seen → upload
- **updated** — Zendesk `updated_at` **or** SHA-256 of the rendered Markdown changed → delete the old OpenAI file, upload the new one (no duplicate versions in the store)
- **skipped** — identical → no API calls
- **removed** — id in state but gone from Zendesk → file deleted from the store (no zombie docs). Safety valve: if >20% of known articles vanish in one run, deletion is aborted (broken scrape ≠ mass delete) and the job exits 1.

State is saved after every upload, so an interrupted run resumes cleanly. Each run ends with `Added: X, Updated: Y, Skipped: Z, Removed: W`.

**Exit codes:** 0 = clean run; 1 = fatal error, any failed upload, or aborted deletion pass — wire scheduler alerts to non-zero exits. Failed uploads are never written to state, so they retry automatically next run.

**State loss guard:** if `state.json` is empty but the vector store already holds files (volume not mounted?), the job refuses to run rather than duplicating every document. Set `FORCE_RESYNC=1` to wipe the store and rebuild cleanly.

**Citations:** answers cite `Article URL:` lines resolved deterministically from `file_search` annotations (`file_id → html_url` via state) — cited URLs cannot be hallucinated.

## Daily job

Deployed as a Docker cron job (Railway / Render / Fly.io — schedule `0 6 * * *`). Mount a volume at `/app/data` so `state.json` survives between runs.

- Job logs: **[link to Railway/Render logs — fill in after deploy]**

## Sanity check

`main.py` ends by asking the assistant *"How do I add a YouTube video?"* and prints the cited answer.

![Sanity check screenshot](sample/sample-chatbot.png)
