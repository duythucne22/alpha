# Tasks

## 1. Scrape ⇒ Markdown

**Goal:** prove you can ingest messy web content and normalize it.

- [x] Pull ≥ 30 articles from support.optisigns.com. *(405 articles scraped, 0 failed — [scraper.py](scraper.py))*
- [x] Convert each article to clean Markdown.
- [x] Save each file as `<slug>.md` (or whatever scheme you like). *(Zendesk's own slug field, URL-parse fallback — `contents/<slug>.md`)*
- [x] Preserve relative links, code blocks, and headings. Remove nav/ads.
- [x] Hint: you can use Zendesk API to read the article. *(list endpoint used directly — full body included, detail API kept only as fallback)*

---

## 2. Build AI Assistant & Programmatically Load Vector Store

- [x] API upload is mandatory—no UI drag-and-drop here. *(all uploads via `vector_store_uploader.py`)*
- [x] Create the Assistant – Use OpenAI Playground UI or Google AI Studio to set up your assistant. *(created programmatically via API, `get_or_create_assistant()`)*
- [x] System prompt (verbatim):
  ```
  You are OptiBot, the customer-support bot for OptiSigns.com.
  • Tone: helpful, factual, concise.
  • Only answer using the uploaded docs.
  • Max 5 bullet points; else link to the doc.
  • Cite up to 3 "Article URL:" lines per reply.
  ```
- [x] Via Python script, upload Markdown files to your chosen AI service's vector store/knowledge base via API (OpenAI Vector Store or Google Gemini equivalent).
- [x] Chunking strategy is up to you; just explain it in the README. *(hybrid structure-aware — documented in [README.md](README.md#chunking-strategy--hybrid-structure-aware))*
- [x] Log how many files and chunks were embedded. *("Embedded 405 files, ~2900 chunks this run" — [main.py:203-204](main.py#L203-L204))*
- [x] Quick sanity check – Test your assistant in the Playground (OpenAI) or AI Studio (Gemini), and ask: "How do I add a YouTube video?"
- [x] Take a screenshot showing a correct answer with citations.

---

## 3. Deploy Scraper as Daily Job

- [x] Wrap your scraper-uploader in `main.py`.
- [x] Dockerize (`Dockerfile`).
- [x] Schedule it to run once per day on DigitalOcean or any cloud/public hosting platform (e.g. Railway, Render, Fly.io, AWS, GCP). *(Railway, cron `0 6 * * *` — `railway.toml`, `railway.cron.json`)*
- [x] Job must re-scrape.
- [x] Job must detect new/updated articles (hash, Last-Modified, etc.). *(Zendesk `updated_at` OR SHA-256 of rendered Markdown)*
- [x] Job must upload only the delta.
- [x] Log counts: added, updated, skipped. *(`Added: X, Updated: Y, Skipped: Z, Removed: W` every run)*
