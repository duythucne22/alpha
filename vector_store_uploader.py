#!/usr/bin/env python3
"""Task 2 — OpenAI vector store + OptiBot assistant.

Provides:
- smart_hybrid_chunking(): structure-aware chunking (H2/H3 headers first,
  recursive character split for oversized sections). Used to log/estimate the
  number of chunks embedded — OpenAI's file_search does its own server-side
  chunking on upload.
- Vector store / file upload / assistant helpers that REUSE existing resources
  via ids persisted in state, instead of creating new ones every run.
- run_sanity_check(): asks "How do I add a YouTube video?" and prints the answer.

Can be run standalone: uploads every contents/*.md, creates/updates the
assistant, then runs the sanity check.
"""

import glob
import logging
import os
import re
import time

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from openai import OpenAI, OpenAIError

logger = logging.getLogger(__name__)

VECTOR_STORE_NAME = "OptiSigns Knowledge Base"
ASSISTANT_NAME = "OptiBot"
ASSISTANT_MODEL = "gpt-4o-mini"
SANITY_QUESTION = "How do I add a YouTube video?"
UPLOAD_DELAY = 0.3  # seconds between uploads, plays nice with OpenAI rate limits

SYSTEM_PROMPT = """You are OptiBot, the customer-support bot for OptiSigns.com.
• Tone: helpful, factual, concise.
• Only answer using the uploaded docs.
• Max 5 bullet points; else link to the doc.
• Cite up to 3 "Article URL:" lines per reply."""

CHUNK_SIZE = 3000
CHUNK_OVERLAP = 150

# OpenAI file_search re-chunks server-side; pass explicit params so chunk size
# and overlap are a deliberate choice, not an inherited default. The local
# splitter above stays as validation + logging.
SERVER_CHUNKING_STRATEGY = {
    "type": "static",
    "static": {"max_chunk_size_tokens": 800, "chunk_overlap_tokens": 200},
}


# ==========================================================================
# Chunking — hybrid structure-aware
# ==========================================================================

def smart_hybrid_chunking(content: str) -> list[Document]:
    """Split article markdown into chunks along its structure.

    1. Pull title + source URL out of the metadata header, strip the header.
    2. Split on H2/H3 headings (MarkdownHeaderTextSplitter) so chunks follow
       the article's own structure.
    3. Any section longer than CHUNK_SIZE chars gets a recursive character
       split (overlap keeps context across the seam).
    4. Every chunk carries article_title + source_url metadata.
    """
    title_match = re.match(r"^#\s+(.+)", content or "")
    article_title = title_match.group(1).strip() if title_match else ""
    source_match = re.search(r"\*\*Source:\*\*\s*(\S+)", content or "")
    source_url = source_match.group(1).strip() if source_match else ""

    # strip metadata header: leading H1, Source/Article URL lines, first ---
    body = re.sub(r"^#\s+.*\n+", "", content or "", count=1)
    body = re.sub(r"\*\*Source:\*\*.*\n+", "", body)
    body = re.sub(r"^Article URL:.*\n+", "", body, flags=re.MULTILINE)
    body = re.sub(r"^---\n+", "", body, count=1, flags=re.MULTILINE)
    body = body.strip()
    if not body:
        return []

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("##", "h2"), ("###", "h3")]
    )
    rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    base_metadata = {"article_title": article_title, "source_url": source_url}
    chunks: list[Document] = []
    for section in md_splitter.split_text(body):
        text = section.page_content.strip()
        if not text:
            continue
        metadata = {**base_metadata, **section.metadata}
        if len(text) > CHUNK_SIZE:
            for piece in rec_splitter.split_text(text):
                chunks.append(Document(page_content=piece, metadata=dict(metadata)))
        else:
            chunks.append(Document(page_content=text, metadata=metadata))
    return chunks


# ==========================================================================
# Vector store + files
# ==========================================================================

def get_or_create_vector_store(client: OpenAI, state: dict) -> str:
    """Reuse the vector store id saved in state; create one only if missing."""
    vs_id = state.get("vector_store_id")
    if vs_id:
        try:
            client.vector_stores.retrieve(vs_id)
            logger.info("Reusing vector store %s", vs_id)
            return vs_id
        except OpenAIError:
            logger.warning("Saved vector store %s not found, creating a new one", vs_id)

    vector_store = client.vector_stores.create(name=VECTOR_STORE_NAME)
    state["vector_store_id"] = vector_store.id
    logger.info("Created vector store %s", vector_store.id)
    return vector_store.id


def upload_markdown_file(client: OpenAI, vector_store_id: str, path: str) -> str:
    """Upload one .md file and attach it to the vector store. Returns file id."""
    with open(path, "rb") as f:
        openai_file = client.files.create(file=f, purpose="assistants")
    client.vector_stores.files.create(
        vector_store_id=vector_store_id,
        file_id=openai_file.id,
        chunking_strategy=SERVER_CHUNKING_STRATEGY,
    )
    return openai_file.id


def find_vector_store_by_name(client: OpenAI, name: str = VECTOR_STORE_NAME) -> str | None:
    """Locate an existing vector store by name (used to re-adopt the store when
    state was lost, instead of silently creating a parallel one)."""
    try:
        for store in client.vector_stores.list(limit=100):
            if store.name == name:
                return store.id
    except OpenAIError as exc:
        logger.warning("Could not list vector stores: %s", exc)
    return None


def store_file_count(client: OpenAI, vector_store_id: str) -> int:
    try:
        return client.vector_stores.retrieve(vector_store_id).file_counts.total
    except OpenAIError as exc:
        logger.warning("Could not read file count for %s: %s", vector_store_id, exc)
        return 0


def wipe_vector_store(client: OpenAI, vector_store_id: str) -> int:
    """Detach + delete every file in the store (FORCE_RESYNC recovery path)."""
    removed = 0
    for vs_file in client.vector_stores.files.list(vector_store_id=vector_store_id):
        remove_file(client, vector_store_id, vs_file.id)
        removed += 1
    logger.info("Wiped %d files from vector store %s", removed, vector_store_id)
    return removed


def remove_file(client: OpenAI, vector_store_id: str, file_id: str) -> None:
    """Detach + delete a previously uploaded file (used when an article changed,
    so the store never holds two versions of the same doc)."""
    try:
        client.vector_stores.files.delete(
            vector_store_id=vector_store_id, file_id=file_id
        )
    except OpenAIError as exc:
        logger.warning("Could not detach file %s from store: %s", file_id, exc)
    try:
        client.files.delete(file_id)
    except OpenAIError as exc:
        logger.warning("Could not delete file %s: %s", file_id, exc)


# ==========================================================================
# Assistant
# ==========================================================================

def get_or_create_assistant(client: OpenAI, state: dict, vector_store_id: str) -> str:
    """Reuse the assistant saved in state (refreshing its prompt + store binding);
    create it only on first run."""
    tool_resources = {"file_search": {"vector_store_ids": [vector_store_id]}}
    assistant_id = state.get("assistant_id")

    if assistant_id:
        try:
            client.beta.assistants.update(
                assistant_id,
                instructions=SYSTEM_PROMPT,
                model=ASSISTANT_MODEL,
                tools=[{"type": "file_search"}],
                tool_resources=tool_resources,
            )
            logger.info("Updated existing assistant %s", assistant_id)
            return assistant_id
        except OpenAIError:
            logger.warning("Saved assistant %s not found, creating a new one", assistant_id)

    assistant = client.beta.assistants.create(
        name=ASSISTANT_NAME,
        model=ASSISTANT_MODEL,
        instructions=SYSTEM_PROMPT,
        tools=[{"type": "file_search"}],
        tool_resources=tool_resources,
    )
    state["assistant_id"] = assistant.id
    logger.info("Created assistant %s", assistant.id)
    return assistant.id


def run_sanity_check(client: OpenAI, assistant_id: str,
                     question: str = SANITY_QUESTION,
                     url_by_file_id: dict[str, str] | None = None) -> str | None:
    """Ask the assistant one question and return/print its answer with citations.

    Citations are resolved deterministically from file_search annotations:
    the retrieved chunk's file_id maps back to the article URL (via state),
    so cited URLs can never be hallucinated."""
    logger.info("Sanity check: %r", question)
    try:
        thread = client.beta.threads.create()
        client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=question
        )
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id, assistant_id=assistant_id
        )
        if run.status != "completed":
            logger.error("Sanity check run ended with status: %s", run.status)
            return None

        messages = client.beta.threads.messages.list(thread_id=thread.id)
        text = messages.data[0].content[0].text
        answer = text.value
        # resolve citation markers deterministically via annotations
        cited_urls: list[str] = []
        for annotation in text.annotations or []:
            file_citation = getattr(annotation, "file_citation", None)
            if not file_citation:
                continue
            url = (url_by_file_id or {}).get(file_citation.file_id)
            if url:
                if url not in cited_urls:
                    cited_urls.append(url)
                answer = answer.replace(annotation.text, "")
            else:
                cited = client.files.retrieve(file_citation.file_id)
                answer = answer.replace(annotation.text, f" [{cited.filename}]")
        if cited_urls:
            answer = answer.rstrip() + "\n\n" + "\n".join(
                f"Article URL: {u}" for u in cited_urls[:3]
            )

        print("\n" + "=" * 60)
        print(f"Q: {question}")
        print("-" * 60)
        print(answer)
        print("=" * 60)
        return answer
    except OpenAIError as exc:
        logger.error("Sanity check failed: %s", exc)
        return None


# ==========================================================================
# Standalone: full upload of contents/*.md
# ==========================================================================

def upload_all(contents_dir: str = "contents") -> None:
    client = OpenAI()
    state: dict = {}
    vs_id = get_or_create_vector_store(client, state)

    total_files = 0
    total_chunks = 0
    for path in sorted(glob.glob(os.path.join(contents_dir, "*.md"))):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        chunks = smart_hybrid_chunking(content)
        try:
            upload_markdown_file(client, vs_id, path)
        except OpenAIError as exc:
            logger.error("Upload failed for %s: %s", path, exc)
            continue
        total_files += 1
        total_chunks += len(chunks)
        logger.info("  uploaded %s (%d chunks)", os.path.basename(path), len(chunks))
        time.sleep(UPLOAD_DELAY)

    logger.info("Embedded %d files, ~%d chunks (local estimate)", total_files, total_chunks)

    assistant_id = get_or_create_assistant(client, state, vs_id)
    print(f"vector_store_id={vs_id}")
    print(f"assistant_id={assistant_id}")
    run_sanity_check(client, assistant_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from dotenv import load_dotenv
    load_dotenv()
    upload_all()
