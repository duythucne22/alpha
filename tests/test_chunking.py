"""Unit tests for the hybrid structure-aware chunking strategy."""

from vector_store_uploader import CHUNK_SIZE, smart_hybrid_chunking

SAMPLE = """# How to Add YouTube Videos

**Source:** https://support.example.com/articles/1-youtube

Article URL: https://support.example.com/articles/1-youtube

---

Intro paragraph.

## Step 1: Open the app

Do the first thing.

## Step 2: Paste the link

Do the second thing.

### Notes

Some notes here.
"""


class TestSmartHybridChunking:
    def test_splits_on_h2_headers(self):
        chunks = smart_hybrid_chunking(SAMPLE)
        assert len(chunks) >= 3  # intro + step1 + step2(+notes)

    def test_metadata_injected_into_every_chunk(self):
        for chunk in smart_hybrid_chunking(SAMPLE):
            assert chunk.metadata["article_title"] == "How to Add YouTube Videos"
            assert chunk.metadata["source_url"] == (
                "https://support.example.com/articles/1-youtube"
            )

    def test_metadata_header_stripped_from_content(self):
        joined = "\n".join(c.page_content for c in smart_hybrid_chunking(SAMPLE))
        assert "**Source:**" not in joined
        assert "Article URL:" not in joined

    def test_long_sections_are_split_with_overlap(self):
        long_section = "word " * 2000  # ~10000 chars, well over CHUNK_SIZE
        content = f"# T\n\n**Source:** https://x.com/1\n\n---\n\n## Big\n\n{long_section}"
        chunks = smart_hybrid_chunking(content)
        assert len(chunks) > 1
        assert all(len(c.page_content) <= CHUNK_SIZE for c in chunks)

    def test_empty_body_returns_no_chunks(self):
        assert smart_hybrid_chunking("# Title\n\n**Source:** https://x.com/1\n\n---\n\n") == []
