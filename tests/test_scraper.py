"""Unit tests for scraper.py — all offline, no network calls."""

from scraper import (
    build_markdown,
    extract_slug_from_url,
    html_to_markdown,
    slug_for,
)


class TestSlugExtraction:
    def test_extracts_slug_from_html_url(self):
        url = ("https://support.optisigns.com/hc/en-us/articles/"
               "53216900345875-What-Access-Does-OptiSigns-Have")
        assert extract_slug_from_url(url) == "what-access-does-optisigns-have"

    def test_returns_none_for_url_without_dash(self):
        assert extract_slug_from_url("https://x.com/articles/12345") is None

    def test_returns_none_for_empty_url(self):
        assert extract_slug_from_url("") is None

    def test_prefers_zendesk_slug_field(self):
        article = {"slug": "My-Custom-Slug", "html_url": "https://x.com/articles/1-Other"}
        assert slug_for(article) == "my-custom-slug"

    def test_falls_back_to_url_when_slug_missing(self):
        article = {"html_url": "https://x.com/articles/1-From-The-Url"}
        assert slug_for(article) == "from-the-url"


class TestHtmlToMarkdown:
    def test_converts_headings_and_lists(self):
        html = "<h2>Setup</h2><ul><li>Step one</li><li>Step two</li></ul>"
        md = html_to_markdown(html)
        assert "## Setup" in md
        assert "Step one" in md and "Step two" in md

    def test_preserves_links(self):
        html = '<p>See <a href="/hc/en-us/articles/123-guide">the guide</a>.</p>'
        md = html_to_markdown(html)
        assert "[the guide](/hc/en-us/articles/123-guide)" in md

    def test_strips_script_and_style(self):
        html = "<p>Visible</p><script>alert(1)</script><style>p{}</style>"
        md = html_to_markdown(html)
        assert "Visible" in md
        assert "alert" not in md and "p{}" not in md

    def test_collapses_excessive_blank_lines(self):
        html = "<p>a</p><br><br><br><br><p>b</p>"
        assert "\n\n\n" not in html_to_markdown(html)


class TestBuildMarkdown:
    def test_header_contains_title_source_and_article_url(self):
        article = {"title": "How to X", "html_url": "https://support.example.com/a/1-x"}
        content = build_markdown(article, "Body text.")
        assert content.startswith("# How to X\n\n")
        assert "**Source:** https://support.example.com/a/1-x" in content
        assert "Article URL: https://support.example.com/a/1-x" in content
        assert "---" in content
        assert content.rstrip().endswith("Body text.")
