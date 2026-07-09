"""Unit tests for delta detection and state management in main.py."""

from main import (
    classify_article,
    deletion_is_suspicious,
    find_removed_ids,
    get_content_hash,
    load_state,
    save_state,
)


class TestClassifyArticle:
    ENTRY = {"updated_at": "2026-07-01T00:00:00Z", "hash": "abc123", "file_id": "file-1"}

    def test_unknown_article_is_added(self):
        assert classify_article(None, "2026-07-01T00:00:00Z", "abc123") == "added"

    def test_changed_timestamp_is_updated(self):
        assert classify_article(self.ENTRY, "2026-07-06T00:00:00Z", "abc123") == "updated"

    def test_changed_hash_is_updated(self):
        assert classify_article(self.ENTRY, "2026-07-01T00:00:00Z", "zzz999") == "updated"

    def test_identical_article_is_skipped(self):
        assert classify_article(self.ENTRY, "2026-07-01T00:00:00Z", "abc123") == "skipped"


class TestContentHash:
    def test_hash_is_deterministic(self):
        assert get_content_hash("hello") == get_content_hash("hello")

    def test_hash_changes_with_content(self):
        assert get_content_hash("hello") != get_content_hash("hello!")

    def test_handles_unicode(self):
        assert len(get_content_hash("tiếng Việt — 中文")) == 64  # sha256 hex


class TestReconciliation:
    def test_finds_ids_missing_from_scrape(self):
        assert find_removed_ids({"1", "2", "3"}, {"2", "3"}) == ["1"]

    def test_no_removals_when_all_present(self):
        assert find_removed_ids({"1", "2"}, {"1", "2", "9"}) == []

    def test_small_deletion_count_is_not_suspicious(self):
        # 5 or fewer missing is always allowed, even at high fractions
        assert deletion_is_suspicious(5, 10) is False

    def test_large_fraction_is_suspicious(self):
        # 100 of 400 known articles vanishing = broken scrape, not mass delete
        assert deletion_is_suspicious(100, 400) is True

    def test_large_count_but_small_fraction_is_fine(self):
        # 50 of 1000 (5%) is a plausible real cleanup
        assert deletion_is_suspicious(50, 1000) is False

    def test_empty_state_is_never_suspicious(self):
        assert deletion_is_suspicious(0, 0) is False


class TestStateRoundtrip:
    def test_missing_file_returns_skeleton(self, tmp_path):
        state = load_state(str(tmp_path / "nope.json"))
        assert state == {"vector_store_id": None, "assistant_id": None, "articles": {}}

    def test_corrupt_file_returns_skeleton(self, tmp_path):
        bad = tmp_path / "state.json"
        bad.write_text("{not json", encoding="utf-8")
        assert load_state(str(bad))["articles"] == {}

    def test_save_then_load_roundtrip(self, tmp_path):
        path = str(tmp_path / "data" / "state.json")  # parent dir auto-created
        state = {
            "vector_store_id": "vs_1",
            "assistant_id": "asst_1",
            "articles": {"42": {"slug": "x", "updated_at": "t", "hash": "h", "file_id": "f"}},
        }
        save_state(state, path)
        assert load_state(path) == state
