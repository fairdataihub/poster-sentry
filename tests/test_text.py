"""Tests for first-page text extraction."""

import sys

from poster_sentry.features import extract_first_page_text


def test_vendored_text_path_needs_no_deep_learning_stack():
    import poster_sentry.p2j  # noqa: F401
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules


def test_unreadable_pdf_gives_empty_text(tmp_path):
    bad = tmp_path / "not_a.pdf"
    bad.write_bytes(b"not a pdf")
    assert extract_first_page_text(str(bad)) == ""
