"""Tests for paragraph/sentence-aware chunking."""
from src.text_chunker import chunk_text, clean_text, count_words


def test_count_words():
    assert count_words("một hai ba") == 3
    assert count_words("") == 0


def test_clean_text_joins_hyphen_linebreaks():
    assert clean_text("exam-\nple") == "example"


def test_clean_text_drops_bare_page_numbers():
    out = clean_text("Đoạn văn.\n\n12\n\nĐoạn khác.")
    assert "12" not in out.split()


def test_clean_text_collapses_whitespace():
    assert clean_text("a    b\t\tc") == "a b c"
    assert clean_text("a\n\n\n\n\nb") == "a\n\nb"


def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_chunk_empty_returns_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_chunk_respects_paragraph_boundaries():
    paras = "\n\n".join(f"Đoạn {i} có vài từ ở đây." for i in range(10))
    chunks = chunk_text(paras, target_words=10, min_words=3)
    assert len(chunks) > 1
    # every chunk is non-empty
    assert all(c.strip() for c in chunks)


def test_chunk_splits_oversized_paragraph_by_sentence():
    # One giant paragraph (no blank lines) far above target → sentence split
    big = " ".join("Đây là câu số %d." % i for i in range(200))
    chunks = chunk_text(big, target_words=30)
    assert len(chunks) > 1
    for c in chunks:
        # allow a little slop but should be in the right ballpark
        assert count_words(c) <= 30 * 3


def test_chunk_no_word_loss_roughly():
    text = "\n\n".join("Câu một. Câu hai. Câu ba." for _ in range(20))
    chunks = chunk_text(text, target_words=15, min_words=5)
    total = sum(count_words(c) for c in chunks)
    # chunking shouldn't drop words (joins may merge whitespace only)
    assert total == count_words(clean_text(text))
