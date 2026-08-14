from finagent.rag.chunking import chunk_filing, chunk_text, split_into_sections

FILING_TEXT = """Item 1. Business
We design things.

Item 1A. Risk Factors
Competition could adversely impact our market share and financial results.
We face intense competition across all our target markets.

Item 2. Properties
Our headquarters is in Santa Clara, California.
"""


def test_split_into_sections_finds_item_headers():
    sections = split_into_sections(FILING_TEXT)
    titles = [s["section"] for s in sections]
    assert titles == ["Item 1. Business", "Item 1A. Risk Factors", "Item 2. Properties"]
    assert "Santa Clara" in sections[2]["text"]
    assert "Competition" in sections[1]["text"]


def test_split_into_sections_falls_back_without_enough_headers():
    sections = split_into_sections("just some plain text with no item headers at all")
    assert len(sections) == 1
    assert sections[0]["section"] is None


def test_chunk_text_returns_whole_text_when_under_chunk_size():
    assert chunk_text("short text", chunk_size=1200) == ["short text"]


def test_chunk_text_returns_empty_list_for_blank_text():
    assert chunk_text("   ") == []


def test_chunk_text_splits_long_text_with_overlap_and_no_mid_word_breaks():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert not chunk.startswith(" ")
        assert not chunk.endswith(" ")
    # every chunk boundary lands on a real word, never mid-token
    for chunk in chunks:
        for token in chunk.split():
            assert token.startswith("word")


def test_chunk_filing_tags_each_chunk_with_its_section():
    chunks = chunk_filing(FILING_TEXT, chunk_size=1200)
    sections = {c["section"] for c in chunks}
    assert sections == {"Item 1. Business", "Item 1A. Risk Factors", "Item 2. Properties"}
    risk_chunk = next(c for c in chunks if c["section"] == "Item 1A. Risk Factors")
    assert "Competition" in risk_chunk["text"]
