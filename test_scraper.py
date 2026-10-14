import pytest

import scraper

CHANNEL_ID = "UC" + "a" * 22


def test_handles_and_links_are_understood():
    assert scraper.parse_channel_ref("@alexruns") == ("handle", "@alexruns")
    assert scraper.parse_channel_ref("alexruns") == ("handle", "@alexruns")
    assert scraper.parse_channel_ref("https://www.youtube.com/@alexruns") == ("handle", "@alexruns")
    assert scraper.parse_channel_ref("youtube.com/@alexruns/videos") == ("handle", "@alexruns")


def test_channel_ids_are_understood():
    assert scraper.parse_channel_ref(CHANNEL_ID) == ("id", CHANNEL_ID)
    assert scraper.parse_channel_ref(f"https://youtube.com/channel/{CHANNEL_ID}") == ("id", CHANNEL_ID)


def test_bad_references_give_a_helpful_error():
    for bad in ["", "   ", "!!", "https://youtube.com/watch?v=abc123"]:
        with pytest.raises(scraper.DataSourceError):
            scraper.parse_channel_ref(bad)


def test_csv_loads_with_common_column_names():
    data = b"Comment,Author,Likes,Date\nLove the pacing in this one,ann,3,2026-01-01\n"
    df = scraper.load_comments_csv(data)
    assert list(df.columns) == scraper.COMMENT_COLUMNS
    assert df["text"].iloc[0] == "Love the pacing in this one"
    assert df["author_hash"].iloc[0] not in (None, "ann")   # identities are hashed


def test_csv_only_needs_a_text_column():
    df = scraper.load_comments_csv(b"text\nJust a comment\n")
    assert len(df) == 1 and df["author_hash"].iloc[0] is None


def test_csv_without_a_comment_column_is_rejected():
    with pytest.raises(scraper.DataSourceError, match="comment column"):
        scraper.load_comments_csv(b"foo,bar\n1,2\n")


def test_oversized_csv_is_rejected():
    with pytest.raises(scraper.DataSourceError, match="too large"):
        scraper.load_comments_csv(b"a" * (scraper.MAX_CSV_BYTES + 1))
