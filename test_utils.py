import pandas as pd

import utils


def comments(texts, authors=None):
    n = len(texts)
    return pd.DataFrame({
        "comment_id": [f"c{i}" for i in range(n)],
        "text": texts,
        "author_hash": authors or [f"a{i}" for i in range(n)],
        "like_count": [0] * n,
        "published_at": [None] * n,
    })


def test_normalise_drops_links_punctuation_and_emoji():
    assert utils.normalise("Great VIDEO!!! https://example.com \U0001F525") == "great video"


def test_clean_comments_removes_blanks_duplicates_and_truncates():
    df = comments(["  hello   there  ", "", "x" * 900, "same id"])
    df.loc[3, "comment_id"] = "c0"
    out = utils.clean_comments(df)
    assert list(out["text"][:1]) == ["hello there"]
    assert len(out) == 2                       # blank and duplicate id removed
    assert out["text"].str.len().max() == utils.MAX_COMMENT_CHARS


def test_text_features_flag_each_kind_of_comment():
    df = comments([
        "first",
        "Check my channel for free gifts",
        "I tried the interval plan from this video and my 5k time dropped by a minute",
        "This is the same copied template comment",
        "This is the same copied template comment",
    ])
    f = utils.add_text_features(df)
    assert f["is_generic"].tolist() == [True, False, False, False, False]
    assert f["is_spam"].tolist() == [False, True, False, False, False]
    assert f["is_duplicate"].tolist() == [False, False, False, True, True]
    assert f["is_substantive"].tolist() == [False, False, True, False, False]


def test_links_count_as_spam():
    f = utils.add_text_features(comments(["visit www.example.com now for a surprise prize"]))
    assert f["is_spam"].iloc[0]


def test_text_features_handle_an_empty_frame():
    assert utils.add_text_features(comments([])).empty


def test_comment_signals():
    df = comments(["first", "first", "A real comment with enough words to count as substantive here"],
                  authors=["a", "a", "b"])
    s = utils.comment_signals(df)
    assert s["n_comments"] == 3
    assert s["unique_author_ratio"] == 0.667
    assert s["generic_ratio"] == 0.667
    assert s["substantive_ratio"] == 0.333
    assert utils.comment_signals(comments([])) == {"n_comments": 0}


def test_engagement_metrics():
    videos = pd.DataFrame({
        "view_count": [1000, 2000, 3000], "like_count": [40, 80, 120], "comment_count": [4, 8, 12],
        "published_at": pd.to_datetime(["2026-09-01", "2026-09-08", "2026-09-15"], utc=True),
    })
    m = utils.engagement_metrics({"subscriber_count": 1000}, videos)
    assert m["videos_analyzed"] == 3
    assert m["median_views"] == 2000
    assert m["views_per_subscriber"] == 2.0
    assert m["likes_per_1k_views"] == 40.0
    assert m["comments_per_1k_views"] == 4.0
    assert m["posts_per_30_days"] == 4.3
    assert m["view_variability"] == 0.5


def test_engagement_metrics_with_no_videos():
    empty = pd.DataFrame(columns=["view_count", "like_count", "comment_count", "published_at"])
    assert utils.engagement_metrics({"subscriber_count": 10}, empty) == {"videos_analyzed": 0}
