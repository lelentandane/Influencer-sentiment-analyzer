"""utils.py - comment cleaning and rule-based quality signals.

No scoring and no LLM calls here. These are plain, reproducible measurements
that later modules combine into scores.
"""

import re

import pandas as pd

MAX_COMMENT_CHARS = 500
MIN_WORDS_FOR_DUP = 3        # short repeats like "first" are generic, not necessarily bots
SUBSTANTIVE_MIN_WORDS = 8

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
NON_WORD_RE = re.compile(r"[^\w\s]")   # removes punctuation and emoji
SPACE_RE = re.compile(r"\s+")

SPAM_PHRASES = (
    "sub4sub", "sub for sub", "check my channel", "check out my channel", "follow me",
    "dm me", "telegram", "whatsapp", "giveaway", "free gift", "click my profile",
)
GENERIC_PHRASES = {
    "first", "nice", "nice video", "great video", "good video", "nice one", "love it",
    "love this", "wow", "amazing", "awesome", "cool", "great", "good", "thanks",
    "thank you", "fire", "lit", "omg", "lol", "best", "hi", "hello",
}


def normalise(text) -> str:
    """Lowercase, drop URLs, punctuation and emoji, collapse spaces."""
    text = URL_RE.sub(" ", str(text).lower())
    return SPACE_RE.sub(" ", NON_WORD_RE.sub(" ", text)).strip()


def clean_comments(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["text"] = (out["text"].fillna("").astype(str)
                   .str.replace(r"\s+", " ", regex=True).str.strip().str[:MAX_COMMENT_CHARS])
    out = out[out["text"] != ""].drop_duplicates(subset="comment_id")
    out["like_count"] = pd.to_numeric(out["like_count"], errors="coerce").fillna(0).astype(int)
    out["published_at"] = pd.to_datetime(out["published_at"], errors="coerce", utc=True)
    return out.reset_index(drop=True)


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        for col in ("is_spam", "is_generic", "is_duplicate", "is_substantive"):
            out[col] = pd.Series(dtype=bool)
        out["norm"] = pd.Series(dtype=str)
        out["word_count"] = pd.Series(dtype=int)
        return out

    out["norm"] = out["text"].map(normalise)
    out["word_count"] = out["norm"].map(lambda s: len(s.split()))
    lowered = out["text"].str.lower()
    has_link = out["text"].map(lambda t: bool(URL_RE.search(t)))
    out["is_spam"] = has_link | lowered.map(lambda t: any(p in t for p in SPAM_PHRASES))
    out["is_generic"] = (out["norm"] == "") | (out["word_count"] <= 2) | out["norm"].isin(GENERIC_PHRASES)
    repeats = out["norm"].map(out["norm"].value_counts())
    out["is_duplicate"] = (repeats >= 2) & (out["word_count"] >= MIN_WORDS_FOR_DUP)
    out["is_substantive"] = ((out["word_count"] >= SUBSTANTIVE_MIN_WORDS)
                             & ~out["is_spam"] & ~out["is_duplicate"])
    return out


def _ratio(part, whole, nd=3):
    return round(float(part) / whole, nd)


def comment_signals(df: pd.DataFrame) -> dict:
    """Share of comments that look generic, duplicated, spammy or substantive."""
    n = len(df)
    if n == 0:
        return {"n_comments": 0}
    f = add_text_features(df)
    authors = f["author_hash"].dropna()
    return {
        "n_comments": n,
        "unique_author_ratio": _ratio(authors.nunique(), len(authors)) if len(authors) else None,
        "duplicate_ratio": _ratio(f["is_duplicate"].sum(), n),
        "generic_ratio": _ratio(f["is_generic"].sum(), n),
        "spam_ratio": _ratio(f["is_spam"].sum(), n),
        "substantive_ratio": _ratio(f["is_substantive"].sum(), n),
        "avg_words": round(float(f["word_count"].mean()), 1),
    }


def _num(x, nd=3):
    return None if x is None or pd.isna(x) else round(float(x), nd)


def engagement_metrics(channel: dict, videos: pd.DataFrame) -> dict:
    """Per-video performance proxies. Medians are used so one viral video does not dominate."""
    if videos.empty:
        return {"videos_analyzed": 0}

    views = videos["view_count"].astype(float)
    safe_views = views.where(views > 0)
    median_views = views.median()
    subs = channel.get("subscriber_count")

    dates = videos["published_at"].dropna().sort_values()
    posts_per_30d = None
    if len(dates) >= 2:
        span_days = (dates.iloc[-1] - dates.iloc[0]).total_seconds() / 86400
        if span_days > 0:
            posts_per_30d = (len(dates) - 1) / span_days * 30

    variability = views.std() / views.mean() if len(views) >= 3 and views.mean() > 0 else None

    return {
        "videos_analyzed": len(videos),
        "median_views": _num(median_views, 0),
        "views_per_subscriber": _num(median_views / subs, 3) if subs else None,
        "likes_per_1k_views": _num((videos["like_count"] / safe_views * 1000).median(), 1),
        "comments_per_1k_views": _num((videos["comment_count"] / safe_views * 1000).median(), 2),
        "posts_per_30_days": _num(posts_per_30d, 1),
        "view_variability": _num(variability, 2),   # high means a few spikes drive the average
    }
