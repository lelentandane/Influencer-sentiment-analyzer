"""scraper.py - load creator data from the YouTube Data API v3 or from a comments CSV.

Both sources produce the same DataFrame layout, so the rest of the pipeline
does not care where the data came from. Commenter identities are hashed on
ingest and raw names are never kept.
"""

import hashlib
import io
import os
import re
from urllib.parse import urlparse

import pandas as pd
import requests

API_BASE = "https://www.googleapis.com/youtube/v3"
TIMEOUT = 10
MAX_CSV_BYTES = 5 * 1024 * 1024
MAX_CSV_ROWS = 5000

COMMENT_COLUMNS = ["comment_id", "video_id", "author_hash", "text", "like_count", "published_at"]
VIDEO_COLUMNS = ["video_id", "title", "published_at", "view_count", "like_count", "comment_count"]

CHANNEL_ID_RE = re.compile(r"^UC[\w-]{22}$")
HANDLE_RE = re.compile(r"^@?[\w.\-]{3,30}$")

ERROR_MESSAGES = {
    "quotaExceeded": "The daily YouTube API quota is used up. Try again tomorrow.",
    "keyInvalid": "The YouTube API key was rejected.",
    "forbidden": "YouTube refused the request.",
}

CSV_ALIASES = {
    "text": ["text", "comment", "comment_text", "body", "content"],
    "author": ["author", "username", "user", "author_name"],
    "like_count": ["like_count", "likes", "likecount"],
    "published_at": ["published_at", "date", "timestamp", "created_at"],
}


class DataSourceError(Exception):
    def __init__(self, message: str, reason: str | None = None):
        super().__init__(message)
        self.reason = reason


# ---------- low-level helpers ----------

def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _to_int(value):
    return None if value is None else int(value)


def _api_get(endpoint: str, **params) -> dict:
    key = os.getenv("YOUTUBE_API_KEY")
    if not key:
        raise DataSourceError("YOUTUBE_API_KEY is not set on the server.", "missingKey")
    try:
        resp = requests.get(f"{API_BASE}/{endpoint}", params={**params, "key": key}, timeout=TIMEOUT)
    except requests.RequestException:
        # Deliberately no exception text: it can contain the request URL and the API key.
        raise DataSourceError("Could not reach the YouTube API.", "network")

    if resp.status_code != 200:
        try:
            reason = resp.json()["error"]["errors"][0]["reason"]
        except (ValueError, KeyError, IndexError, TypeError):
            reason = None
        raise DataSourceError(
            ERROR_MESSAGES.get(reason, f"The YouTube API returned an error (status {resp.status_code})."),
            reason)
    try:
        return resp.json()
    except ValueError:
        raise DataSourceError("Unexpected response from the YouTube API.", "badResponse")


def parse_channel_ref(ref: str) -> tuple[str, str]:
    """Return ("handle", "@name") or ("id", "UC...") from a handle, ID or channel link."""
    ref = (ref or "").strip()
    if not ref:
        raise DataSourceError("Enter a YouTube handle or channel link.")

    if "youtube.com" in ref.lower() or "youtu.be" in ref.lower():
        parsed = urlparse(ref if "://" in ref else "https://" + ref)
        parts = [p for p in parsed.path.split("/") if p]
        if parts and parts[0].startswith("@"):
            return "handle", parts[0]
        if len(parts) >= 2 and parts[0] == "channel" and CHANNEL_ID_RE.match(parts[1]):
            return "id", parts[1]
        raise DataSourceError("Use a link like youtube.com/@handle or youtube.com/channel/UC...")

    if CHANNEL_ID_RE.match(ref):
        return "id", ref
    if HANDLE_RE.match(ref):
        return "handle", ref if ref.startswith("@") else "@" + ref
    raise DataSourceError("That does not look like a YouTube handle or channel link.")


# ---------- YouTube API ----------

def get_channel(ref: str) -> dict:
    kind, value = parse_channel_ref(ref)
    params = {"part": "snippet,statistics,contentDetails"}
    params["id" if kind == "id" else "forHandle"] = value
    items = _api_get("channels", **params).get("items", [])
    if not items:
        raise DataSourceError("Channel not found. Check the handle or link.", "notFound")

    item = items[0]
    stats = item.get("statistics", {})
    hidden = stats.get("hiddenSubscriberCount", False) or "subscriberCount" not in stats
    return {
        "channel_id": item["id"],
        "title": item["snippet"].get("title", ""),
        "description": item["snippet"].get("description", "")[:1000],
        "subscriber_count": None if hidden else int(stats["subscriberCount"]),
        "total_views": int(stats.get("viewCount", 0)),
        "video_count": int(stats.get("videoCount", 0)),
        "uploads_playlist": item["contentDetails"]["relatedPlaylists"]["uploads"],
    }


def get_recent_videos(uploads_playlist: str, n: int = 10) -> pd.DataFrame:
    n = max(1, min(n, 50))
    data = _api_get("playlistItems", part="contentDetails", playlistId=uploads_playlist, maxResults=n)
    ids = [i["contentDetails"]["videoId"] for i in data.get("items", [])]
    if not ids:
        return pd.DataFrame(columns=VIDEO_COLUMNS)

    details = _api_get("videos", part="snippet,statistics", id=",".join(ids)).get("items", [])
    rows = []
    for v in details:
        s = v.get("statistics", {})
        rows.append({
            "video_id": v["id"],
            "title": v["snippet"].get("title", ""),
            "published_at": v["snippet"].get("publishedAt"),
            "view_count": _to_int(s.get("viewCount")),
            "like_count": _to_int(s.get("likeCount")),       # missing if likes are hidden
            "comment_count": _to_int(s.get("commentCount")),  # missing if comments are off
        })
    df = pd.DataFrame(rows, columns=VIDEO_COLUMNS)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)
    for col in ("view_count", "like_count", "comment_count"):
        df[col] = pd.to_numeric(df[col])
    return df.sort_values("published_at", ascending=False).reset_index(drop=True)


def get_comments(video_id: str, max_comments: int = 100) -> pd.DataFrame:
    """Newest top-level comments for a video (replies are not included)."""
    max_comments = max(1, min(max_comments, 500))
    rows, token = [], None
    while len(rows) < max_comments:
        params = dict(part="snippet", videoId=video_id, order="time", textFormat="plainText",
                      maxResults=min(100, max_comments - len(rows)))
        if token:
            params["pageToken"] = token
        try:
            data = _api_get("commentThreads", **params)
        except DataSourceError as e:
            if e.reason == "commentsDisabled":
                return pd.DataFrame(columns=COMMENT_COLUMNS)
            raise

        for item in data.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]
            author = (s.get("authorChannelId") or {}).get("value") or s.get("authorDisplayName", "")
            rows.append({
                "comment_id": item["id"],
                "video_id": video_id,
                "author_hash": _hash(author) if author else None,
                "text": s.get("textDisplay", ""),
                "like_count": s.get("likeCount", 0),
                "published_at": s.get("publishedAt"),
            })
        token = data.get("nextPageToken")
        if not token:
            break
    return pd.DataFrame(rows, columns=COMMENT_COLUMNS)


def fetch_creator_data(ref: str, n_videos: int = 10, comment_videos: int = 5,
                       comments_per_video: int = 100) -> dict:
    """Channel info, recent videos and a comment sample.
    Costs roughly 3 + comment_videos YouTube API quota units."""
    channel = get_channel(ref)
    videos = get_recent_videos(channel["uploads_playlist"], n_videos)
    frames = [get_comments(vid, comments_per_video)
              for vid in videos["video_id"].head(max(0, comment_videos))]
    frames = [f for f in frames if not f.empty]
    comments = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COMMENT_COLUMNS)
    return {"channel": channel, "videos": videos, "comments": comments}


# ---------- CSV fallback (any platform) ----------

def load_comments_csv(data: bytes) -> pd.DataFrame:
    """Load comments exported from any platform. Only a text column is required."""
    if len(data) > MAX_CSV_BYTES:
        raise DataSourceError("That file is too large (limit 5 MB).")
    try:
        raw = pd.read_csv(io.BytesIO(data), dtype=str, nrows=MAX_CSV_ROWS, on_bad_lines="skip")
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
        raise DataSourceError("Could not read that file as a CSV.")

    raw.columns = [str(c).strip().lower() for c in raw.columns]

    def pick(name):
        return next((c for c in CSV_ALIASES[name] if c in raw.columns), None)

    text_col = pick("text")
    if text_col is None:
        raise DataSourceError("The CSV needs a comment column named one of: "
                              + ", ".join(CSV_ALIASES["text"]) + ".")
    author_col, likes_col, date_col = pick("author"), pick("like_count"), pick("published_at")

    out = pd.DataFrame({
        "comment_id": [f"csv-{i}" for i in range(len(raw))],
        "video_id": None,
        "author_hash": (raw[author_col].fillna("").map(lambda a: _hash(a.strip().lower()) if a.strip() else None)
                        if author_col else None),
        "text": raw[text_col],
        "like_count": raw[likes_col] if likes_col else 0,
        "published_at": raw[date_col] if date_col else None,
    }, columns=COMMENT_COLUMNS)
    return out
