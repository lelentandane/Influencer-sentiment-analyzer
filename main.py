"""main.py - FastAPI app. Run from backend/: uvicorn main:app --reload"""

import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles

import matcher
import sample_data
import scoring
import scraper
import sentiment
import utils

app = FastAPI(title="Micro-Influencer Audience Quality & Brand-Fit Analyzer")

# Simple in-memory limiter to protect API credits. Per process, resets on restart.
# Behind a reverse proxy every visitor shares the proxy's address, so use a real limiter there.
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_HOUR", "10"))
_hits: dict[str, deque] = defaultdict(deque)

SOURCE_STATUS = {"quotaExceeded": 429, "missingKey": 503, "network": 502,
                 "badResponse": 502, "notFound": 404}


def _check_rate_limit(ip: str) -> None:
    now = time.time()
    hits = _hits[ip]
    while hits and now - hits[0] > 3600:
        hits.popleft()
    if len(hits) >= RATE_LIMIT:
        raise HTTPException(429, "Too many analyses from this address. Please try again in an hour.")
    hits.append(now)


def _clean(value, name: str, max_len: int, required: bool = False, min_len: int = 0):
    value = (value or "").strip()
    if not value:
        if required:
            raise HTTPException(422, f"{name} is required.")
        return None
    if len(value) < min_len:
        raise HTTPException(422, f"{name} is too short. Add a little more detail.")
    if len(value) > max_len:
        raise HTTPException(422, f"{name} is too long (limit {max_len} characters).")
    return value


def _parse_fee(raw: Optional[str]):
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        fee = float(raw)
    except ValueError:
        raise HTTPException(422, "The fee must be a number.")
    if not (0 < fee <= 1e9):
        raise HTTPException(422, "The fee must be greater than zero.")
    return fee


def _require_services(mode: str) -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise HTTPException(503, "Live analysis needs an Anthropic API key configured on the server. "
                                 "You can still view the sample report.")
    if mode == "youtube" and not os.getenv("YOUTUBE_API_KEY"):
        raise HTTPException(503, "Analyzing a YouTube channel needs a YouTube API key configured on the "
                                 "server. You can still view the sample report.")


@app.get("/api/status")
def status():
    return {"youtube": bool(os.getenv("YOUTUBE_API_KEY")), "llm": bool(os.getenv("ANTHROPIC_API_KEY"))}


@app.get("/api/sample")
def sample():
    return sample_data.build_sample()


@app.post("/api/analyze")
def analyze(
    request: Request,
    mode: str = Form("youtube"),
    creator: Optional[str] = Form(None),
    creator_description: Optional[str] = Form(None),
    brand_description: str = Form(""),
    target_audience: Optional[str] = Form(None),
    fee: Optional[str] = Form(None),
    comments_file: Optional[UploadFile] = File(None),
):
    # 1. Validate input before spending anything
    if mode not in ("youtube", "csv"):
        raise HTTPException(422, "Mode must be 'youtube' or 'csv'.")
    brand_description = _clean(brand_description, "The product description", 500, required=True, min_len=10)
    target_audience = _clean(target_audience, "The target audience", 200)
    creator = _clean(creator, "The creator", 200, required=(mode == "youtube"))
    fee_value = _parse_fee(fee)
    if mode == "csv":
        creator_description = _clean(creator_description, "The creator description", 800,
                                     required=True, min_len=10)
        if comments_file is None:
            raise HTTPException(422, "Upload a CSV file of comments.")
    _require_services(mode)
    _check_rate_limit(request.client.host if request.client else "unknown")

    warnings: list[str] = []

    # 2. Load data
    if mode == "youtube":
        try:
            data = scraper.fetch_creator_data(creator)
        except scraper.DataSourceError as e:
            raise HTTPException(SOURCE_STATUS.get(e.reason, 400), str(e))
        channel, videos = data["channel"], data["videos"]
        comments = utils.clean_comments(data["comments"])
        metrics = utils.engagement_metrics(channel, videos)
        creator_info = {"name": channel["title"], "subscribers": channel["subscriber_count"],
                        "videos": channel["video_count"]}
        profile = {"name": channel["title"], "description": channel["description"],
                   "video_titles": videos["title"].head(10).tolist()}
    else:
        try:
            raw = comments_file.file.read(scraper.MAX_CSV_BYTES + 1)
            comments = utils.clean_comments(scraper.load_comments_csv(raw))
        except scraper.DataSourceError as e:
            raise HTTPException(400, str(e))
        metrics = None
        name = creator or "This creator"
        creator_info = {"name": name, "subscribers": None, "videos": None}
        profile = {"name": name, "description": creator_description, "video_titles": []}
        warnings.append("Engagement metrics and the cost estimate need channel data, so they are only "
                        "available when analyzing a YouTube channel.")

    # 3. Audience quality (LLM + rules)
    audience = None
    try:
        if comments.empty:
            warnings.append("No comments could be read, so audience quality was not scored.")
        else:
            audience = sentiment.analyze_comments(comments)
            audience["signals"] = utils.comment_signals(comments)
            if audience["low_sample"]:
                warnings.append(f"Only {audience['classified']} comments could be analyzed, so audience "
                                "scores are less reliable.")

        # 4. Brand fit (LLM)
        profile["themes"] = audience["themes"] if audience else []
        fit = matcher.analyze_fit(profile, {"description": brand_description,
                                            "target_audience": target_audience})
    except sentiment.AnalysisError as e:
        raise HTTPException(502, str(e))

    # 5. Scores
    health = scoring.engagement_health(metrics)
    engagement = {"score": health["score"], "parts": health["parts"], "metrics": metrics} if health else None
    trust = audience["trust"] if audience else None
    cost = scoring.cost_per_quality_engagement(fee_value, metrics, trust)
    if fee_value and cost is None and mode == "youtube":
        warnings.append("The cost estimate could not be calculated from the available data.")

    return {
        "is_sample": False,
        "mode": mode,
        "creator": creator_info,
        "brand": {"description": brand_description, "target_audience": target_audience},
        "overall": scoring.overall(trust["score"] if trust else None, fit["fit_score"],
                                   engagement["score"] if engagement else None),
        "audience": audience,
        "engagement": engagement,
        "brand_fit": fit,
        "cost": cost,
        "warnings": warnings,
    }


# Serve the dashboard from the same origin (no CORS setup). Must stay last.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
