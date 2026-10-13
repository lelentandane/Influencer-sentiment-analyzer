"""scoring.py - turn measurements into scores.

Benchmarks and weights are heuristic defaults chosen for readability, not
validated industry standards. Change them here.
"""

# A metric at or above its benchmark earns the full 100 for that part.
BENCHMARKS = {
    "views_per_subscriber": 0.30,
    "likes_per_1k_views": 40.0,
    "comments_per_1k_views": 3.0,
}
CV_BEST, CV_WORST = 0.3, 1.5     # view variability: at or below BEST = 100, at or above WORST = 0

WEIGHTS = {"trust": 0.40, "fit": 0.35, "engagement": 0.25}
LABELS = [(75, "Strong fit"), (55, "Promising, with caveats"), (35, "Proceed with caution"), (0, "Not recommended")]
NAMES = {"trust": "audience quality", "fit": "brand fit", "engagement": "engagement"}


def _scale(value, full):
    if value is None:
        return None
    return int(round(100 * min(max(value / full, 0), 1)))


def _consistency(cv):
    if cv is None:
        return None
    return int(round(100 * min(max((CV_WORST - cv) / (CV_WORST - CV_BEST), 0), 1)))


def engagement_health(metrics: dict | None) -> dict | None:
    """Average of the available parts, each scored 0 to 100 against a benchmark."""
    if not metrics or not metrics.get("videos_analyzed"):
        return None
    parts = {
        "reach": _scale(metrics.get("views_per_subscriber"), BENCHMARKS["views_per_subscriber"]),
        "interaction": _scale(metrics.get("likes_per_1k_views"), BENCHMARKS["likes_per_1k_views"]),
        "conversation": _scale(metrics.get("comments_per_1k_views"), BENCHMARKS["comments_per_1k_views"]),
        "consistency": _consistency(metrics.get("view_variability")),
    }
    present = [v for v in parts.values() if v is not None]
    if not present:
        return None
    return {"score": int(round(sum(present) / len(present))), "parts": parts}


def cost_per_quality_engagement(fee, metrics: dict | None, trust: dict | None) -> dict | None:
    """fee / (typical views x interaction rate x authenticity).

    Interaction rate is (likes + comments) per view. Authenticity discounts engagement
    that looks like spam or copy-pasted templates. It assumes a sponsored video performs
    like a typical recent video, which is a simplification."""
    if not fee or fee <= 0 or not metrics or not trust:
        return None
    views = metrics.get("median_views")
    likes = metrics.get("likes_per_1k_views")
    comments = metrics.get("comments_per_1k_views")
    if not views or likes is None or comments is None:
        return None

    interaction_rate = (likes + comments) / 1000
    authenticity = trust["components"]["authenticity"]
    quality = views * interaction_rate * authenticity
    if quality <= 0:
        return None
    return {
        "fee": float(fee),
        "expected_views": views,
        "interaction_rate": round(interaction_rate, 4),
        "authenticity": authenticity,
        "quality_engagements": round(quality, 1),
        "cost_per_quality_engagement": round(fee / quality, 2),
    }


def overall(trust: int | None, fit: int | None, engagement: int | None) -> dict | None:
    """Weighted average over the components that could be measured."""
    used = {k: v for k, v in {"trust": trust, "fit": fit, "engagement": engagement}.items() if v is not None}
    if not used:
        return None
    total = sum(WEIGHTS[k] for k in used)
    score = int(round(sum(v * WEIGHTS[k] for k, v in used.items()) / total))
    label = next(name for threshold, name in LABELS if score >= threshold)

    if len(used) == 1:
        (only,) = used
        summary = f"Only {NAMES[only]} could be scored."
    else:
        best, worst = max(used, key=used.get), min(used, key=used.get)
        summary = ("All measured areas score about the same." if used[best] == used[worst] else
                   f"Strongest area: {NAMES[best]} ({used[best]}). Weakest area: {NAMES[worst]} ({used[worst]}).")
    return {"score": score, "label": label, "summary": summary,
            "weights": {k: round(WEIGHTS[k] / total, 2) for k in used}}
