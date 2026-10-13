"""sample_data.py - a clearly labeled demo report.

Lets anyone explore the dashboard without API keys. The numbers are invented,
but every score is computed with the real scoring functions so the sample is
internally consistent.
"""

import scoring
import sentiment


def build_sample() -> dict:
    counts = {"supportive": 57, "question": 21, "critical": 9, "toxic": 3, "generic": 50, "spam": 10}
    sentiments = {"positive": 93, "neutral": 47, "negative": 10}
    n = sum(counts.values())

    trust = sentiment.trust_from_parts(depth=(57 + 21 + 9) / n, bot_like=0.11, toxic=3 / n)
    audience = {
        "classified": n,
        "coverage": 1.0,
        "low_sample": False,
        "category_counts": counts,
        "category_shares": {k: round(v / n, 3) for k, v in counts.items()},
        "sentiment_shares": {k: round(v / n, 3) for k, v in sentiments.items()},
        "themes": ["beginner running shoes", "injury recovery", "race day nutrition", "training plans"],
        "trust": trust,
        "signals": {"n_comments": n, "unique_author_ratio": 0.96, "duplicate_ratio": 0.05,
                    "generic_ratio": 0.36, "spam_ratio": 0.06, "substantive_ratio": 0.41, "avg_words": 9.2},
    }

    metrics = {"videos_analyzed": 10, "median_views": 8300.0, "views_per_subscriber": 0.16,
               "likes_per_1k_views": 28.0, "comments_per_1k_views": 1.6,
               "posts_per_30_days": 4.4, "view_variability": 0.62}
    health = scoring.engagement_health(metrics)
    engagement = {"score": health["score"], "parts": health["parts"], "metrics": metrics}

    fit = {
        "fit_score": 82,
        "summary": ("The channel focuses on beginner-friendly running content, which closely matches a shoe "
                    "aimed at new runners. Recent titles cover training plans and gear, and the audience "
                    "often asks which shoes the creator uses."),
        "strengths": ["Recent videos are about training plans and gear for new runners",
                      "Commenters frequently ask which shoes the creator uses"],
        "risks": ["No recent video mentions sustainability, so the eco angle may need explaining",
                  "Most gear content covers road shoes, not trail"],
        "suggested_angle": ("A 4-week beginner plan video where the shoes are part of the plan, "
                            "with a short segment on the materials."),
        "evidence_strength": "strong",
    }

    return {
        "is_sample": True,
        "mode": "youtube",
        "creator": {"name": "Sample: Trail & Track Running", "subscribers": 52000, "videos": 214},
        "brand": {"description": "Eco-friendly running shoes for beginner runners", "target_audience": None},
        "overall": scoring.overall(trust["score"], fit["fit_score"], engagement["score"]),
        "audience": audience,
        "engagement": engagement,
        "brand_fit": fit,
        "cost": scoring.cost_per_quality_engagement(500, metrics, trust),
        "warnings": ["This is sample data for demonstration. It does not describe a real creator."],
    }
