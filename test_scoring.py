import scoring


def metrics(**overrides):
    base = {"videos_analyzed": 5, "views_per_subscriber": 0.15, "likes_per_1k_views": 20.0,
            "comments_per_1k_views": 3.0, "view_variability": 0.3, "median_views": 10000}
    base.update(overrides)
    return base


def test_engagement_health_scores_each_part_against_its_benchmark():
    result = scoring.engagement_health(metrics())
    assert result["parts"] == {"reach": 50, "interaction": 50, "conversation": 100, "consistency": 100}
    assert result["score"] == 75


def test_engagement_health_is_capped_at_100_per_part():
    result = scoring.engagement_health(metrics(views_per_subscriber=5, likes_per_1k_views=500))
    assert result["parts"]["reach"] == 100 and result["parts"]["interaction"] == 100


def test_engagement_health_needs_videos():
    assert scoring.engagement_health(None) is None
    assert scoring.engagement_health({"videos_analyzed": 0}) is None


def test_spiky_views_score_zero_for_consistency():
    result = scoring.engagement_health(metrics(view_variability=2.0))
    assert result["parts"]["consistency"] == 0


def test_cost_per_quality_engagement():
    trust = {"components": {"authenticity": 0.9}}
    result = scoring.cost_per_quality_engagement(
        500, metrics(likes_per_1k_views=40.0, comments_per_1k_views=4.0), trust)
    assert result["quality_engagements"] == 396.0   # 10000 * 0.044 * 0.9
    assert result["cost_per_quality_engagement"] == 1.26


def test_cost_needs_a_fee_and_data():
    trust = {"components": {"authenticity": 0.9}}
    assert scoring.cost_per_quality_engagement(None, metrics(), trust) is None
    assert scoring.cost_per_quality_engagement(0, metrics(), trust) is None
    assert scoring.cost_per_quality_engagement(500, None, trust) is None
    assert scoring.cost_per_quality_engagement(500, metrics(), None) is None


def test_overall_uses_weights_and_labels():
    result = scoring.overall(80, 60, None)
    assert result["score"] == 71   # (0.40*80 + 0.35*60) / 0.75
    assert result["label"] == "Promising, with caveats"
    assert "audience quality (80)" in result["summary"] and "brand fit (60)" in result["summary"]
    assert round(sum(result["weights"].values()), 2) == 1.0


def test_overall_label_thresholds():
    assert scoring.overall(90, 90, 90)["label"] == "Strong fit"
    assert scoring.overall(40, 40, 40)["label"] == "Proceed with caution"
    assert scoring.overall(10, 10, 10)["label"] == "Not recommended"


def test_overall_edge_cases():
    assert scoring.overall(None, None, None) is None
    assert scoring.overall(None, 90, None)["summary"] == "Only brand fit could be scored."
    assert scoring.overall(70, 70, 70)["summary"] == "All measured areas score about the same."
