import sentiment


def test_trust_score_combines_three_transparent_parts():
    result = sentiment.trust_from_parts(depth=0.8, bot_like=0.1, toxic=0.05)
    assert result["components"] == {"depth": 0.8, "authenticity": 0.9, "civility": 0.95}
    assert result["score"] == 86   # 100 * (0.5*0.8 + 0.35*0.9 + 0.15*0.95)


def test_perfect_and_worst_cases():
    assert sentiment.trust_from_parts(1, 0, 0)["score"] == 100
    assert sentiment.trust_from_parts(0, 1, 1)["score"] == 0


def test_weights_add_up_to_one():
    assert abs(sum(sentiment.TRUST_WEIGHTS.values()) - 1) < 1e-9
