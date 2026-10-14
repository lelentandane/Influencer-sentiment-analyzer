import copy

import pytest
from fastapi.testclient import TestClient

import main
import sample_data

client = TestClient(main.app)

CSV = (b"comment,author,likes,date\n"
       b"I tried the interval plan and my 5k time dropped by a minute,ann,12,2026-09-28\n"
       b"What shoes are you wearing in the second half of the video?,bob,5,2026-09-28\n")
FORM = {"mode": "csv", "brand_description": "Eco-friendly running shoes",
        "creator_description": "Beginner running tips and gear reviews"}


def post_csv(**overrides):
    return client.post("/api/analyze", data={**FORM, **overrides},
                       files={"comments_file": ("c.csv", CSV, "text/csv")})


@pytest.fixture(autouse=True)
def reset_limiter():
    main._hits.clear()
    yield
    main._hits.clear()


@pytest.fixture
def live(monkeypatch):
    """Pretend the keys exist and replace the two AI calls with the sample results."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    sample = sample_data.build_sample()
    monkeypatch.setattr(main.sentiment, "analyze_comments", lambda df: copy.deepcopy(sample["audience"]))
    monkeypatch.setattr(main.matcher, "analyze_fit", lambda profile, brand: copy.deepcopy(sample["brand_fit"]))


def test_status_reports_which_keys_are_set(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    assert client.get("/api/status").json() == {"youtube": False, "llm": False}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert client.get("/api/status").json() == {"youtube": False, "llm": True}


def test_sample_report_needs_no_keys(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    body = client.get("/api/sample").json()
    assert body["is_sample"] is True
    assert 0 <= body["overall"]["score"] <= 100


def test_input_is_validated_before_anything_is_spent(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert post_csv(mode="nope").status_code == 422
    assert post_csv(brand_description="shoes").status_code == 422          # too short
    assert post_csv(fee="abc").status_code == 422
    assert post_csv(fee="-5").status_code == 422
    no_file = client.post("/api/analyze", data=FORM)
    assert no_file.status_code == 422
    no_creator = client.post("/api/analyze", data={"mode": "youtube", "brand_description": "Eco-friendly shoes"})
    assert no_creator.status_code == 422


def test_missing_key_points_to_the_sample(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    res = post_csv()
    assert res.status_code == 503
    assert "sample" in res.json()["detail"].lower()


def test_youtube_mode_needs_a_youtube_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    res = client.post("/api/analyze", data={"mode": "youtube", "creator": "@alexruns",
                                            "brand_description": "Eco-friendly running shoes"})
    assert res.status_code == 503 and "YouTube" in res.json()["detail"]


def test_csv_analysis_returns_a_report_without_engagement(live):
    res = post_csv(fee="500")
    assert res.status_code == 200
    body = res.json()
    assert body["mode"] == "csv" and body["is_sample"] is False
    assert body["engagement"] is None and body["cost"] is None
    assert body["overall"]["score"] is not None
    assert any("Engagement metrics" in w for w in body["warnings"])


def test_bad_csv_returns_400(live):
    res = client.post("/api/analyze", data=FORM, files={"comments_file": ("c.csv", b"foo,bar\n1,2\n", "text/csv")})
    assert res.status_code == 400


def test_rate_limit_returns_429(live, monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMIT", 1)
    assert post_csv().status_code == 200
    assert post_csv().status_code == 429
