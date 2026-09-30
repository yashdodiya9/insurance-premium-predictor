import pytest
from app.model.predict import class_labels, predict_output

SAMPLE = {
    "bmi": 22.5,
    "age_grp": "adult",
    "lifestyle_risk": "low",
    "city_tier": 1,
    "income_lpa": 10.0,
    "occupation": "private_job",
}


def test_prediction_has_expected_shape():
    out = predict_output(SAMPLE)
    assert set(out) == {"predicted_category", "confidence", "class_probabilities"}
    assert out["predicted_category"] in class_labels
    assert set(out["class_probabilities"]) == set(class_labels)


def test_probabilities_are_consistent():
    out = predict_output(SAMPLE)
    probs = out["class_probabilities"]
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)
    assert out["confidence"] == pytest.approx(probs[out["predicted_category"]], abs=1e-4)


def test_outputs_are_plain_python_types():
    out = predict_output(SAMPLE)
    assert type(out["predicted_category"]) is str
    assert type(out["confidence"]) is float
    assert all(type(p) is float for p in out["class_probabilities"].values())


@pytest.mark.parametrize(
    "occupation",
    ["retired", "freelancer", "student", "government_job", "business_owner", "unemployed", "private_job"],
)
def test_every_supported_occupation_can_be_scored(occupation):
    out = predict_output({**SAMPLE, "occupation": occupation})
    assert out["predicted_category"] in class_labels
