import pandas as pd
import pytest
from app.features import (
    FEATURE_COLUMNS,
    add_features,
    compute_bmi,
    get_age_group,
    get_city_tier,
    get_lifestyle_risk,
)


def test_compute_bmi():
    assert compute_bmi(70, 1.75) == pytest.approx(22.857, abs=1e-3)


@pytest.mark.parametrize(
    "age, expected",
    [
        (1, "young"),
        (24, "young"),
        (25, "adult"),
        (44, "adult"),
        (45, "middle-aged"),
        (59, "middle-aged"),
        (60, "senior"),
        (119, "senior"),
    ],
)
def test_age_group_boundaries(age, expected):
    assert get_age_group(age) == expected


@pytest.mark.parametrize(
    "smoker, bmi, expected",
    [
        (True, 31, "high"),
        (True, 30, "medium"),  # BMI must be strictly above 30 to be "high"
        (True, 22, "medium"),
        (False, 28, "medium"),
        (False, 27, "low"),  # BMI must be strictly above 27 to count
        (False, 22, "low"),
    ],
)
def test_lifestyle_risk(smoker, bmi, expected):
    assert get_lifestyle_risk(smoker, bmi) == expected


@pytest.mark.parametrize(
    "city, expected",
    [
        ("Mumbai", 1),
        ("mumbai", 1),
        ("  DELHI ", 1),
        ("Jaipur", 2),
        ("surat", 2),
        ("Pithampur", 3),
    ],
)
def test_city_tier_is_case_and_whitespace_insensitive(city, expected):
    assert get_city_tier(city) == expected


def test_add_features_builds_every_model_column_and_leaves_input_untouched():
    raw = pd.DataFrame(
        [
            {"age": 21, "weight": 60, "height": 1.66, "income_lpa": 10, "smoker": True,
             "city": "surat", "occupation": "retired"},
            {"age": 50, "weight": 100, "height": 1.7, "income_lpa": 5, "smoker": True,
             "city": "Mumbai", "occupation": "private_job"},
        ]
    )
    original_columns = list(raw.columns)

    out = add_features(raw)

    assert list(raw.columns) == original_columns  # input not mutated
    assert set(FEATURE_COLUMNS) <= set(out.columns)
    assert out.loc[0, "age_grp"] == "young"
    assert out.loc[0, "lifestyle_risk"] == "medium"
    assert out.loc[0, "city_tier"] == 2
    assert out.loc[1, "lifestyle_risk"] == "high"  # smoker with BMI ~34.6
    assert out.loc[1, "city_tier"] == 1
