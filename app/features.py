"""Feature engineering shared by training (train.py) and serving (app/schema/user_input.py).

This module is the single source of truth for how raw user details are turned into
model features. Keeping it in one place prevents training/serving skew: if a rule
changes here, both the trained model and the API pick it up.
"""
import pandas as pd

from app.config.city_tier import tier_1_cities, tier_2_cities

TARGET = "insurance_premium_category"

# Columns the model is trained on (and that predict.py must be given).
CATEGORICAL_FEATURES = ["occupation", "age_grp", "lifestyle_risk", "city_tier"]
NUMERIC_FEATURES = ["income_lpa", "bmi"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Raw columns needed to derive the features above.
RAW_COLUMNS = ["age", "weight", "height", "income_lpa", "smoker", "city", "occupation"]


def compute_bmi(weight: float, height: float) -> float:
    """BMI from weight in kilograms and height in meters."""
    return weight / (height ** 2)


def get_age_group(age: int) -> str:
    if age < 25:
        return "young"
    elif age < 45:
        return "adult"
    elif age < 60:
        return "middle-aged"
    return "senior"


def get_lifestyle_risk(smoker: bool, bmi: float) -> str:
    if smoker and bmi > 30:
        return "high"
    elif smoker or bmi > 27:
        return "medium"
    return "low"


def get_city_tier(city: str) -> int:
    """1 for major metros, 2 for large cities, 3 for everything else."""
    city = city.strip().title()
    if city in tier_1_cities:
        return 1
    elif city in tier_2_cities:
        return 2
    return 3


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a raw dataframe with the derived model features added."""
    out = df.copy()
    out["bmi"] = [compute_bmi(w, h) for w, h in zip(out["weight"], out["height"])]
    out["age_grp"] = out["age"].map(get_age_group)
    out["lifestyle_risk"] = [get_lifestyle_risk(s, b) for s, b in zip(out["smoker"], out["bmi"])]
    out["city_tier"] = out["city"].map(get_city_tier)
    return out
