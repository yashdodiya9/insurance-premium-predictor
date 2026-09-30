from typing import Annotated, Literal
from pydantic import BaseModel, Field, computed_field, field_validator
from app.features import compute_bmi, get_age_group, get_city_tier, get_lifestyle_risk

# pydantic model
class UserInput(BaseModel):
    age: Annotated[int, Field(..., gt=0, lt=120, description="Age of user")]
    weight: Annotated[float, Field(..., gt=0, description="Weight of user in kilograms")]
    height: Annotated[float, Field(..., gt=0, lt=2.5, description="Height of user in meters")]
    income_lpa: Annotated[float, Field(..., gt=0, description="Annual salary of user in lpa")]
    smoker: Annotated[bool, Field(..., description="Is user a smoker?")]
    city: Annotated[str, Field(..., description="City of user")]
    occupation: Annotated[
        Literal[
            "retired",
            "freelancer",
            "student",
            "government_job",
            "business_owner",
            "unemployed",
            "private_job",
        ],
        Field(..., description="Occupation of user"),
    ]

    @field_validator("city")
    @classmethod
    def normalize_city(cls, v: str) -> str:
        return v.strip().title()

    # The derived features below come from app/features.py, the same code used in training.
    @computed_field
    @property
    def bmi(self) -> float:
        return compute_bmi(self.weight, self.height)

    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        return get_lifestyle_risk(self.smoker, self.bmi)

    @computed_field
    @property
    def age_grp(self) -> str:
        return get_age_group(self.age)

    @computed_field
    @property
    def city_tier(self) -> int:
        return get_city_tier(self.city)
