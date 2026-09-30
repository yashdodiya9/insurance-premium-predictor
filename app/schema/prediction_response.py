from typing import Dict
from pydantic import BaseModel, ConfigDict, Field

class PredictionResponse(BaseModel):
    """Shape of the JSON returned by POST /predict."""

    predicted_category: str = Field(
        ..., description="Predicted insurance premium category (Low, Medium or High)"
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Model's probability for the predicted category"
    )
    class_probabilities: Dict[str, float] = Field(
        ..., description="Probability for every possible category"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "predicted_category": "High",
                "confidence": 0.81,
                "class_probabilities": {"High": 0.81, "Low": 0.04, "Medium": 0.15},
            }
        }
    )
