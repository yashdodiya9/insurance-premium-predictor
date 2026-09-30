import logging

from fastapi import FastAPI, HTTPException
from app.model.predict import MODEL_VERSION, model, predict_output
from app.schema.prediction_response import PredictionResponse
from app.schema.user_input import UserInput

logger = logging.getLogger("insurance_api")

app = FastAPI(
    title="Insurance Premium Category API",
    description="Predicts whether a customer falls into a Low, Medium or High "
    "insurance premium category.",
    version=MODEL_VERSION,
)


@app.get("/", tags=["General"])
def home():
    return {"message": "Insurance premium prediction API"}


@app.get("/health", tags=["General"])
def health_check():
    """Machine-readable endpoint to confirm the API is up and the model is loaded."""
    return {"status": "OK", "version": MODEL_VERSION, "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_premium(data: UserInput):
    user_input = {
        "bmi": data.bmi,
        "age_grp": data.age_grp,
        "lifestyle_risk": data.lifestyle_risk,
        "city_tier": data.city_tier,
        "income_lpa": data.income_lpa,
        "occupation": data.occupation,
    }

    try:
        # Returning the dict lets FastAPI validate it against PredictionResponse.
        return predict_output(user_input)
    except Exception:
        # Log the details server-side; don't leak internals to the client.
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again later.")
