import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException
from app.db import get_stats, init_db, log_prediction
from app.model.predict import MODEL_VERSION, model, predict_output
from app.schema.prediction_response import PredictionResponse
from app.schema.user_input import UserInput

logger = logging.getLogger("insurance_api")

@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()  # creates the prediction_logs table if DATABASE_URL is set
    yield


app = FastAPI(
    lifespan=lifespan,
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
def predict_premium(data: UserInput, background_tasks: BackgroundTasks):
    user_input = {
        "bmi": data.bmi,
        "age_grp": data.age_grp,
        "lifestyle_risk": data.lifestyle_risk,
        "city_tier": data.city_tier,
        "income_lpa": data.income_lpa,
        "occupation": data.occupation,
    }

    try:
        result = predict_output(user_input)
    except Exception:
        # Log the details server-side; don't leak internals to the client.
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again later.")

    # Store the request and result in PostgreSQL after the response is sent.
    background_tasks.add_task(log_prediction, {**user_input, "age": data.age}, result, MODEL_VERSION)
    return result


@app.get("/stats", tags=["Analytics"])
def stats():
    """Summary of logged predictions, aggregated in SQL (needs DATABASE_URL)."""
    try:
        summary = get_stats()
    except Exception:
        logger.exception("Stats query failed")
        raise HTTPException(status_code=503, detail="Database unavailable.")
    if summary is None:
        raise HTTPException(status_code=503, detail="Database not configured.")
    return summary
