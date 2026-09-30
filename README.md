# Insurance Premium Category Predictor

A machine learning API that predicts whether a customer falls into a **Low**, **Medium** or **High** insurance premium category, based on a few personal and lifestyle details. The model is served with **FastAPI**, packaged with **Docker**, and comes with a **Streamlit** web app for trying it out.

## Features

- **REST API** built with FastAPI, with auto-generated interactive docs at `/docs`
- **Input validation** with Pydantic (ranges, allowed occupations, type checks)
- **Derived features computed server-side**: BMI, age group, lifestyle risk and city tier are calculated from the raw inputs, so clients only send simple values
- **One source of truth for features**: `app/features.py` is used by both the API and the training script, so they cannot drift apart
- **Model comparison with cross-validation**: `train.py` compares four models (including a naive baseline) and reports accuracy, precision, recall and F1
- **Typed responses**: predicted category, confidence, and the probability of every class
- **Health endpoint** reporting API status, model version and whether the model is loaded
- **Automated tests** with pytest, a **Dockerfile**, and a **Render** blueprint for deployment
- **Streamlit frontend** that shows the prediction, confidence and a probability chart

## Tech stack

Python 3.12 · FastAPI · Pydantic v2 · scikit-learn · pandas · Streamlit · pytest · Docker · uv

## Project structure

```
.
├── app/
│   ├── main.py                      # FastAPI app and endpoints
│   ├── features.py                  # Feature engineering shared by training and serving
│   ├── config/city_tier.py          # Tier 1 / Tier 2 city lists
│   ├── schema/
│   │   ├── user_input.py            # Request model (uses app/features.py)
│   │   └── prediction_response.py   # Response model
│   └── model/
│       ├── predict.py               # Loads the model and runs predictions
│       ├── model.pkl                # Trained scikit-learn pipeline (created by train.py)
│       └── metrics.json             # Cross-validation and test results (created by train.py)
├── train.py                         # Trains, compares and exports the model
├── frontend/app.py                  # Streamlit UI
├── tests/                           # pytest test suite
├── notebooks/ML_FastAPI.ipynb       # Original exploration notebook
├── data/insurance.csv               # Training data
├── Dockerfile / .dockerignore       # Container image for the API
├── render.yaml                      # Render deployment blueprint
├── pyproject.toml / uv.lock         # Dependencies (uv)
└── requirements*.txt                # pip-compatible dependency lists
```

## Getting started

### 1. Install dependencies

With [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv sync
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the API

Run this from the project root:

```bash
uv run uvicorn app.main:app --reload
# or, without uv:
uvicorn app.main:app --reload
```

The API is now at http://127.0.0.1:8000 and the interactive docs are at http://127.0.0.1:8000/docs.

### 3. Start the Streamlit app

In a second terminal:

```bash
uv run streamlit run frontend/app.py
# or, without uv:
streamlit run frontend/app.py
```

By default it talks to `http://127.0.0.1:8000`. To point it at another API, set the `API_URL` environment variable (or the `API_URL` Streamlit secret).

## Training and evaluation

Train (or retrain) the model with:

```bash
uv run python train.py
# options: --data data/insurance.csv --output app/model/model.pkl --cv-repeats 3 --seed 42
```

The script:

1. Builds features with `app/features.py` (the same code the API uses).
2. Sets aside a stratified 20% test set that is never used for model selection.
3. Compares four models with 5-fold cross-validation repeated 3 times on the remaining data.
4. Picks the model with the best cross-validated macro F1 and evaluates it once on the test set.
5. Refits the chosen model on all the data, saves it to `app/model/model.pkl`, and writes all results to `app/model/metrics.json`.

To train on your own data, pass `--data` with a CSV that has the same columns as `data/insurance.csv`.

### Results on the included dataset

Cross-validation on the training split (mean over 15 folds; precision, recall and F1 are macro-averaged):

| Model                          | Accuracy | Precision | Recall | F1    |
| ------------------------------ | -------- | --------- | ------ | ----- |
| Logistic Regression (selected) | 0.725    | 0.744     | 0.729  | 0.719 |
| Random Forest                  | 0.692    | 0.713     | 0.695  | 0.686 |
| Gradient Boosting              | 0.583    | 0.635     | 0.591  | 0.582 |
| Baseline (most frequent class) | 0.312    | 0.104     | 0.333  | 0.159 |

On the 20-row held-out test set the selected model scored 0.80 accuracy and 0.78 macro F1.

**How to read these numbers:** every real model clearly beats the baseline, but the dataset has only 100 rows, so fold-to-fold scores vary by about 5 to 8 percentage points. The gap between Logistic Regression and Random Forest is smaller than that variation, so treat the ranking of those two as inconclusive. An earlier version of this project reported 85% accuracy from a single 20-row split; cross-validation shows that figure was optimistic.

## Running the tests

```bash
uv run pytest
# or, with pip:
pip install -r requirements-dev.txt
pytest
```

The suite covers the feature-engineering rules (including boundary values), the model's output shape and probabilities, the training pipeline end to end, and the API (success, validation errors, and the generic 500 response).

## Docker

Build and run the API in a container:

```bash
docker build -t insurance-api .
docker run --rm -p 8000:8000 insurance-api
```

Then open http://127.0.0.1:8000/docs. The image installs only the API's runtime dependencies (`requirements-api.txt`), runs as a non-root user, and includes a health check.

## Deployment

### API on Render

1. Push this project to GitHub (make sure `app/model/model.pkl` is committed).
2. In the Render dashboard choose **New > Blueprint**, connect the repository, and apply. Render reads `render.yaml` and builds the Dockerfile. (Alternatively: **New > Web Service** and pick the Docker runtime.)
3. When the deploy finishes, open `https://<your-service>.onrender.com/health` and `/docs`.

On Render's free plan the service spins down after about 15 minutes without traffic, so the first request after a pause can take up to a minute.

### Frontend on Streamlit Community Cloud

1. On [share.streamlit.io](https://share.streamlit.io) create a new app from the repository with `frontend/app.py` as the main file.
2. Under **Advanced settings > Secrets**, add:

   ```toml
   API_URL = "https://<your-service>.onrender.com"
   ```

3. Deploy. The app's dependencies come from `frontend/requirements.txt`.

## API reference

### `GET /health`

```json
{ "status": "OK", "version": "1.0.0", "model_loaded": true }
```

### `POST /predict`

**Request body**

| Field        | Type    | Rules                                                                                         |
| ------------ | ------- | --------------------------------------------------------------------------------------------- |
| `age`        | integer | greater than 0 and less than 120                                                              |
| `weight`     | float   | kilograms, greater than 0                                                                     |
| `height`     | float   | meters, greater than 0 and less than 2.5                                                      |
| `income_lpa` | float   | annual income in lakhs per annum, greater than 0                                              |
| `smoker`     | boolean |                                                                                               |
| `city`       | string  | any city name; matched against the tier lists after trimming and title-casing, unknown cities count as Tier 3 |
| `occupation` | string  | one of `retired`, `freelancer`, `student`, `government_job`, `business_owner`, `unemployed`, `private_job` |

**Example**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "weight": 65,
    "height": 1.7,
    "income_lpa": 10,
    "smoker": false,
    "city": "Mumbai",
    "occupation": "private_job"
  }'
```

**Response** `200 OK`

```json
{
  "predicted_category": "Low",
  "confidence": 0.69,
  "class_probabilities": {
    "High": 0.0,
    "Low": 0.69,
    "Medium": 0.31
  }
}
```

The exact probabilities depend on the model currently saved in `app/model/model.pkl`; the shape of the response is always the same.

**Errors**

| Status | Meaning                                                                          |
| ------ | -------------------------------------------------------------------------------- |
| `422`  | The request body failed validation (the response lists which fields are invalid) |
| `500`  | The prediction failed unexpectedly; details are logged server-side               |

## How it works

The API accepts raw user details and turns them into the features the model was trained on:

| Feature          | Derived from        | Rule                                                                                   |
| ---------------- | ------------------- | -------------------------------------------------------------------------------------- |
| `bmi`            | weight, height      | weight / height²                                                                       |
| `age_grp`        | age                 | `young` (<25), `adult` (<45), `middle-aged` (<60), `senior` (60+)                      |
| `lifestyle_risk` | smoker, bmi         | `high` if smoker and BMI > 30; `medium` if smoker or BMI > 27; otherwise `low`         |
| `city_tier`      | city                | 1 for major metros, 2 for large cities, 3 for everything else (see `app/config/city_tier.py`) |

These features, together with `income_lpa` and `occupation`, go into a scikit-learn `Pipeline` that one-hot encodes the categorical columns (unseen categories are ignored rather than raising an error) and feeds everything to the selected classifier. The rules live in [`app/features.py`](app/features.py).

## Limitations

- The training set is a small sample of 100 rows, so this project demonstrates building, evaluating and serving an ML model; it is not a production-grade pricing model.
- `model.pkl` is a pickle file. Only load pickles you trust, and keep the scikit-learn version the same as in `uv.lock` / `requirements-api.txt` (currently 1.9.1). If you upgrade scikit-learn, retrain with `train.py`.
- The notebook in `notebooks/` is the original exploration and still contains its own copy of the feature code; `train.py` is the reference training path.

## Roadmap

- [x] Compare several models with cross-validation and report precision, recall and F1
- [x] Share feature-engineering code between training and serving
- [x] Add automated tests (`pytest`)
- [x] Add a Dockerfile
- [ ] Train and evaluate on a larger dataset
- [ ] Deploy the API and frontend and add the live links here
- [ ] Run the tests automatically with GitHub Actions

## Author

Yash Dodiya
