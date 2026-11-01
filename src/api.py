import json
import math
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from schemas import PredictionRequest, PredictionResponse


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_pipeline(model_uri):
    # Load the exact artifact referenced by the training output.
    pipeline = mlflow.sklearn.load_model(model_uri)

    # Avoid every concurrent request launching its own pool of forest threads.
    pipeline.set_params(regressor__n_jobs=1)

    return pipeline


def create_app(candidate_path=None, model_loader=None):
    # Tests can inject a temporary manifest and a lightweight model loader.
    manifest_path = (
        Path(candidate_path)
        if candidate_path is not None
        else PROJECT_ROOT / "models" / "candidate.json"
    )

    @asynccontextmanager
    async def lifespan(app):
        # Read the manifest once at startup.
        # A later change to candidate.json requires restarting this service.
        candidate = json.loads(manifest_path.read_text(encoding="utf-8"))

        if candidate.get("validation_gate_passed") is not True:
            raise RuntimeError("Candidate has not passed its validation gate")

        feature_columns = candidate["feature_columns"]
        expected_columns = set(PredictionRequest.model_fields)

        # Catch a mismatch between the API contract and training metadata.
        if (
            set(feature_columns) != expected_columns
            or len(feature_columns) != len(expected_columns)
        ):
            raise RuntimeError("API fields do not match the model features")

        database_path = (PROJECT_ROOT / "mlflow.db").as_posix()
        mlflow.set_tracking_uri(f"sqlite:///{database_path}")

        loader = model_loader or load_pipeline

        # Model loading performs blocking work.
        # Await its completion in a worker thread before serving requests.
        app.state.model = await run_in_threadpool(
            loader,
            candidate["model_uri"],
        )
        app.state.candidate = candidate

        # FastAPI accepts requests after startup reaches this point.
        yield

        app.state.model = None

    app = FastAPI(
        title="BikeDemand API",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health():
        # Liveness: the application can respond.
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request):
        # Readiness: a model is available for prediction.
        if getattr(request.app.state, "model", None) is None:
            raise HTTPException(status_code=503, detail="Model unavailable")

        return {
            "status": "ready",
            "model_uri": request.app.state.candidate["model_uri"],
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest, request: Request):
        candidate = request.app.state.candidate

        # Preserve the feature order recorded during training.
        features = pd.DataFrame(
            [payload.model_dump()],
            columns=candidate["feature_columns"],
        )

        # Reuse the loaded pipeline's fitted encoder and forest.
        # Serving must never call fit() or fit_transform().
        values = request.app.state.model.predict(features)

        if len(values) != 1:
            raise HTTPException(
                status_code=500,
                detail="Model returned an invalid prediction count",
            )

        prediction = float(values[0])

        if not math.isfinite(prediction) or prediction < 0:
            raise HTTPException(
                status_code=500,
                detail="Model returned an invalid prediction",
            )

        return PredictionResponse(
            request_id=str(uuid4()),
            predicted_rentals=prediction,
            model_uri=candidate["model_uri"],
        )

    return app


# Creating the application does not load the model.x
# Loading happens when the server starts its lifespan.
app = create_app()