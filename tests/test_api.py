import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import create_app


FEATURE_COLUMNS = [
    "hr", "mnth", "weekday", "workingday", "holiday",
    "weathersit", "temp", "hum", "windspeed",
]
MODEL_URI = "models:/test-candidate"


class StubModel:
    # A predictable substitute for the real forest.
    # These tests check API behavior, not prediction accuracy.
    def __init__(self):
        self.value = 123.5
        self.received_frames = []

    def predict(self, frame):
        self.received_frames.append(frame.copy())
        return np.full(len(frame), self.value)


@pytest.fixture
def payload():
    return {
        "hr": 17,
        "mnth": 7,
        "weekday": 2,
        "workingday": 1,
        "holiday": 0,
        "weathersit": 1,
        "temp": 0.7,
        "hum": 0.65,
        "windspeed": 0.15,
    }


@pytest.fixture
def candidate_file(tmp_path):
    # pytest creates an isolated temporary directory for each test.
    path = tmp_path / "candidate.json"
    path.write_text(
        json.dumps({
            "validation_gate_passed": True,
            "model_uri": MODEL_URI,
            "feature_columns": FEATURE_COLUMNS,
        }),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def api_case(candidate_file):
    model = StubModel()
    load_calls = []

    def loader(model_uri):
        # Record loading without accessing MLflow.
        load_calls.append(model_uri)
        return model

    app = create_app(
        candidate_path=candidate_file,
        model_loader=loader,
    )

    # Entering the context runs startup; exiting runs shutdown.
    with TestClient(app) as client:
        yield client, model, load_calls


def test_health_and_readiness(api_case):
    client, _, _ = api_case

    assert client.get("/health").json() == {"status": "ok"}

    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["model_uri"] == MODEL_URI


def test_prediction_preserves_inputs_and_model_identity(api_case, payload):
    client, model, _ = api_case

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()
    assert result["predicted_rentals"] == pytest.approx(123.5)
    assert result["model_uri"] == MODEL_URI
    assert isinstance(result["request_id"], str)
    assert result["request_id"]

    # Verify the DataFrame actually supplied to the model.
    frame = model.received_frames[0]
    assert frame.columns.tolist() == FEATURE_COLUMNS
    assert frame.iloc[0].to_dict() == payload


def test_model_is_loaded_once_for_multiple_requests(api_case, payload):
    client, model, load_calls = api_case

    first = client.post("/predict", json=payload)
    second = client.post("/predict", json=payload)

    assert first.status_code == second.status_code == 200
    assert load_calls == [MODEL_URI]
    assert len(model.received_frames) == 2
    assert first.json()["request_id"] != second.json()["request_id"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("hr", 25),           # Outside the permitted range.
        ("hr", "17"),         # String instead of a strict integer.
        ("temp", 1.2),        # Not a valid normalized temperature.
        ("hum", 101.0),       # Percentage supplied instead of normalized value.
        ("workingday", 0),    # Conflicts with the weekday/holiday inputs.
        ("weathersit", 5),    # Unknown weather category.
    ],
)
def test_invalid_inputs_do_not_reach_model(api_case, payload, field, value):
    client, model, _ = api_case
    payload[field] = value

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert model.received_frames == []


def test_missing_input_is_rejected(api_case, payload):
    client, model, _ = api_case
    payload.pop("temp")

    assert client.post("/predict", json=payload).status_code == 422
    assert model.received_frames == []


def test_extra_input_is_rejected(api_case, payload):
    client, model, _ = api_case
    payload["cnt"] = 200

    assert client.post("/predict", json=payload).status_code == 422
    assert model.received_frames == []


@pytest.mark.parametrize("invalid_prediction", [np.nan, np.inf, -1.0])
def test_invalid_model_outputs_are_rejected(api_case, payload, invalid_prediction):
    client, model, _ = api_case
    model.value = invalid_prediction

    response = client.post("/predict", json=payload)

    assert response.status_code == 500
    assert response.json()["detail"] == "Model returned an invalid prediction"


def test_failed_quality_gate_prevents_startup(candidate_file):
    candidate = json.loads(candidate_file.read_text(encoding="utf-8"))
    candidate["validation_gate_passed"] = False
    candidate_file.write_text(json.dumps(candidate), encoding="utf-8")

    app = create_app(
        candidate_path=candidate_file,
        model_loader=lambda uri: StubModel(),
    )

    with pytest.raises(RuntimeError, match="validation gate"):
        with TestClient(app):
            pass


def test_feature_mismatch_prevents_startup(candidate_file):
    candidate = json.loads(candidate_file.read_text(encoding="utf-8"))
    candidate["feature_columns"].remove("temp")
    candidate_file.write_text(json.dumps(candidate), encoding="utf-8")

    app = create_app(
        candidate_path=candidate_file,
        model_loader=lambda uri: StubModel(),
    )

    with pytest.raises(RuntimeError, match="API fields"):
        with TestClient(app):
            pass