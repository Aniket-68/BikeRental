from pydantic import BaseModel, ConfigDict, Field, model_validator


class PredictionRequest(BaseModel):
    # Reject unexpected fields, incorrect types, NaN, and infinity.
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )

    # Match the feature names used during training.
    hr: int = Field(ge=0, le=23)
    mnth: int = Field(ge=1, le=12)
    weekday: int = Field(ge=0, le=6)  # UCI: Sunday=0, Saturday=6.
    workingday: int = Field(ge=0, le=1)
    holiday: int = Field(ge=0, le=1)
    weathersit: int = Field(ge=1, le=4)

    # These are UCI-normalized values, not Celsius or percentages.
    temp: float = Field(ge=0, le=1)
    hum: float = Field(ge=0, le=1)
    windspeed: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def check_calendar(self):
        # A working day must be a weekday that is not a holiday.
        expected = int(self.weekday not in (0, 6) and self.holiday == 0)

        if self.workingday != expected:
            raise ValueError("workingday conflicts with weekday or holiday")

        return self


class PredictionResponse(BaseModel):
    request_id: str
    predicted_rentals: float = Field(ge=0, allow_inf_nan=False)
    model_uri: str