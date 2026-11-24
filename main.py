"""FastAPI service for the term-deposit subscription model.

Run:   uv run fastapi dev main.py
Docs:  http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = "model.joblib"
ml = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the pipeline once at startup, not on every request.
    ml["model"] = joblib.load(MODEL_PATH)
    yield
    ml.clear()


app = FastAPI(
    title="Term Deposit Subscription Predictor",
    description=(
        "Predicts whether a bank client will subscribe to a term deposit "
        "if called in the next marketing campaign (UCI Bank Marketing data)."
    ),
    lifespan=lifespan,
)


class Client(BaseModel):
    """One client, described with information the bank has BEFORE the call."""

    age: int = Field(..., ge=18, le=100, description="Age in years")
    job: Literal[
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician",
        "unemployed", "unknown",
    ]
    marital: Literal["divorced", "married", "single"]
    education: Literal["primary", "secondary", "tertiary", "unknown"]
    default: Literal["no", "yes"] = Field(..., description="Has credit in default?")
    balance: int = Field(..., description="Average yearly account balance (euros)")
    housing: Literal["no", "yes"] = Field(..., description="Has a housing loan?")
    loan: Literal["no", "yes"] = Field(..., description="Has a personal loan?")
    contact: Literal["cellular", "telephone", "unknown"]
    day: int = Field(..., ge=1, le=31, description="Day of month of the planned call")
    month: Literal[
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    ]
    campaign: int = Field(..., ge=1, description="Number of contacts in this campaign, including this one")
    pdays: int = Field(..., ge=-1, description="Days since last contact in a previous campaign (-1 = never)")
    previous: int = Field(..., ge=0, description="Contacts before this campaign")
    poutcome: Literal["failure", "other", "success", "unknown"] = Field(
        ..., description="Outcome of the previous campaign"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 35,
                    "job": "management",
                    "marital": "married",
                    "education": "tertiary",
                    "default": "no",
                    "balance": 1500,
                    "housing": "yes",
                    "loan": "no",
                    "contact": "cellular",
                    "day": 15,
                    "month": "may",
                    "campaign": 2,
                    "pdays": -1,
                    "previous": 0,
                    "poutcome": "unknown",
                }
            ]
        }
    }


class Prediction(BaseModel):
    subscribe: bool
    label: Literal["yes", "no"]
    probability_yes: float = Field(..., description="Model's probability the client subscribes")


@app.get("/")
def root():
    return {"status": "ok", "docs": "/docs"}


@app.post("/predict", response_model=Prediction)
def predict(client: Client) -> Prediction:
    row = pd.DataFrame([client.model_dump()])  # one-row DataFrame
    pred = int(ml["model"].predict(row)[0])
    proba = float(ml["model"].predict_proba(row)[0, 1])
    return Prediction(
        subscribe=bool(pred), label="yes" if pred else "no", probability_yes=round(proba, 4)
    )
