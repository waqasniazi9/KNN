import os
from contextlib import asynccontextmanager
from typing import Literal
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from train import train, FEATURES

@asynccontextmanager
async def lifespan(app: FastAPI):
    default,tuned,report,*_=train()
    app.state.models={"default":default,"tuned":tuned}
    app.state.report=report
    yield

app=FastAPI(title="LoanLens Educational KNN API",lifespan=lifespan)
origins=os.getenv("ALLOWED_ORIGINS","http://localhost:3000").split(",")
app.add_middleware(CORSMiddleware,allow_origins=[x.strip() for x in origins],allow_methods=["GET","POST"],allow_headers=["Content-Type"])
class Applicant(BaseModel):
    Gender: Literal["Male","Female"] | None = None
    Married: Literal["Yes","No"] | None = None
    Dependents: Literal["0","1","2","3+"] | None = None
    Education: Literal["Graduate","Not Graduate"] | None = None
    Self_Employed: Literal["Yes","No"] | None = None
    ApplicantIncome: float | None = Field(default=None,ge=0,allow_inf_nan=False)
    CoapplicantIncome: float | None = Field(default=None,ge=0,allow_inf_nan=False)
    LoanAmount: float | None = Field(default=None,gt=0,allow_inf_nan=False)
    Loan_Amount_Term: float | None = Field(default=None,gt=0,allow_inf_nan=False)
    Credit_History: Literal[0,1] | None = None
    Property_Area: Literal["Urban","Semiurban","Rural"] | None = None
    model: Literal["default","tuned"] = "tuned"
@app.get("/health")
def health(): return {"status":"ok"}
@app.get("/metrics")
def metrics(): return app.state.report
@app.post("/predict")
def predict(request: Applicant):
    values=request.model_dump(exclude={"model"})
    row=pd.DataFrame([{k:(np.nan if v is None else v) for k,v in values.items()}])[FEATURES]
    estimator=app.state.models[request.model]
    outcome=int(estimator.predict(row)[0])
    fraction=float(estimator.predict_proba(row)[0,list(estimator.classes_).index(1)])
    return {"label":"Approved" if outcome else "Rejected","encoded_label":outcome,"model":request.model,"k":estimator.named_steps["knn"].n_neighbors,"approval_neighbor_fraction":fraction,"notice":"Educational model output, not an actual lending decision. Neighbor fractions are not calibrated probabilities."}
