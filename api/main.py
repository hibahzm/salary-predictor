"""FastAPI salary prediction service."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schema import PredictionRequest, PredictionResponse
import predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    yield


app = FastAPI(
    title="Salary Prediction API",
    description="Predicts data science salaries using a Random Forest model.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict", response_model=PredictionResponse)
def predict_salary(
    experience_level: str,
    employment_type:  str,
    job_title:        str,
    company_size:     str,
    remote_ratio:     int,
    company_region:   str,
    employee_region:  str,
):
    try:
        req = PredictionRequest(
            experience_level=experience_level,
            employment_type=employment_type,
            job_title=job_title,
            company_size=company_size,
            remote_ratio=remote_ratio,
            company_region=company_region,
            employee_region=employee_region,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    salary = predictor.predict(
        req.experience_level,
        req.employment_type,
        req.job_title,
        req.company_size,
        req.remote_ratio,
        req.company_region,
        req.employee_region,
    )

    return PredictionResponse(
        predicted_salary_usd=round(salary, 2),
        experience_level=req.experience_level,
        employment_type=req.employment_type,
        job_title=req.job_title,
        company_size=req.company_size,
        remote_ratio=req.remote_ratio,
        company_region=req.company_region,
        employee_region=req.employee_region,
    )