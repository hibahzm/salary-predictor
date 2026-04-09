"""
Input / output schemas for /predict.

Includes company_region and employee_region —
both mapped from raw country codes in train.py.
All fields are strict Literals — no free text, no encoder crashes.
"""
from typing import Literal
from pydantic import BaseModel

JobTitle = Literal[
    "Data Scientist",
    "Data Engineer",
    "ML Engineer",
    "Data Analyst",
    "Research Scientist",
    "Other",
]

Region = Literal[
    "North America",
    "Europe",
    "Asia",
    "South America",
    "Africa",
    "Oceania",
    "Other",
]


class PredictionRequest(BaseModel):
    experience_level: Literal["EN", "MI", "SE", "EX"]
    employment_type:  Literal["FT", "PT", "CT", "FL"]
    job_title:        JobTitle
    company_size:     Literal["S", "M", "L"]
    remote_ratio:     Literal[0, 50, 100]
    company_region:   Region
    employee_region:  Region


class PredictionResponse(BaseModel):
    predicted_salary_usd: float
    experience_level:     str
    employment_type:      str
    job_title:            str
    company_size:         str
    remote_ratio:         int
    company_region:       str
    employee_region:      str