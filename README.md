# Salary Prediction & Insights Platform

An end-to-end **machine learning + LLM-powered analytics system** that predicts salaries for data science professionals and generates intelligent, narrative-driven career insights.

---

## Overview

This project combines:

* Machine Learning model (Random Forest)
* LLM-powered insights (GPT-based system)
* Interactive Streamlit dashboard
* FastAPI backend for real-time predictions
* Supabase for data storage and analytics
* Precomputed insights for fast, scalable storytelling

Instead of generating insights on-demand, the system builds a **library of intelligent insights once**, then dynamically assembles them per user.

---

## Key Features

### Salary Prediction

* Predicts annual salary based on:

  * Experience level
  * Job title
  * Company size
  * Remote ratio
  * Region (company + employee)
* Powered by a trained **Random Forest Regressor**

---

### AI-Powered Insights (GPT-based)

* Uses **OpenAI GPT models** for generation
* Generates structured JSON insights
* Covers 10 intelligent scenarios:

  * Junior career growth paths
  * Mid-level optimization strategies
  * Senior global market comparison
  * Executive positioning
  * Job title salary ranking
  * Regional salary gaps
  * Remote work impact
  * Company size strategy
  * Overall market overview
  * Top earner humor insight

---

### Streamlit Dashboard

* Interactive UI for:

  * Salary prediction
  * Personalized insights
  * Global salary analytics
* Dynamic charts powered by Plotly
* Insight cards assembled based on user profile

---

### Data Layer (Supabase)

Two main tables:

* `salary_aggregates`

  * Aggregated statistics from dataset
* `insights`

  * Precomputed GPT-generated insights (JSON stored once)

---

## Architecture

```
LOCAL MACHINE
────────────────────────────────────────────
CSV Dataset
   ↓
compute_aggregates.py
   → stores stats in Supabase

generate_insights.py
   → calls GPT once per scenario
   → stores insights in Supabase

FastAPI
   → serves /predict endpoint

Streamlit Dashboard
   → calls API + reads Supabase
   → builds personalized story


CLOUD
────────────────────────────────────────────
Supabase
   → aggregated salary data
   → precomputed GPT insights

Streamlit Cloud
   → dashboard hosting

Vercel 
   → API deployment
```

---

## Project Structure

```
salary-predictor/
├── data/
│   └── ds_salaries.csv
│
├── model/
│   ├── train.py
│   └── saved/
│       ├── model.pkl
│       └── encoders.pkl
│
├── api/
│   ├── main.py
├   ├── main.py
│   ├── predictor.py
│   ├── schema.py
│   ├── config.py
│      
│
├── pipeline/
│   ├── compute_aggregates.py
│   ├── generate_insights.py
│   ├── supabase_client.py
│   ├── gpt_client
│   └── run_pipeline.py
│
├── dashboard/
│   └── app.py
│
├── requirements.txt
├── vercel.json
├── .env.example
└── README.md
```

---

## Setup Instructions

### 1️⃣ Clone Project

```bash
git clone https://github.com/hibahzm/salary-predictor.git
cd salary-predictor
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Mac / Linux
venv\Scripts\activate      # Windows
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Environment Variables

Create `.env` file:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

API_URL=http://localhost:8000

# GPT (OpenAI) 
OPENAI_API_KEY=your_openai_api_key
```

---

## Pipeline Execution

Run full pipeline:

```bash
python pipeline/run_pipeline.py
```

This will:

1. Compute aggregated statistics from dataset
2. Generate insights using **GPT**
3. Store everything in Supabase

---

## Run Locally

### Start FastAPI

```bash
uvicorn api.main:app --reload --port 8000
```

Test:

```bash
curl "http://localhost:8000/predict?experience_level=SE&employment_type=FT&job_title=Data Scientist&company_size=M&remote_ratio=100&company_region=North America&employee_region=Europe"
```

---

### Start Dashboard

```bash
streamlit run dashboard/app.py
```

Open:

```
http://localhost:8501
```

---

## Deployment

### Vercel (FastAPI)

```bash
vercel --prod
```

### Streamlit Cloud

Add secrets:

```toml
SUPABASE_URL="..."
SUPABASE_KEY="..."
API_URL="https://your-api.vercel.app"
```

---

## Design Philosophy

* No real-time LLM calls in dashboard
* Precompute insights once
* Fast UI response
* Modular pipeline architecture
* GPT used only for structured intelligence generation
* Supabase acts as the single source of truth

---

## Example Insights

* Entry-level engineers see the biggest jump when moving to ML roles
* North America pays ~2x more than most regions
* Remote jobs slightly increase salary in senior roles
* Small companies can outperform large ones at senior level in niche roles

---

## Tech Stack

* Python
* FastAPI
* Streamlit
* Scikit-learn
* Pandas / NumPy
* Plotly
* Supabase
* OpenAI GPT

---

## Notes

* Model + encoders saved in `/model/saved`
* Insights generated once and reused
* System optimized for scalability

---

## Status

✔ MVP Complete
✔ Pipeline Working
✔ Dashboard Integrated
✔ GPT Insight Engine Active
✔ Ready for Deployment

---

## Author

Built as a full-stack AI/ML portfolio project combining:

* Machine Learning
* LLM engineering (GPT)
* Data engineering pipeline design
* Full-stack dashboard development
