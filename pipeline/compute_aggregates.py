"""
Step 1 of the pipeline.

Reads ds_salaries.csv → computes aggregated salary stats
→ stores in Supabase salary_aggregates table.

What we compute:
  - avg/min/max salary by experience_level
  - avg/min/max salary by job_title (grouped)
  - avg/min/max salary by company_region
  - avg/min/max salary by company_size
  - avg/min/max salary by remote_ratio
  - overall dataset stats (for context)

These aggregates are what we feed to the LLM — not raw predictions.
The LLM gets CONTEXT, not just a number.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.supabase_client import get_client

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ds_salaries.csv")

# Same grouping as train.py
JOB_GROUPS: dict[str, list[str]] = {
    "Data Scientist": [
        "Data Scientist", "Applied Data Scientist", "Staff Data Scientist",
        "Lead Data Scientist", "Principal Data Scientist", "Data Science Consultant",
        "Data Science Engineer", "Data Science Manager", "Director of Data Science",
        "Head of Data Science",
    ],
    "Data Engineer": [
        "Data Engineer", "Big Data Engineer", "Cloud Data Engineer",
        "Lead Data Engineer", "Principal Data Engineer", "Data Engineering Manager",
        "Director of Data Engineering", "ETL Developer", "Analytics Engineer",
        "Data Architect", "Big Data Architect",
    ],
    "ML Engineer": [
        "Machine Learning Engineer", "ML Engineer", "Lead Machine Learning Engineer",
        "Machine Learning Developer", "Machine Learning Infrastructure Engineer",
        "Machine Learning Manager", "Head of Machine Learning",
        "Applied Machine Learning Scientist",
    ],
    "Data Analyst": [
        "Data Analyst", "Business Data Analyst", "Lead Data Analyst",
        "BI Data Analyst", "Marketing Data Analyst", "Financial Data Analyst",
        "Finance Data Analyst", "Product Data Analyst", "Principal Data Analyst",
        "Data Analytics Engineer", "Data Analytics Manager", "Data Analytics Lead",
        "Data Specialist",
    ],
    "Research Scientist": [
        "Research Scientist", "Machine Learning Scientist", "AI Scientist",
        "NLP Engineer", "Computer Vision Engineer", "3D Computer Vision Researcher",
        "Computer Vision Software Engineer",
    ],
}

REGION_MAP: dict[str, str] = {
    "US": "North America", "CA": "North America", "MX": "North America", "PR": "North America",
    "DE": "Europe", "GB": "Europe", "FR": "Europe", "ES": "Europe", "PT": "Europe",
    "IT": "Europe", "NL": "Europe", "BE": "Europe", "CH": "Europe", "AT": "Europe",
    "PL": "Europe", "RO": "Europe", "DK": "Europe", "CZ": "Europe", "IE": "Europe",
    "GR": "Europe", "HU": "Europe", "HR": "Europe", "LU": "Europe", "SI": "Europe",
    "EE": "Europe", "MT": "Europe", "UA": "Europe", "BG": "Europe", "RS": "Europe",
    "JE": "Europe",
    "IN": "Asia", "JP": "Asia", "CN": "Asia", "PK": "Asia", "SG": "Asia",
    "AE": "Asia", "IL": "Asia", "IR": "Asia", "IQ": "Asia", "TR": "Asia",
    "VN": "Asia", "MY": "Asia", "PH": "Asia", "HK": "Asia",
    "BR": "South America", "CL": "South America", "CO": "South America",
    "AR": "South America", "BO": "South America",
    "NG": "Africa", "KE": "Africa", "DZ": "Africa", "TN": "Africa",
    "AU": "Oceania", "NZ": "Oceania",
}


def group_job_title(title: str) -> str:
    for group, titles in JOB_GROUPS.items():
        if title in titles:
            return group
    return "Other"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["job_title"]       = df["job_title"].apply(group_job_title)
    df["company_region"]  = df["company_location"].map(REGION_MAP).fillna("Other")
    df = df.drop_duplicates()
    df = df.dropna(subset=["salary_in_usd"])
    return df


def compute_stats(df: pd.DataFrame, group_col: str, category: str) -> list[dict]:
    """Compute avg/min/max/count for a given grouping column."""
    grouped = df.groupby(group_col)["salary_in_usd"].agg(["mean", "min", "max", "count"])
    records = []
    for key, row in grouped.iterrows():
        records.append({
            "category":   category,
            "key":        str(key),
            "avg_salary": round(float(row["mean"]), 2),
            "min_salary": round(float(row["min"]),  2),
            "max_salary": round(float(row["max"]),  2),
            "count":      int(row["count"]),
        })
    return records


def run() -> None:
    print("=" * 55)
    print("  Step 1 — Computing Aggregates")
    print("=" * 55)

    df = load_dataset()
    print(f"Dataset loaded: {len(df)} rows after cleaning")

    all_records = []

    # By experience level
    all_records += compute_stats(df, "experience_level", "experience")

    # By job title
    all_records += compute_stats(df, "job_title", "job_title")

    # By company region
    all_records += compute_stats(df, "company_region", "region")

    # By company size
    all_records += compute_stats(df, "company_size", "company_size")

    # By remote ratio
    all_records += compute_stats(df, "remote_ratio", "remote")

    # By employment type
    all_records += compute_stats(df, "employment_type", "employment_type")

    # Overall stats (category="overall", key="all")
    all_records.append({
        "category":   "overall",
        "key":        "all",
        "avg_salary": round(float(df["salary_in_usd"].mean()),   2),
        "min_salary": round(float(df["salary_in_usd"].min()),    2),
        "max_salary": round(float(df["salary_in_usd"].max()),    2),
        "count":      len(df),
    })

    # Clear old aggregates and insert fresh
    client = get_client()
    client.table("salary_aggregates").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()

    client.table("salary_aggregates").insert(all_records).execute()

    print(f"Stored {len(all_records)} aggregate records in Supabase")

    # Print a summary
    print("\nSample aggregates:")
    for r in all_records[:5]:
        print(f"  {r['category']:15} {r['key']:20} avg=${r['avg_salary']:>10,.0f}")

    print("\n✓ Aggregates complete")


if __name__ == "__main__":
    run()