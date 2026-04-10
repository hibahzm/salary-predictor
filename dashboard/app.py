"""
Streamlit dashboard 

Selection logic:
  Base scenarios  → always shown (2, based on experience)
  Region trigger  → if user is in low-paying region → show region comparison
  Remote trigger  → if on-site → show remote work impact
  Size trigger    → if small company → show company size strategy
  Title trigger   → if Data Analyst → show career switch
  Employment      → if contract/freelance → show employment type insight
  Top earner      → if Executive + Large + North America → show roast
  Max 6 tabs shown at once
"""
import os
import json
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "page" not in st.session_state:
    st.session_state["page"] = "predict"

page = st.session_state["page"]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  [data-testid="collapsedControl"] {{ display: none; }}
  section[data-testid="stSidebar"]  {{ display: none; }}

  div[data-testid="stHorizontalBlock"] button {{
    border-radius: 24px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.45rem 1.6rem !important;
    border: 2px solid #1565C0 !important;
    transition: all 0.2s ease !important;
  }}
  div[data-testid="stHorizontalBlock"] button[kind="primary"] {{
    background-color: #1565C0 !important;
    color: white !important;
  }}
  div[data-testid="stHorizontalBlock"] button[kind="secondary"] {{
    background-color: white !important;
    color: #1565C0 !important;
  }}
  div[data-testid="stHorizontalBlock"] button[kind="secondary"]:hover {{
    background-color: #E3F2FD !important;
  }}

  .headline-stat {{
    font-size: 1.15rem;
    font-weight: 700;
    color: #0D47A1;
    background: #E3F2FD;
    border-left: 5px solid #1565C0;
    padding: 0.7rem 1.1rem;
    border-radius: 0 10px 10px 0;
    margin-bottom: 0.9rem;
    line-height: 1.5;
  }}
  .fun-fact {{
    font-size: 0.88rem;
    color: #4A3000;
    background: #FFF8E1;
    border-left: 5px solid #F9A825;
    padding: 0.55rem 1rem;
    border-radius: 0 10px 10px 0;
    margin-top: 0.7rem;
    line-height: 1.5;
  }}
  .insight-narrative {{
    font-size: 1rem;
    line-height: 1.8;
    color: #eaeaea;;
    margin: 0 0 0.5rem 0;
  }}
  .salary-card {{
    background: linear-gradient(135deg, #0D47A1 0%, #1976D2 60%, #42A5F5 100%);
    color: white;
    padding: 2.2rem;
    border-radius: 18px;
    text-align: center;
    margin: 1rem 0 1.5rem 0;
    box-shadow: 0 4px 20px rgba(21,101,192,0.3);
  }}
  .trigger-badge {{
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 600;
    background: #E8F5E9;
    color: #2E7D32;
    border: 1px solid #A5D6A7;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    margin: 0.2rem 0.2rem 0.6rem 0;
  }}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
API_URL  = os.getenv("API_URL", "http://localhost:8000")
EXP_MAP  = {"EN": "Entry Level", "MI": "Mid Level", "SE": "Senior",   "EX": "Executive"}
EMP_MAP  = {"FT": "Full-time",   "PT": "Part-time", "CT": "Contract", "FL": "Freelance"}
SIZE_MAP = {"S": "Small",        "M": "Medium",     "L": "Large"}
REM_MAP  = {0: "On-site (0%)", 50: "Hybrid (50%)", 100: "Remote (100%)"}

JOB_TITLES = ["Data Scientist","Data Engineer","ML Engineer","Data Analyst","Research Scientist","Other"]
REGIONS    = ["North America","Europe","Asia","South America","Africa","Oceania","Other"]

PALETTE = {
    "experience":   ["#BBDEFB","#64B5F6","#1E88E5","#0D47A1"],
    "job_title":    ["#C8E6C9","#66BB6A","#2E7D32","#1B5E20","#43A047","#00897B"],
    "region":       ["#FFE0B2","#FFA726","#E65100","#BF360C","#FF7043","#FF8A65"],
    "company_size": ["#E1BEE7","#AB47BC","#6A1B9A"],
    "remote":       ["#FFCCBC","#FF7043","#BF360C"],
    "employment":   ["#B2EBF2","#26C6DA","#00838F","#004D40"],
    "generic":      ["#CFD8DC","#78909C","#37474F"],
}

# Base scenarios always shown per experience level
BASE_SCENARIOS = {
    "EN": ["overall_market", "junior_career_paths"],
    "MI": ["overall_market", "mid_optimization"],
    "SE": ["overall_market", "senior_global_market"],
    "EX": ["overall_market", "executive_positioning"],
}

# Regions considered "low paying" — will trigger region comparison insight
LOW_PAYING_REGIONS = {"Asia", "Africa", "South America", "Oceania", "Other"}


# ── Smart insight selector ────────────────────────────────────────────────────
def get_scenarios(
    experience:      str,
    job_title:       str,
    company_size:    str,
    remote_ratio:    int,
    company_region:  str,
    employee_region: str,
    employment:      str,
    predicted_salary: float,
    all_insights:    dict,
) -> tuple[list[str], list[str]]:
    """
    Returns (scenario_list, trigger_reasons).

    scenario_list  — ordered list of scenario keys to show
    trigger_reasons — human-readable explanation of why each was added
    """
    scenarios = list(BASE_SCENARIOS.get(experience, ["overall_market"]))
    reasons:  list[str] = [f"Based on your {EXP_MAP[experience]} experience"]

    def _add(key: str, reason: str) -> None:
        if key not in scenarios and key in all_insights:
            scenarios.append(key)
            reasons.append(reason)

    # ── Trigger: low-paying region ────────────────────────────────────────────
    if company_region in LOW_PAYING_REGIONS or employee_region in LOW_PAYING_REGIONS:
        _add("region_comparison",
             f"You are based in {employee_region or company_region} — see how salaries compare globally")

    # ── Trigger: on-site worker ───────────────────────────────────────────────
    if remote_ratio == 0:
        _add("remote_work_impact",
             "You work on-site — see how much remote workers earn for the same role")

    # ── Trigger: small company ────────────────────────────────────────────────
    if company_size == "S":
        _add("company_size_strategy",
             "You are at a small company — see the salary gap vs large companies")

    # ── Trigger: Data Analyst wanting to grow ────────────────────────────────
    if job_title == "Data Analyst":
        _add("career_switch",
             "As a Data Analyst, see how much you could earn by switching roles")

    # ── Trigger: contract / freelance ────────────────────────────────────────
    if employment in ("CT", "FL"):
        _add("employment_type",
             f"You are {EMP_MAP[employment]} — see how your salary compares to full-time")

    # ── Trigger: top earner (Executive + Large + North America) ──────────────
    if experience == "EX" and company_size == "L" and company_region == "North America":
        _add("top_earner_roast",
             "You are in the top compensation tier — we have something special for you 🏆")

    # ── Always add job title ranking if not already present ──────────────────
    _add("job_title_ranking", "See how your role ranks against other data science titles")

    # ── Cap at 6 tabs ─────────────────────────────────────────────────────────
    return scenarios[:6], reasons[:6]


# ── Chart builder ─────────────────────────────────────────────────────────────
def bar_chart(
    labels:     list,
    values:     list,
    title:      str,
    palette:    str      = "generic",
    highlight:  int | None = None,
    horizontal: bool     = False,
    height:     int      = 290,
) -> go.Figure:
    n       = len(labels)
    palette_colors = PALETTE.get(palette, PALETTE["generic"])
    if n <= len(palette_colors):
        colors = list(palette_colors[:n])
    else:
        colors = (palette_colors * ((n // len(palette_colors)) + 1))[:n]

    if highlight is not None and 0 <= highlight < n:
        colors[highlight] = palette_colors[-1]

    text_labels = [f"${v:,.0f}" for v in values]

    if horizontal:
        trace = go.Bar(
            x=values, y=labels, orientation="h",
            marker_color=colors,
            text=text_labels, textposition="outside",
            textfont=dict(size=11, color="#ffffff"),
            cliponaxis=False,
            hovertemplate="%{y}<br><b>$%{x:,.0f}</b><extra></extra>",
        )
        xaxis = dict(tickformat="$,.0f", gridcolor="#EEEEEE", showline=False, zeroline=False)
        yaxis = dict(showgrid=False, showline=False, tickfont=dict(size=11))
    else:
        trace = go.Bar(
            x=labels, y=values,
            marker_color=colors,
            text=text_labels, textposition="outside",
            textfont=dict(size=11, color="#ffffff"),
            cliponaxis=False,
            hovertemplate="%{x}<br><b>$%{y:,.0f}</b><extra></extra>",
        )
        xaxis = dict(tickfont=dict(color="#333333", size=11), showgrid=False, showline=False)
        yaxis = dict(tickfont=dict(color="#333333"),tickformat="$,.0f", gridcolor="#EEEEEE", showline=False, zeroline=False)

    fig = go.Figure(trace)
    fig.update_layout(
        template="plotly_white",

        font=dict(color="#333333"),

        xaxis=dict(
            tickfont=dict(color="#333333"),
            gridcolor="#EEEEEE",
            showline=False,
            zeroline=False
        ),

        yaxis=dict(
            tickfont=dict(color="#333333"),
            gridcolor="#EEEEEE",
            showline=False,
            zeroline=False
        ),

        plot_bgcolor="white",
        paper_bgcolor="white",

        height=height,
        margin=dict(t=45, b=25, l=10, r=65),
        showlegend=False,
    )
    return fig


# ── Supabase ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_supabase() -> Client:
    url = os.getenv("SUPABASE_URL") or st.secrets.get("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY") or st.secrets.get("SUPABASE_KEY", "")
    if not url or not key:
        st.error("Missing Supabase credentials.")
        st.stop()
    return create_client(url, key)


@st.cache_data(ttl=300)
def load_insights() -> dict[str, dict]:
    try:
        resp = get_supabase().table("insights").select("*").execute()
        result = {}
        for row in (resp.data or []):
            try:
                full = json.loads(row["narrative"])
            except Exception:
                full = {"title": row.get("title",""), "narrative": row.get("narrative","")}
            full["chart_type"] = row.get("chart_type", "none")
            full["chart_data"] = row.get("chart_data") or {}
            full["audience"]   = row.get("audience", "ALL")
            result[row["scenario"]] = full
        return result
    except Exception as exc:
        st.warning(f"Could not load insights: {exc}")
        return {}


@st.cache_data(ttl=300)
def load_aggregates() -> pd.DataFrame:
    try:
        resp = get_supabase().table("salary_aggregates").select("*").execute()
        return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()
    except Exception as exc:
        st.warning(f"Could not load aggregates: {exc}")
        return pd.DataFrame()


# ── Insight card ──────────────────────────────────────────────────────────────
def render_insight_card(insight: dict) -> None:
    headline  = insight.get("headline",  "")
    narrative = insight.get("narrative", "")
    fun_fact  = insight.get("fun_fact",  "")

    if headline:
        st.markdown(f'<div class="headline-stat">📊 {headline}</div>', unsafe_allow_html=True)
    if narrative:
        st.markdown(f'<p class="insight-narrative">{narrative}</p>', unsafe_allow_html=True)

    chart_type = insight.get("chart_type", "none")
    chart_data = insight.get("chart_data") or {}
    if chart_type == "bar" and chart_data:
        labels = chart_data.get("labels", [])
        values = chart_data.get("values", [])
        hi     = chart_data.get("highlight_index", 0)
        title  = chart_data.get("title", "")
        if labels and values:
            fig = bar_chart(labels, values, title, palette="generic", highlight=hi)
            st.plotly_chart(fig, use_container_width=True)

    if fun_fact:
        st.markdown(f'<div class="fun-fact">💡 {fun_fact}</div>', unsafe_allow_html=True)


# ── Header + Nav ──────────────────────────────────────────────────────────────
st.markdown("## 💰 Salary Predictor")
st.caption("Live predictions · Personalized AI insights · Random Forest model")
st.write("")

nav1, nav2, _space = st.columns([1, 1, 6])
with nav1:
    if st.button("💰 Predict", use_container_width=True,
                 type="primary" if page == "predict" else "secondary"):
        st.session_state["page"] = "predict"
        st.rerun()
with nav2:
    if st.button("📊 Explore", use_container_width=True,
                 type="primary" if page == "explore" else "secondary"):
        st.session_state["page"] = "explore"
        st.rerun()

st.divider()
page = st.session_state["page"]


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
if page == "predict":

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            experience   = st.selectbox("Experience Level", list(EXP_MAP.keys()), format_func=lambda x: EXP_MAP[x])
            employment   = st.selectbox("Employment Type",  list(EMP_MAP.keys()), format_func=lambda x: EMP_MAP[x])
            job_title    = st.selectbox("Job Title",        JOB_TITLES)
        with c2:
            company_size   = st.selectbox("Company Size",   list(SIZE_MAP.keys()), format_func=lambda x: SIZE_MAP[x])
            remote_ratio   = st.selectbox("Remote Policy",  [0, 50, 100],          format_func=lambda x: REM_MAP[x])
        with c3:
            company_region  = st.selectbox("Company Region", REGIONS)
            employee_region = st.selectbox("Your Region",    REGIONS)

        submitted = st.form_submit_button(
            "🚀 Predict My Salary",
            use_container_width=True,
            type="primary",
        )

    if submitted:
        with st.spinner("Calculating..."):
            try:
                resp = requests.get(
                    f"{API_URL}/predict",
                    params={
                        "experience_level": experience,
                        "employment_type":  employment,
                        "job_title":        job_title,
                        "company_size":     company_size,
                        "remote_ratio":     remote_ratio,
                        "company_region":   company_region,
                        "employee_region":  employee_region,
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                salary = resp.json()["predicted_salary_usd"]

                # ── Salary card ────────────────────────────────────────────
                st.markdown(f"""
                <div class="salary-card">
                    <div style="font-size:0.95rem;opacity:0.88;margin-bottom:0.4rem;">
                        {EXP_MAP[experience]} · {job_title} · {SIZE_MAP[company_size]} Company · {REM_MAP[remote_ratio]} · {company_region}
                    </div>
                    <div style="font-size:3.2rem;font-weight:800;letter-spacing:-1px;line-height:1.1;">
                        ${salary:,.0f}
                    </div>
                    <div style="font-size:0.95rem;opacity:0.88;margin-top:0.4rem;">
                        estimated annual salary · USD
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Smart scenario selection ───────────────────────────────
                all_insights = load_insights()
                scenarios, reasons = get_scenarios(
                    experience      = experience,
                    job_title       = job_title,
                    company_size    = company_size,
                    remote_ratio    = remote_ratio,
                    company_region  = company_region,
                    employee_region = employee_region,
                    employment      = employment,
                    predicted_salary= salary,
                    all_insights    = all_insights,
                )

                available = [s for s in scenarios if s in all_insights]

                if available:
                    st.subheader(f"📖 Your Personalized Insights")

                    # Show why each insight was selected as small badges
                    badges_html = "".join(
                        f'<span class="trigger-badge">✓ {r}</span>'
                        for r in reasons
                    )
                    st.markdown(badges_html, unsafe_allow_html=True)

                    tab_labels = [all_insights[s].get("title", s) for s in available]
                    tabs       = st.tabs(tab_labels)
                    for tab, scenario in zip(tabs, available):
                        with tab:
                            render_insight_card(all_insights[scenario])
                else:
                    st.info("No insights found. Run `python pipeline/run_pipeline.py` first.")

            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach API at `{API_URL}`. Run: `uvicorn api.main:app --reload`")
            except requests.exceptions.HTTPError as e:
                st.error(f"API error: {e.response.text}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EXPLORE
# ═══════════════════════════════════════════════════════════════════════════════
else:
    df_agg   = load_aggregates()
    insights = load_insights()

    if df_agg.empty:
        st.info("No data yet. Run `python pipeline/run_pipeline.py` first.")
        st.stop()

    def get_cat(cat: str) -> pd.DataFrame:
        return df_agg[df_agg["category"] == cat].copy()

    # ── Opening story ─────────────────────────────────────────────────────────
    if "overall_market" in insights:
        ov = insights["overall_market"]
        st.markdown(f"### {ov.get('title','Market Overview')}")
        if ov.get("headline"):
            st.markdown(f'<div class="headline-stat">📊 {ov["headline"]}</div>', unsafe_allow_html=True)
        if ov.get("narrative"):
            st.markdown(f'<p class="insight-narrative">{ov["narrative"]}</p>', unsafe_allow_html=True)
        if ov.get("fun_fact"):
            st.markdown(f'<div class="fun-fact">💡 {ov["fun_fact"]}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Step 1 ────────────────────────────────────────────────────────────────
    st.markdown("### 📈 Step 1 — Experience Is Everything")
    exp_df    = get_cat("experience")
    order     = ["EN","MI","SE","EX"]
    lmap      = {"EN":"Entry Level","MI":"Mid Level","SE":"Senior","EX":"Executive"}
    exp_df["label"] = exp_df["key"].map(lmap)
    exp_df = exp_df.set_index("key").reindex(order).reset_index().dropna(subset=["avg_salary"])

    if "junior_career_paths" in insights and insights["junior_career_paths"].get("headline"):
        st.markdown(f'<div class="headline-stat">📊 {insights["junior_career_paths"]["headline"]}</div>', unsafe_allow_html=True)

    fig1 = bar_chart(
        labels    = exp_df["label"].tolist(),
        values    = exp_df["avg_salary"].tolist(),
        title     = "Every level up earns you more — the biggest jump is at the top",
        palette   = "experience",
        highlight = 3,
        height    = 300,
    )
    st.plotly_chart(fig1, use_container_width=True)

    if "junior_career_paths" in insights and insights["junior_career_paths"].get("fun_fact"):
        st.markdown(f'<div class="fun-fact">💡 {insights["junior_career_paths"]["fun_fact"]}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Step 2 ────────────────────────────────────────────────────────────────
    st.markdown("### 💼 Step 2 — Your Title Changes Your Salary More Than You Think")
    title_df = get_cat("job_title").sort_values("avg_salary", ascending=True)

    if "job_title_ranking" in insights and insights["job_title_ranking"].get("headline"):
        st.markdown(f'<div class="headline-stat">📊 {insights["job_title_ranking"]["headline"]}</div>', unsafe_allow_html=True)

    fig2 = bar_chart(
        labels     = title_df["key"].tolist(),
        values     = title_df["avg_salary"].tolist(),
        title      = "Same seniority level, very different paycheck depending on your title",
        palette    = "job_title",
        highlight  = len(title_df) - 1,
        horizontal = True,
        height     = 320,
    )
    st.plotly_chart(fig2, use_container_width=True)

    if "job_title_ranking" in insights and insights["job_title_ranking"].get("fun_fact"):
        st.markdown(f'<div class="fun-fact">💡 {insights["job_title_ranking"]["fun_fact"]}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Step 3 ────────────────────────────────────────────────────────────────
    st.markdown("### 🌍 Step 3 — Geography Still Matters — A Lot")
    reg_df = get_cat("region").sort_values("avg_salary", ascending=True)

    if "region_comparison" in insights and insights["region_comparison"].get("headline"):
        st.markdown(f'<div class="headline-stat">📊 {insights["region_comparison"]["headline"]}</div>', unsafe_allow_html=True)
    if "region_comparison" in insights and insights["region_comparison"].get("narrative"):
        st.markdown(f'<p class="insight-narrative">{insights["region_comparison"]["narrative"]}</p>', unsafe_allow_html=True)

    fig3 = bar_chart(
        labels     = reg_df["key"].tolist(),
        values     = reg_df["avg_salary"].tolist(),
        title      = "The same role pays very differently depending on where the company is based",
        palette    = "region",
        highlight  = len(reg_df) - 1,
        horizontal = True,
        height     = 320,
    )
    st.plotly_chart(fig3, use_container_width=True)

    if "region_comparison" in insights and insights["region_comparison"].get("fun_fact"):
        st.markdown(f'<div class="fun-fact">💡 {insights["region_comparison"]["fun_fact"]}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Step 4 ────────────────────────────────────────────────────────────────
    st.markdown("### 🔧 Step 4 — Two More Levers Worth Knowing")

    col_l, col_r = st.columns(2)

    with col_l:
        if "remote_work_impact" in insights and insights["remote_work_impact"].get("headline"):
            st.markdown(f'<div class="headline-stat">📊 {insights["remote_work_impact"]["headline"]}</div>', unsafe_allow_html=True)

        rem_df = get_cat("remote").copy()
        rem_df["label"] = rem_df["key"].apply(
            lambda x: {0:"On-site",50:"Hybrid",100:"Remote"}.get(int(float(x)), str(x))
        )
        rem_df = rem_df.sort_values("avg_salary")
        fig4 = bar_chart(
            labels    = rem_df["label"].tolist(),
            values    = rem_df["avg_salary"].tolist(),
            title     = "Remote premium or penalty?",
            palette   = "remote",
            highlight = len(rem_df) - 1,
            height    = 280,
        )
        st.plotly_chart(fig4, use_container_width=True)

        if "remote_work_impact" in insights and insights["remote_work_impact"].get("fun_fact"):
            st.markdown(f'<div class="fun-fact">💡 {insights["remote_work_impact"]["fun_fact"]}</div>', unsafe_allow_html=True)

    with col_r:
        if "company_size_strategy" in insights and insights["company_size_strategy"].get("headline"):
            st.markdown(f'<div class="headline-stat">📊 {insights["company_size_strategy"]["headline"]}</div>', unsafe_allow_html=True)

        size_df = get_cat("company_size").copy()
        size_df["label"] = size_df["key"].map(SIZE_MAP)
        size_df = size_df.sort_values("avg_salary")
        fig5 = bar_chart(
            labels    = size_df["label"].tolist(),
            values    = size_df["avg_salary"].tolist(),
            title     = "Big company vs startup — real numbers",
            palette   = "company_size",
            highlight = len(size_df) - 1,
            height    = 280,
        )
        st.plotly_chart(fig5, use_container_width=True)

        if "company_size_strategy" in insights and insights["company_size_strategy"].get("fun_fact"):
            st.markdown(f'<div class="fun-fact">💡 {insights["company_size_strategy"]["fun_fact"]}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Step 5 — Employment type ───────────────────────────────────────────────
    st.markdown("### 📋 Step 5 — Full-time vs Contract vs Freelance")

    emp_df = get_cat("employment_type").copy()
    emp_label_map = {"FT": "Full-time", "PT": "Part-time", "CT": "Contract", "FL": "Freelance"}
    emp_df["label"] = emp_df["key"].map(emp_label_map)
    emp_df = emp_df.sort_values("avg_salary")

    if "employment_type" in insights and insights["employment_type"].get("headline"):
        st.markdown(f'<div class="headline-stat">📊 {insights["employment_type"]["headline"]}</div>', unsafe_allow_html=True)
    if "employment_type" in insights and insights["employment_type"].get("narrative"):
        st.markdown(f'<p class="insight-narrative">{insights["employment_type"]["narrative"]}</p>', unsafe_allow_html=True)

    fig6 = bar_chart(
        labels    = emp_df["label"].tolist(),
        values    = emp_df["avg_salary"].tolist(),
        title     = "The employment type you choose changes your salary significantly",
        palette   = "employment",
        highlight = len(emp_df) - 1,
        height    = 280,
    )
    st.plotly_chart(fig6, use_container_width=True)

    if "employment_type" in insights and insights["employment_type"].get("fun_fact"):
        st.markdown(f'<div class="fun-fact">💡 {insights["employment_type"]["fun_fact"]}</div>', unsafe_allow_html=True)

    st.divider()

    # ── All insight cards ─────────────────────────────────────────────────────
    st.markdown("### 🧠 All AI-Generated Insights")
    st.caption("Generated once by a local LLM · Updates when you re-run the pipeline")

    aud_filter = st.radio(
        "Show insights for:",
        ["Everyone","Entry Level","Mid Level","Senior","Executive"],
        horizontal=True,
    )
    aud_code = {
        "Everyone":    None,
        "Entry Level": "EN",
        "Mid Level":   "MI",
        "Senior":      "SE",
        "Executive":   "EX",
    }[aud_filter]

    badges = {"EN":"🟢","MI":"🔵","SE":"🔴","EX":"⭐","ALL":"🌐"}
    aud_labels = {"EN":"Entry Level","MI":"Mid Level","SE":"Senior","EX":"Executive","ALL":"All Levels"}

    shown = 0
    for scenario, insight in insights.items():
        aud = insight.get("audience","ALL")
        if aud_code and aud not in (aud_code, "ALL"):
            continue
        badge = badges.get(aud, "🌐")
        label = aud_labels.get(aud, aud)
        with st.expander(f"{badge} {label}  ·  {insight.get('title', scenario)}"):
            render_insight_card(insight)
        shown += 1

    if shown == 0:
        st.info("No insights match this filter.")