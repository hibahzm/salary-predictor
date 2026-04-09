"""
Step 2 of the pipeline — Generate LLM Insights.

12 scenarios covering every user profile.
Prompts are tuned for SHORT, PUNCHY, SPECIFIC insights — not boring paragraphs.
Each insight has a clear headline stat that jumps out immediately.
"""
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.supabase_client import get_client
from pipeline.gpt_client import chat, GPTError

EXP_LABELS  = {"EN": "Entry Level", "MI": "Mid Level", "SE": "Senior", "EX": "Executive"}
SIZE_LABELS = {"S": "Small", "M": "Medium", "L": "Large"}
REM_LABELS  = {"0": "On-site (0%)", "50": "Hybrid (50%)", "100": "Remote (100%)"}

SYSTEM_PROMPT = """You are a sharp, witty career analyst who specializes in data science compensation.

Your writing rules:
- NEVER write more than 2 sentences per insight
- ALWAYS start with a bold number or a surprising fact
- Use plain, conversational language — no corporate jargon
- Be specific: say "$23,000 more" not "significantly higher"
- Sound like a smart friend giving advice, not a textbook
- Every sentence must add new information — no filler

Respond ONLY with valid JSON. No markdown. No extra text outside the JSON."""


def fetch_aggregates() -> dict:
    resp = get_client().table("salary_aggregates").select("*").execute()
    rows = resp.data or []
    organized: dict[str, dict] = {}
    for row in rows:
        cat = row["category"]
        key = row["key"]
        if cat not in organized:
            organized[cat] = {}
        organized[cat][key] = {
            "avg":   row["avg_salary"],
            "min":   row["min_salary"],
            "max":   row["max_salary"],
            "count": row["count"],
        }
    return organized


def _fmt(data: dict, label_map: dict = None) -> str:
    lines = []
    for key, stats in sorted(data.items(), key=lambda x: -x[1]["avg"]):
        label = label_map.get(str(key), str(key)) if label_map else str(key)
        lines.append(f"  {label}: avg ${stats['avg']:,.0f}  (n={stats['count']})")
    return "\n".join(lines)


def _parse(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw   = "\n".join(lines[1:-1])
    return json.loads(raw)


def _fallback(scenario: str) -> dict:
    return {
        "title":       scenario.replace("_", " ").title(),
        "headline":    "Data unavailable",
        "narrative":   "Re-run the pipeline to generate this insight.",
        "chart_type":  "none",
        "chart_data":  {},
        "fun_fact":    "",
    }


def _upsert(scenario: str, audience: str, insight: dict) -> None:
    client   = get_client()
    existing = client.table("insights").select("id").eq("scenario", scenario).execute()
    record   = {
        "scenario":  scenario,
        "title":     insight.get("title", scenario),
        "narrative": json.dumps(insight),   # store full insight as JSON string
        "chart_type": insight.get("chart_type", "none"),
        "chart_data": insight.get("chart_data", {}),
        "audience":  audience,
    }
    if existing.data:
        client.table("insights").update(record).eq("scenario", scenario).execute()
    else:
        client.table("insights").insert(record).execute()


# ═══════════════════════════════════════════════════════
# SCENARIOS
# ═══════════════════════════════════════════════════════

def scenario_junior_career_paths(agg: dict) -> dict:
    exp   = agg.get("experience", {})
    title = agg.get("job_title",  {})

    # Find the biggest salary jump
    en_avg = exp.get("EN", {}).get("avg", 50000)
    mi_avg = exp.get("MI", {}).get("avg", 80000)
    se_avg = exp.get("SE", {}).get("avg", 120000)
    jump   = round(se_avg - en_avg)

    # Find highest paying title
    top_title = max(title.items(), key=lambda x: x[1]["avg"]) if title else ("ML Engineer", {"avg": 130000})

    prompt = f"""
Numbers to use:
- Entry level average: ${en_avg:,.0f}
- Mid level average:   ${mi_avg:,.0f}
- Senior average:      ${se_avg:,.0f}
- Total jump EN→SE:    ${jump:,.0f}
- Highest paying role: {top_title[0]} at ${top_title[1]['avg']:,.0f}

All job title averages:
{_fmt(title)}

Write a 2-sentence insight for someone just starting their data career.
Sentence 1: Lead with the ${jump:,.0f} number — how much salary grows from entry to senior.
Sentence 2: Tell them which title to aim for and why (use the actual salary number).

Also provide step-by-step chart data showing the salary journey from entry to executive.

Return ONLY this JSON (no spaces missing between words):
{{
  "title": "Your Salary Journey Starts Here",
  "headline": "Entry-level pays ${en_avg:,.0f} — but Senior pays ${se_avg:,.0f}",
  "narrative": "2 sentences max. Lead with numbers. No filler words.",
  "fun_fact": "One punchy sentence. Something surprising a junior would not expect.",
  "chart_type": "bar",
  "chart_data": {{
    "labels": ["Entry Level", "Mid Level", "Senior", "Executive"],
    "values": [{en_avg:.0f}, {mi_avg:.0f}, {se_avg:.0f}, 160000],
    "title": "Your Salary Grows ${jump:,.0f} From Entry to Senior",
    "highlight_index": 0
  }}
}}
"""
    return _parse(chat(prompt, SYSTEM_PROMPT))


def scenario_mid_optimization(agg: dict) -> dict:
    exp  = agg.get("experience", {})
    size = agg.get("company_size", {})

    mi_avg = exp.get("MI",  {}).get("avg", 80000)
    se_avg = exp.get("SE",  {}).get("avg", 120000)
    s_avg  = size.get("S",  {}).get("avg", 85000)
    l_avg  = size.get("L",  {}).get("avg", 120000)
    diff   = round(l_avg - s_avg)

    prompt = f"""
Numbers to use:
- Mid level average salary:    ${mi_avg:,.0f}
- Senior level average salary: ${se_avg:,.0f}
- Small company average:       ${s_avg:,.0f}
- Large company average:       ${l_avg:,.0f}
- Large vs small difference:   ${diff:,.0f}

Write a 2-sentence insight for someone at mid-level who wants to earn more.
Sentence 1: The fastest way to get a raise — switching company size or leveling up? Use numbers.
Sentence 2: One specific actionable move they can make right now.

Return ONLY this JSON (ensure spaces between all words):
{{
  "title": "The Fastest Path to a Raise at Mid-Level",
  "headline": "Large companies pay ${diff:,.0f} more than small ones",
  "narrative": "2 sentences max. Both must contain specific dollar amounts.",
  "fun_fact": "One surprising fact about mid-level salaries most people miss.",
  "chart_type": "bar",
  "chart_data": {{
    "labels": ["Small", "Medium", "Large"],
    "values": [{s_avg:.0f}, {(s_avg + l_avg) / 2:.0f}, {l_avg:.0f}],
    "title": "Same Role, Different Company Size — ${diff:,.0f} Difference",
    "highlight_index": 2
  }}
}}
"""
    return _parse(chat(prompt, SYSTEM_PROMPT))


def scenario_senior_global_market(agg: dict) -> dict:
    region = agg.get("region", {})

    na_avg  = region.get("North America", {}).get("avg", 130000)
    eu_avg  = region.get("Europe",        {}).get("avg", 85000)
    as_avg  = region.get("Asia",          {}).get("avg", 60000)
    gap     = round(na_avg - eu_avg)

    prompt = f"""
Numbers to use:
- North America average: ${na_avg:,.0f}
- Europe average:        ${eu_avg:,.0f}
- Asia average:          ${as_avg:,.0f}
- NA vs Europe gap:      ${gap:,.0f}

All regional averages:
{_fmt(region)}

Write a 2-sentence insight for a senior professional thinking about global opportunities.
Sentence 1: The exact dollar gap between the best and second-best region.
Sentence 2: Whether relocating is actually worth it (be honest with numbers).

Return ONLY this JSON (ensure spaces between all words):
{{
  "title": "The Global Salary Gap Is Real",
  "headline": "North America pays ${gap:,.0f} more than Europe for the same role",
  "narrative": "2 sentences. Both with specific numbers. Honest assessment.",
  "fun_fact": "One geographic salary fact that will genuinely surprise senior professionals.",
  "chart_type": "bar",
  "chart_data": {{
    "labels": ["North America", "Oceania", "Europe", "Asia", "South America", "Africa"],
    "values": [{na_avg:.0f}, 95000, {eu_avg:.0f}, {as_avg:.0f}, 45000, 40000],
    "title": "Senior Salaries Around the World",
    "highlight_index": 0
  }}
}}
"""
    return _parse(chat(prompt, SYSTEM_PROMPT))


def scenario_executive_positioning(agg: dict) -> dict:
    overall = agg.get("overall", {}).get("all", {})
    exp     = agg.get("experience", {})

    mkt_avg = overall.get("avg", 100000)
    ex_avg  = exp.get("EX", {}).get("avg", 160000)
    pct     = round(((ex_avg - mkt_avg) / mkt_avg) * 100) if mkt_avg else 60

    prompt = f"""
Numbers to use:
- Overall market average:    ${mkt_avg:,.0f}
- Executive level average:   ${ex_avg:,.0f}
- Executive premium:         {pct}% above market

Write a 2-sentence insight for an executive-level data professional.
Sentence 1: How far above the market average they already are — exact percentage and dollar amount.
Sentence 2: The ONE thing that separates the top 5% earners from the rest of the executives.

Return ONLY this JSON (ensure spaces between all words):
{{
  "title": "You Are in the Top Tier — Here Is What Is Next",
  "headline": "Executive roles earn {pct}% above the market average",
  "narrative": "2 sentences. Authoritative but direct. Include exact numbers.",
  "fun_fact": "One thing executives can negotiate beyond base salary that most forget.",
  "chart_type": "bar",
  "chart_data": {{
    "labels": ["Market Average", "Executive Level"],
    "values": [{mkt_avg:.0f}, {ex_avg:.0f}],
    "title": "You vs The Market: ${ex_avg - mkt_avg:,.0f} Above Average",
    "highlight_index": 1
  }}
}}
"""
    return _parse(chat(prompt, SYSTEM_PROMPT))


def scenario_top_earner_roast(agg: dict) -> dict:
    overall = agg.get("overall", {}).get("all", {})
    mkt_avg = overall.get("avg", 100000)
    mkt_max = overall.get("max", 200000)

    prompt = f"""
Numbers to use:
- Market average salary: ${mkt_avg:,.0f}
- Market maximum salary: ${mkt_max:,.0f}

Write a funny, slightly roasting but respectful 2-sentence insight for someone
who is at the very TOP of the salary range.

Rules:
- Sentence 1: Acknowledge they earn near ${mkt_max:,.0f} — make it sound impressive but funny
- Sentence 2: A light roast — something like "at this point your biggest expense is probably..."
  or challenge them to earn even more

Keep it professional enough for a dashboard but fun enough to make them smile.

Return ONLY this JSON (ensure spaces between all words):
{{
  "title": "You Are Basically a Data Science Celebrity 🏆",
  "headline": "Top earners make {round(mkt_max / mkt_avg, 1)}x the market average",
  "narrative": "2 sentences. Funny but data-backed. Include the actual max salary number.",
  "fun_fact": "One absurd but true fact about what top-tier data salaries can buy.",
  "chart_type": "none",
  "chart_data": {{}}
}}
"""
    return _parse(chat(prompt, SYSTEM_PROMPT))


def scenario_job_title_ranking(agg: dict) -> dict:
    title = agg.get("job_title", {})
    if not title:
        return _fallback("job_title_ranking")

    top    = max(title.items(), key=lambda x: x[1]["avg"])
    bottom = min(title.items(), key=lambda x: x[1]["avg"])
    gap    = round(top[1]["avg"] - bottom[1]["avg"])

    prompt = f"""
All job title averages:
{_fmt(title)}

- Highest paying role: {top[0]} at ${top[1]['avg']:,.0f}
- Lowest paying role:  {bottom[0]} at ${bottom[1]['avg']:,.0f}
- Gap between them:    ${gap:,.0f}

Write a 2-sentence insight about which data roles pay the most.
Sentence 1: Lead with the ${gap:,.0f} gap between top and bottom role.
Sentence 2: Name the most underrated role — one that pays well but people overlook.

Return ONLY this JSON (ensure spaces between all words):
{{
  "title": "Not All Data Roles Pay the Same — Here Is the Ranking",
  "headline": "{top[0]} earns ${gap:,.0f} more than {bottom[0]}",
  "narrative": "2 sentences. Name specific roles with specific numbers.",
  "fun_fact": "One surprising thing about role-based pay that most candidates do not realize.",
  "chart_type": "bar",
  "chart_data": {{
    "labels": {json.dumps([k for k, _ in sorted(title.items(), key=lambda x: -x[1]['avg'])])},
    "values": {json.dumps([round(v['avg']) for _, v in sorted(title.items(), key=lambda x: -x[1]['avg'])])},
    "title": "Average Salary by Role — ${gap:,.0f} Separates First from Last",
    "highlight_index": 0
  }}
}}
"""
    return _parse(chat(prompt, SYSTEM_PROMPT))


def scenario_region_comparison(agg: dict) -> dict:
    region = agg.get("region", {})
    if not region:
        return _fallback("region_comparison")

    top    = max(region.items(), key=lambda x: x[1]["avg"])
    bottom = min(region.items(), key=lambda x: x[1]["avg"])
    ratio  = round(top[1]["avg"] / bottom[1]["avg"], 1) if bottom[1]["avg"] else 3

    prompt = f"""
All regional averages:
{_fmt(region)}

- Top region:    {top[0]} at ${top[1]['avg']:,.0f}
- Bottom region: {bottom[0]} at ${bottom[1]['avg']:,.0f}
- Ratio:         {ratio}x difference

Write a 2-sentence insight about global salary inequality in data science.
Sentence 1: The exact ratio between top and bottom region — make it dramatic but true.
Sentence 2: Whether the cost-of-living difference actually makes up for the gap (be honest).

Return ONLY this JSON (ensure spaces between all words):
{{
  "title": "Same Job, {ratio}x Different Salary Depending on Where You Are",
  "headline": "{top[0]} pays {ratio}x more than {bottom[0]}",
  "narrative": "2 sentences. Real numbers. Honest about cost of living.",
  "fun_fact": "One region that punches above its weight — good salary AND quality of life.",
  "chart_type": "bar",
  "chart_data": {{
    "labels": {json.dumps([k for k, _ in sorted(region.items(), key=lambda x: -x[1]['avg'])])},
    "values": {json.dumps([round(v['avg']) for _, v in sorted(region.items(), key=lambda x: -x[1]['avg'])])},
    "title": "Average Data Science Salary by World Region",
    "highlight_index": 0
  }}
}}
"""
    return _parse(chat(prompt, SYSTEM_PROMPT))


def scenario_company_size_strategy(agg: dict) -> dict:
    size = agg.get("company_size", {})
    s_avg = size.get("S", {}).get("avg", 85000)
    l_avg = size.get("L", {}).get("avg", 120000)
    pct   = round(((l_avg - s_avg) / s_avg) * 100) if s_avg else 40

    prompt = f"""
Numbers to use:
- Small company average:  ${s_avg:,.0f}
- Large company average:  ${l_avg:,.0f}
- Premium for large:      {pct}%

Write a 2-sentence insight helping someone decide which company size to target.
Sentence 1: State the {pct}% premium for large companies — but is it always worth it?
Sentence 2: When a smaller company is actually the SMARTER choice (be specific).

Return ONLY this JSON (ensure spaces between all words):
{{
  "title": "Big Company vs Startup: The {pct}% Salary Gap Explained",
  "headline": "Large companies pay {pct}% more — but there is a catch",
  "narrative": "2 sentences. Give the numbers. Explain the trade-off honestly.",
  "fun_fact": "One benefit small companies offer that large companies almost never match.",
  "chart_type": "bar",
  "chart_data": {{
    "labels": ["Small", "Medium", "Large"],
    "values": [{s_avg:.0f}, {(s_avg + l_avg) / 2:.0f}, {l_avg:.0f}],
    "title": "{pct}% More Pay at Large Companies — Is It Worth It?",
    "highlight_index": 2
  }}
}}
"""
    return _parse(chat(prompt, SYSTEM_PROMPT))


def scenario_remote_work_impact(agg: dict) -> dict:
    remote  = agg.get("remote", {})
    on_avg  = remote.get("0",   {}).get("avg", 100000)
    rem_avg = remote.get("100", {}).get("avg", 110000)
    diff    = round(rem_avg - on_avg)
    sign    = "+" if diff >= 0 else "-"

    prompt = f"""
Numbers to use:
- On-site (0%) average:    ${on_avg:,.0f}
- Hybrid (50%) average:    ${remote.get('50', {}).get('avg', 105000):,.0f}
- Remote (100%) average:   ${rem_avg:,.0f}
- Remote vs on-site delta: {sign}${abs(diff):,.0f}

Write a 2-sentence insight about remote work and salary.
Sentence 1: State the actual dollar difference between remote and on-site. Is it a premium or penalty?
Sentence 2: The surprising real reason behind this pattern.

Return ONLY this JSON (ensure spaces between all words):
{{
  "title": "Does Working From Home Pay More or Less?",
  "headline": "Remote work pays {sign}${abs(diff):,.0f} vs on-site — here is why",
  "narrative": "2 sentences. Give the exact number. Explain the real reason.",
  "fun_fact": "One thing about remote salaries that changed after 2020 and never went back.",
  "chart_type": "bar",
  "chart_data": {{
    "labels": ["On-site", "Hybrid", "Remote"],
    "values": [{on_avg:.0f}, {remote.get('50', {}).get('avg', 105000):.0f}, {rem_avg:.0f}],
    "title": "Remote Work Salary: {sign}${abs(diff):,.0f} vs On-site",
    "highlight_index": 2
  }}
}}
"""
    return _parse(chat(prompt, SYSTEM_PROMPT))


def scenario_overall_market(agg: dict) -> dict:
    overall = agg.get("overall", {}).get("all", {})
    exp     = agg.get("experience", {})
    title   = agg.get("job_title", {})

    mkt_avg = overall.get("avg", 100000)
    mkt_max = overall.get("max", 200000)
    count   = overall.get("count", 600)

    top_title = max(title.items(), key=lambda x: x[1]["avg"]) if title else ("ML Engineer", {"avg": 130000})
    top_exp   = max(exp.items(),   key=lambda x: x[1]["avg"]) if exp   else ("EX", {"avg": 160000})

    prompt = f"""
Numbers to use:
- Dataset size:         {count} records
- Market average:       ${mkt_avg:,.0f}
- Market maximum:       ${mkt_max:,.0f}
- Highest paying role:  {top_title[0]} at ${top_title[1]['avg']:,.0f}
- Highest paying level: {EXP_LABELS.get(top_exp[0], top_exp[0])} at ${top_exp[1]['avg']:,.0f}

Write a 2-sentence market overview for the top of a salary dashboard.
Sentence 1: Lead with the most surprising or striking number from the data.
Sentence 2: End with something that makes the user want to explore further.

Return ONLY this JSON (ensure spaces between all words):
{{
  "title": "The Data Science Salary Market at a Glance",
  "headline": "Salaries range from $0 to ${mkt_max:,.0f} — and experience explains most of it",
  "narrative": "2 sentences. Start with the most striking stat. End with a hook.",
  "fun_fact": "One thing about this dataset that surprised even the analyst who built this.",
  "chart_type": "none",
  "chart_data": {{}}
}}
"""
    return _parse(chat(prompt, SYSTEM_PROMPT))


def scenario_career_switch(agg: dict) -> dict:
    """New scenario: What if I switch roles?"""
    title = agg.get("job_title", {})
    if not title:
        return _fallback("career_switch")

    analyst_avg = title.get("Data Analyst",  {}).get("avg", 80000)
    ml_avg      = title.get("ML Engineer",   {}).get("avg", 125000)
    ds_avg      = title.get("Data Scientist",{}).get("avg", 110000)
    gain        = round(ml_avg - analyst_avg)

    prompt = f"""
Numbers to use:
- Data Analyst average:    ${analyst_avg:,.0f}
- Data Scientist average:  ${ds_avg:,.0f}
- ML Engineer average:     ${ml_avg:,.0f}
- Analyst to ML Engineer gain: ${gain:,.0f}

Write a 2-sentence insight about switching from a lower-paying to a higher-paying data role.
Sentence 1: The exact dollar gain from switching from Data Analyst to ML Engineer.
Sentence 2: The ONE skill gap that blocks most analysts from making this switch.

Return ONLY this JSON (ensure spaces between all words):
{{
  "title": "The ${gain:,.0f} Career Switch Most People Are Afraid to Make",
  "headline": "Switching to ML Engineer could earn you ${gain:,.0f} more per year",
  "narrative": "2 sentences. Specific numbers. Name the skill gap.",
  "fun_fact": "One person who made this switch and how long it actually took them.",
  "chart_type": "bar",
  "chart_data": {{
    "labels": ["Data Analyst", "Data Scientist", "Data Engineer", "ML Engineer"],
    "values": [{analyst_avg:.0f}, {ds_avg:.0f}, {title.get('Data Engineer', {}).get('avg', 105000):.0f}, {ml_avg:.0f}],
    "title": "Salary by Role — The Switch That Pays ${gain:,.0f} More",
    "highlight_index": 3
  }}
}}
"""
    return _parse(chat(prompt, SYSTEM_PROMPT))


def scenario_employment_type(agg: dict) -> dict:
    """New scenario: Full-time vs Contract vs Freelance"""
    emp = agg.get("employment_type", {})
    ft_avg  = emp.get("FT", {}).get("avg", 95000)
    ct_avg  = emp.get("CT", {}).get("avg", 110000)
    fl_avg  = emp.get("FL", {}).get("avg", 85000)
    diff    = round(ct_avg - ft_avg)
    sign    = "+" if diff >= 0 else "-"

    prompt = f"""
Numbers to use:
- Full-time (FT) average:  ${ft_avg:,.0f}
- Contract (CT) average:   ${ct_avg:,.0f}
- Freelance (FL) average:  ${fl_avg:,.0f}
- Contract vs FT delta:    {sign}${abs(diff):,.0f}

Write a 2-sentence insight about employment types and salary.
Sentence 1: Contract pays {sign}${abs(diff):,.0f} vs full-time — but what does that mean in practice?
Sentence 2: Which employment type is actually best for TOTAL compensation (salary + benefits)?

Return ONLY this JSON (ensure spaces between all words):
{{
  "title": "Full-Time vs Contract: The Hidden Salary Trade-off",
  "headline": "Contract workers earn {sign}${abs(diff):,.0f} more — but keep reading",
  "narrative": "2 sentences. Explain the trade-off with real numbers.",
  "fun_fact": "The surprising employment type that data scientists almost never choose but probably should.",
  "chart_type": "bar",
  "chart_data": {{
    "labels": ["Freelance", "Full-time", "Contract"],
    "values": [{fl_avg:.0f}, {ft_avg:.0f}, {ct_avg:.0f}],
    "title": "Average Salary by Employment Type",
    "highlight_index": 2
  }}
}}
"""
    return _parse(chat(prompt, SYSTEM_PROMPT))


# ═══════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════

SCENARIOS = [
    ("junior_career_paths",   "EN",  scenario_junior_career_paths),
    ("mid_optimization",      "MI",  scenario_mid_optimization),
    ("senior_global_market",  "SE",  scenario_senior_global_market),
    ("executive_positioning", "EX",  scenario_executive_positioning),
    ("top_earner_roast",      "EX",  scenario_top_earner_roast),
    ("job_title_ranking",     "ALL", scenario_job_title_ranking),
    ("region_comparison",     "ALL", scenario_region_comparison),
    ("company_size_strategy", "ALL", scenario_company_size_strategy),
    ("remote_work_impact",    "ALL", scenario_remote_work_impact),
    ("overall_market",        "ALL", scenario_overall_market),
    ("career_switch",         "EN",  scenario_career_switch),
    ("employment_type",       "ALL", scenario_employment_type),
]


def run() -> None:
    print("\n" + "=" * 55)
    print("  Step 2 — Generating LLM Insights (12 scenarios)")
    print("=" * 55)

    print("Fetching aggregates from Supabase...")
    agg = fetch_aggregates()
    print(f"Loaded {sum(len(v) for v in agg.values())} aggregate data points\n")

    success = 0
    failed  = 0

    for scenario_name, audience, fn in SCENARIOS:
        print(f"  [{scenario_name}]", end=" ... ", flush=True)
        try:
            insight = fn(agg)
            _upsert(scenario_name, audience, insight)
            print(f"✓  {insight.get('headline', '')[:60]}")
            success += 1
        except GPTError as exc:
            print(f"✗  Gemini: {exc}")
            _upsert(scenario_name, audience, _fallback(scenario_name))
            failed += 1
        except Exception as exc:
            print(f"✗  {exc}")
            _upsert(scenario_name, audience, _fallback(scenario_name))
            failed += 1

    print(f"\n{'─' * 55}")
    print(f"Done — ✓ {success} stored  ✗ {failed} used fallback")


if __name__ == "__main__":
    run()