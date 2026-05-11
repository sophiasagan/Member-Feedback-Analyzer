# Member Feedback Analyzer

> NPS survey comments → AI classification → actionable executive insights

![demo placeholder](docs/demo.gif)

*Replace `docs/demo.gif` with a screen recording of the Streamlit dashboard*

---

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![Claude](https://img.shields.io/badge/Claude-Haiku%204.5-D97706?logo=anthropic&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-2.x-150458?logo=pandas&logoColor=white)
![Altair](https://img.shields.io/badge/Altair-5.x-4C78A8)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

Upload a credit union NPS CSV and get back a live dashboard:

- **Classify** every comment by sentiment, theme, urgency, and department in batches via Claude
- **Visualize** trends, theme volume, and sentiment distribution with interactive charts
- **Drill down** into any theme to see individual comments alongside an AI-generated insight card
- **Generate** a one-page executive summary of the top 3 actionable issues

---

## Architecture

```
CSV upload / sample data
        │
        ▼
┌───────────────────┐
│  engine/ingestor  │  Cleans text, validates nps_score, adds nps_category
└────────┬──────────┘
         │  pd.DataFrame (comment, nps_score, nps_category, channel, segment, date)
         ▼
┌───────────────────┐
│ engine/classifier │  Batches of 20 → claude-haiku-4-5 → sentiment + theme + routing
└────────┬──────────┘
         │  list[dict] merged back into DataFrame
         ▼
┌───────────────────┐
│ engine/summarizer │  Per-theme: top-10 negative comments → Claude → root_cause + action + priority
└────────┬──────────┘
         │  dict[theme → insight]
         ▼
┌───────────────────┐
│     app.py        │  Streamlit: KPIs · trend chart · theme breakdown · drill-down · exec summary
└───────────────────┘
```

### Classification output schema

Each comment is classified with:

| Field | Type | Description |
|---|---|---|
| `sentiment` | `positive` \| `negative` \| `neutral` | Overall sentiment |
| `sentiment_score` | float −1 → 1 | Continuous score |
| `primary_theme` | enum (8 themes) | Dominant topic |
| `secondary_theme` | enum \| null | Secondary topic if clear |
| `is_actionable` | bool | Implies a concrete improvement |
| `urgency` | `high` \| `medium` \| `low` | Response urgency |
| `suggested_department` | enum (5 depts) | Routing target |

### Themes

`wait_times` · `mobile_app` · `loan_process` · `staff_friendliness` · `rates` · `atm_availability` · `communication` · `account_fees` · `other`

### Department routing

| Department | Handles |
|---|---|
| `retail` | Branch staff, teller interactions, ATM hardware |
| `digital` | Mobile app, online banking, website |
| `lending` | Loans, mortgages, credit cards, product rates |
| `marketing` | Fees, promotions, member communications |
| `operations` | Wait times, phone service, call-center processes |

---

## Quick start

```bash
# 1. Clone and install
git clone <repo-url>
cd cu_feedback_analyzer
pip install -r requirements.txt

# 2. Add your Anthropic API key
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...

# 3. Run
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501), click **Use sample data**, then **▶ Run Analysis**.

---

## Project structure

```
cu_feedback_analyzer/
├── app.py                   # Streamlit dashboard
├── engine/
│   ├── ingestor.py          # CSV loading and cleaning
│   ├── classifier.py        # Claude batch classification
│   └── summarizer.py        # Claude per-theme insights
├── data/
│   └── sample_feedback.csv  # 200-row synthetic NPS dataset
├── requirements.txt
└── .env.example
```

---

## CSV format

Your input CSV must have these columns (extra columns are ignored):

| Column | Type | Notes |
|---|---|---|
| `response_date` | date string | Any format pandas can parse |
| `nps_score` | integer 0–10 | Rows outside range are dropped |
| `comment` | string | Blank rows are dropped |
| `channel` | string | e.g. `branch`, `mobile`, `phone` |
| `member_segment` | string | e.g. `retail`, `business`, `senior` |
| `member_tenure_years` | float | Optional but preserved |

---

## Cost

Classification uses `claude-haiku-4-5` with prompt caching on the system prompt. For a 200-comment dataset:

- ~10 API calls (batches of 20)
- System prompt cached after the first call (~90% token reduction on prompt)
- Typical cost: **< $0.01** per full run

---

## License

MIT
# Member-Feedback-Analyzer
