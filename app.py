from pathlib import Path

import altair as alt
import anthropic
import pandas as pd
import streamlit as st

from engine.classifier import classify_batch
from engine.ingestor import load_feedback
from engine.summarizer import generate_insights

st.set_page_config(
    page_title="Member Feedback Analyzer",
    page_icon="🏦",
    layout="wide",
)

SAMPLE_CSV = Path(__file__).parent / "data" / "sample_feedback.csv"

# ── Helpers ──────────────────────────────────────────────────────────────────

def _nps(df: pd.DataFrame) -> int:
    total = len(df)
    if total == 0:
        return 0
    p = (df["nps_category"] == "promoter").sum()
    d = (df["nps_category"] == "detractor").sum()
    return round((p / total - d / total) * 100)


def _top_complaint(df: pd.DataFrame) -> str:
    if "primary_theme" not in df.columns or df.empty:
        return "—"
    worst = (
        df[df["sentiment_score"] < 0]
        .groupby("primary_theme")["sentiment_score"]
        .mean()
    )
    if worst.empty:
        return "—"
    return worst.idxmin().replace("_", " ").title()


def _run_exec_summary(insights: dict) -> str:
    top3 = sorted(insights.items(), key=lambda x: x[1]["priority_score"], reverse=True)[:3]
    brief_lines = []
    for theme, data in top3:
        brief_lines.append(
            f"Theme: {theme.replace('_', ' ').title()}\n"
            f"Root cause: {data['root_cause']}\n"
            f"Recommended action: {data['recommended_action']}\n"
            f"Priority: {data['priority_score']}/10 | "
            f"Comments: {data['comment_count']} | "
            f"Avg sentiment: {data['avg_sentiment']:.2f}"
        )
    brief = "\n\n".join(brief_lines)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    "Write a one-page executive summary for credit union leadership on the top 3 "
                    "member experience issues below. Structure: opening paragraph with overall "
                    "member sentiment context; one section per issue (bold heading + 2-3 sentences "
                    "on the problem and recommended action); closing paragraph with overall "
                    "recommendation. Plain business prose — no bullet points.\n\n"
                    f"{brief}"
                ),
            }
        ],
    )
    return response.content[0].text


# ── Sidebar: data loading ─────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏦 Member Feedback")
    st.markdown("---")

    uploaded = st.file_uploader("Upload NPS CSV", type="csv")

    use_sample = st.button("Use sample data")

    if use_sample:
        st.session_state["df_raw"] = load_feedback(SAMPLE_CSV)
        for key in ("df_classified", "insights", "exec_summary"):
            st.session_state.pop(key, None)

    if uploaded is not None:
        raw_bytes = uploaded.read()
        new_df = load_feedback(raw_bytes)
        prev = st.session_state.get("_upload_name")
        if prev != uploaded.name:
            st.session_state["df_raw"] = new_df
            st.session_state["_upload_name"] = uploaded.name
            for key in ("df_classified", "insights", "exec_summary"):
                st.session_state.pop(key, None)

    if "df_raw" not in st.session_state:
        st.info("Upload a CSV or click **Use sample data** to begin.")
        st.stop()

    df_raw: pd.DataFrame = st.session_state["df_raw"]

    # ── Sidebar: filters ──────────────────────────────────────────────────────

    st.markdown("### Filters")

    df_raw["response_date"] = pd.to_datetime(df_raw["response_date"], errors="coerce")
    min_d = df_raw["response_date"].min().date()
    max_d = df_raw["response_date"].max().date()

    date_sel = st.date_input("Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    start_d = date_sel[0] if isinstance(date_sel, (tuple, list)) and len(date_sel) > 0 else min_d
    end_d = date_sel[1] if isinstance(date_sel, (tuple, list)) and len(date_sel) > 1 else max_d

    channels = ["All"] + sorted(df_raw["channel"].dropna().unique().tolist())
    channel_sel = st.selectbox("Channel", channels)

    segments = ["All"] + sorted(df_raw["member_segment"].dropna().unique().tolist())
    segment_sel = st.selectbox("Member segment", segments)

    # ── Sidebar: run analysis ─────────────────────────────────────────────────

    st.markdown("---")
    run_btn = st.button("▶ Run Analysis", type="primary", use_container_width=True)

if run_btn:
    with st.spinner("Classifying comments with Claude…"):
        comments = df_raw["comment"].tolist()
        classifications = classify_batch(comments)
        clf_df = pd.DataFrame(classifications)
        df_classified = pd.concat(
            [df_raw.reset_index(drop=True), clf_df.reset_index(drop=True)], axis=1
        )
        st.session_state["df_classified"] = df_classified
    with st.spinner("Generating theme insights…"):
        st.session_state["insights"] = generate_insights(df_classified)
    st.session_state.pop("exec_summary", None)
    st.rerun()

if "df_classified" not in st.session_state:
    st.info("Click **▶ Run Analysis** in the sidebar to classify comments and generate insights.")
    st.stop()

df_classified: pd.DataFrame = st.session_state["df_classified"].copy()
insights: dict = st.session_state.get("insights", {})

# ── Apply filters ─────────────────────────────────────────────────────────────

mask = (
    (df_classified["response_date"].dt.date >= start_d)
    & (df_classified["response_date"].dt.date <= end_d)
)
if channel_sel != "All":
    mask &= df_classified["channel"] == channel_sel
if segment_sel != "All":
    mask &= df_classified["member_segment"] == segment_sel

df = df_classified[mask].copy()

if df.empty:
    st.warning("No data matches the current filters.")
    st.stop()

# ── Section 1: KPIs ───────────────────────────────────────────────────────────

st.header("Member Experience Dashboard")

total = len(df)
detractors = (df["nps_category"] == "detractor").sum()
pct_det = round(detractors / total * 100, 1)
nps_val = _nps(df)
top_complaint = _top_complaint(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("NPS Score", nps_val, help="% Promoters − % Detractors")
c2.metric("Responses", f"{total:,}")
c3.metric("% Detractors", f"{pct_det}%")
c4.metric("Top Complaint", top_complaint)

st.markdown("---")

# ── Section 2: Sentiment Trend ────────────────────────────────────────────────

st.subheader("Sentiment Trend")

date_span = (df["response_date"].max() - df["response_date"].min()).days
period_freq = "W" if date_span <= 90 else "M"
period_label = "Weekly" if period_freq == "W" else "Monthly"

df["_period"] = df["response_date"].dt.to_period(period_freq).dt.start_time
trend = (
    df.groupby("_period")["sentiment_score"]
    .mean()
    .reset_index()
    .rename(columns={"_period": "date", "sentiment_score": "avg_sentiment"})
    .dropna()
)

if len(trend) >= 2:
    line = (
        alt.Chart(trend)
        .mark_line(point=True, color="#1f77b4")
        .encode(
            x=alt.X("date:T", title=f"{period_label} Period"),
            y=alt.Y(
                "avg_sentiment:Q",
                title="Avg Sentiment Score",
                scale=alt.Scale(domain=[-1, 1]),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Period"),
                alt.Tooltip("avg_sentiment:Q", title="Avg Sentiment", format=".2f"),
            ],
        )
        .properties(height=240)
    )
    zero_line = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(strokeDash=[4, 4], color="#aaa")
        .encode(y="y:Q")
    )
    st.altair_chart(line + zero_line, use_container_width=True)
else:
    st.info("Not enough date variance in the filtered range to plot a trend.")

st.markdown("---")

# ── Section 3: Theme Breakdown ────────────────────────────────────────────────

st.subheader("Theme Breakdown")

theme_stats = (
    df.groupby("primary_theme")
    .agg(comment_count=("comment", "count"), avg_sentiment=("sentiment_score", "mean"))
    .reset_index()
)
theme_stats["theme_label"] = (
    theme_stats["primary_theme"].str.replace("_", " ").str.title()
)

bar = (
    alt.Chart(theme_stats)
    .mark_bar()
    .encode(
        x=alt.X("comment_count:Q", title="Comment Count"),
        y=alt.Y("theme_label:N", sort="-x", title=""),
        color=alt.Color(
            "avg_sentiment:Q",
            scale=alt.Scale(scheme="redyellowgreen", domain=[-1, 1]),
            title="Avg Sentiment",
        ),
        tooltip=[
            alt.Tooltip("theme_label:N", title="Theme"),
            alt.Tooltip("comment_count:Q", title="Comments"),
            alt.Tooltip("avg_sentiment:Q", title="Avg Sentiment", format=".2f"),
        ],
    )
    .properties(height=max(200, len(theme_stats) * 35))
)
st.altair_chart(bar, use_container_width=True)

st.markdown("---")

# ── Section 4: Drill-Down ─────────────────────────────────────────────────────

st.subheader("Theme Drill-Down")

available_themes = sorted(df["primary_theme"].unique())

selected_theme = st.selectbox(
    "Select theme",
    available_themes,
    format_func=lambda t: t.replace("_", " ").title(),
)

theme_df = df[df["primary_theme"] == selected_theme].sort_values("sentiment_score")

col_card, col_table = st.columns([1, 2])

with col_card:
    st.markdown(f"**Insight — {selected_theme.replace('_', ' ').title()}**")
    if selected_theme in insights:
        ins = insights[selected_theme]
        m1, m2 = st.columns(2)
        m1.metric("Priority", f"{ins['priority_score']} / 10")
        m2.metric("Avg Sentiment", f"{ins['avg_sentiment']:.2f}")
        st.metric("Comments (full dataset)", ins["comment_count"])
        st.markdown(f"**Root Cause**\n\n{ins['root_cause']}")
        st.markdown(f"**Recommended Action**\n\n{ins['recommended_action']}")
    else:
        st.info("Insight unavailable — fewer than 5 comments for this theme in the full dataset.")

with col_table:
    st.markdown(f"**Comments ({len(theme_df)} in filtered view)**")
    display_cols = [c for c in ["response_date", "nps_score", "sentiment", "sentiment_score", "comment"] if c in theme_df.columns]
    st.dataframe(
        theme_df[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=360,
        column_config={
            "sentiment_score": st.column_config.NumberColumn(format="%.2f"),
            "response_date": st.column_config.DateColumn(label="Date"),
            "nps_score": st.column_config.NumberColumn(label="NPS"),
        },
    )

st.markdown("---")

# ── Executive Summary ─────────────────────────────────────────────────────────

if st.button("📄 Generate Executive Summary", type="primary"):
    if not insights:
        st.warning("No insights available — run analysis first.")
    else:
        with st.spinner("Writing executive summary…"):
            st.session_state["exec_summary"] = _run_exec_summary(insights)

if "exec_summary" in st.session_state:
    st.subheader("Executive Summary")
    st.markdown(st.session_state["exec_summary"])
