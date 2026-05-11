import json

import anthropic
import pandas as pd

MODEL = "claude-haiku-4-5"

_SYSTEM_PROMPT = """You are a credit union member experience strategist analyzing NPS feedback.

Given a theme name and the most negative member comments for that theme, return a JSON object with exactly these keys:
- "root_cause": one sentence identifying the underlying systemic problem
- "recommended_action": 2-3 sentences describing a concrete, specific action the credit union should take
- "priority_score": integer 1-10 (10 = most urgent; weight by comment volume and sentiment severity)

Be specific. Ground every claim in the actual comments. No generic platitudes."""


def generate_insights(df: pd.DataFrame) -> dict:
    """Generate AI insights for themes with 5+ classified comments.

    Args:
        df: classified DataFrame — must have columns: comment, primary_theme, sentiment_score

    Returns:
        dict keyed by theme with keys: root_cause, recommended_action, priority_score,
        comment_count, avg_sentiment
    """
    client = anthropic.Anthropic()
    results = {}

    for theme, group in df.groupby("primary_theme"):
        if len(group) < 5:
            continue

        avg_sentiment = float(group["sentiment_score"].mean())
        comment_count = int(len(group))

        top_negative = group.nsmallest(10, "sentiment_score")["comment"].tolist()

        ai = _call_claude(client, theme, top_negative)

        results[theme] = {
            "root_cause": ai.get("root_cause", ""),
            "recommended_action": ai.get("recommended_action", ""),
            "priority_score": int(ai.get("priority_score", 5)),
            "comment_count": comment_count,
            "avg_sentiment": round(avg_sentiment, 3),
        }

    return results


def _call_claude(client: anthropic.Anthropic, theme: str, comments: list[str]) -> dict:
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(comments))

    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=[
            {
                "type": "text",
                "text": _SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": (
                    f"Theme: {theme}\n\n"
                    f"Most negative member comments:\n{numbered}\n\n"
                    "Return only the JSON object, no extra text."
                ),
            }
        ],
    )

    raw = response.content[0].text.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]

    return json.loads(raw)
