import json
import re

import anthropic
from tqdm import tqdm

BATCH_SIZE = 20
MODEL = "claude-haiku-4-5"

THEMES = [
    "wait_times", "mobile_app", "loan_process", "staff_friendliness",
    "rates", "atm_availability", "communication", "account_fees", "other",
]

_SYSTEM_PROMPT = """You are a credit union member feedback analyst. Classify NPS survey comments.

For each comment return a JSON object with exactly these keys:
- "sentiment": "positive" | "negative" | "neutral"
- "sentiment_score": float from -1.0 (very negative) to 1.0 (very positive)
- "primary_theme": one of: wait_times, mobile_app, loan_process, staff_friendliness, rates, atm_availability, communication, account_fees, other
- "secondary_theme": another theme from the same list if clearly present, otherwise null
- "is_actionable": true if feedback implies a specific improvement the credit union could make
- "urgency": "high" | "medium" | "low"
- "suggested_department": "retail" | "digital" | "lending" | "marketing" | "operations"

Department routing guide:
- retail: branch staff, teller interactions, ATM hardware issues
- digital: mobile app, online banking, website
- lending: loans, mortgages, credit cards, rates on financial products
- marketing: fees, promotions, member communications, products
- operations: wait times, phone service, general call-center processes"""


def classify_batch(comments: list[str]) -> list[dict]:
    """Classify NPS comments using Claude. Returns one result dict per comment."""
    client = anthropic.Anthropic()
    results: list[dict] = []
    batches = [comments[i : i + BATCH_SIZE] for i in range(0, len(comments), BATCH_SIZE)]

    for batch in tqdm(batches, desc="Classifying", unit="batch"):
        results.extend(_call_claude(client, batch))

    return results


def _call_claude(client: anthropic.Anthropic, batch: list[str]) -> list[dict]:
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(batch))

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
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
                    f"Classify these {len(batch)} member feedback comments.\n"
                    "Return a JSON array — one object per comment, in order.\n"
                    "Output only the JSON array, no extra text.\n\n"
                    f"{numbered}"
                ),
            }
        ],
    )

    raw = response.content[0].text.strip()

    # Strip any prose Claude wraps around the array
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]

    parsed: list[dict] = json.loads(raw)

    if len(parsed) != len(batch):
        raise ValueError(
            f"Claude returned {len(parsed)} results for {len(batch)} comments"
        )

    return parsed
