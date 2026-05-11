import io
from pathlib import Path

import pandas as pd


def load_feedback(filepath_or_bytes) -> pd.DataFrame:
    """Load and clean NPS feedback CSV. Accepts a file path or raw bytes."""
    if isinstance(filepath_or_bytes, (str, Path)):
        df = pd.read_csv(filepath_or_bytes)
    else:
        df = pd.read_csv(io.BytesIO(filepath_or_bytes))

    df["comment"] = df["comment"].astype(str).str.strip()
    df = df[df["comment"].replace("", pd.NA).notna()]
    df = df[df["comment"] != "nan"]

    df["nps_score"] = pd.to_numeric(df["nps_score"], errors="coerce")
    df = df[df["nps_score"].between(0, 10)]
    df["nps_score"] = df["nps_score"].astype(int)

    df["nps_category"] = pd.cut(
        df["nps_score"],
        bins=[-1, 6, 8, 10],
        labels=["detractor", "passive", "promoter"],
    )

    df = df.reset_index(drop=True)
    return df
