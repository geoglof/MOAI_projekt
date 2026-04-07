"""Clean raw news articles: remove duplicates, irrelevant, non-English, empty."""

from pathlib import Path

import pandas as pd
from langdetect import detect, LangDetectException

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Must contain at least one of these — directly about SPY / US stock market
MUST_HAVE = [
    "s&p", "s&p 500", "spy", "dow", "nasdaq", "wall street",
    "stock market", "stock futures", "equity futures",
    "us stocks", "u.s. stocks",
]

# Supportive keywords — article must also feel financial (catches edge cases)
FINANCE_KEYWORDS = [
    "investor", "trading", "rally", "earnings", "fed", "inflation",
    "recession", "portfolio", "treasury", "yield", "economy", "gdp",
    "sector", "futures", "etf", "index", "bull", "bear",
]

# Junk phrases that indicate irrelevant articles
BLACKLIST = [
    "casino", "betting", "golf", "masters 2026", "nba", "nfl", "wnba",
    "disneyland", "disney", "cruise", "librarian", "recipe", "manitowoc",
]


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def is_relevant(title: str, description: str) -> bool:
    """Check if article is about the US stock market, not just any finance topic."""
    combined = f"{title} {description}".lower()

    # Reject blacklisted junk
    if any(bl in combined for bl in BLACKLIST):
        return False

    # Must mention SPY/S&P/US market directly
    has_must = any(kw in combined for kw in MUST_HAVE)
    # OR must have at least 2 finance keywords (catches relevant articles that don't name SPY directly)
    finance_hits = sum(1 for kw in FINANCE_KEYWORDS if kw in combined)

    return has_must or finance_hits >= 2


def clean(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    original = len(df)

    # Drop rows with empty titles
    df = df.dropna(subset=["title"])
    df = df[df["title"].str.strip() != ""]

    # Drop duplicates by title
    df = df.drop_duplicates(subset=["title"])

    # Keep only English articles
    df = df[df["title"].apply(is_english)]

    # Keep only financially relevant articles
    df = df[df.apply(lambda r: is_relevant(str(r["title"]), str(r["description"])), axis=1)]

    df = df.reset_index(drop=True)

    print(f"Cleaning: {original} → {len(df)} articles")
    print(f"  Removed {original - len(df)} irrelevant/duplicate/non-English rows")

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    return df


if __name__ == "__main__":
    clean(
        RAW_DIR / "spy_newsapi_articles.csv",
        PROCESSED_DIR / "spy_articles_clean.csv",
    )
