"""Clean raw news articles: remove duplicates, irrelevant, non-English, empty."""

from pathlib import Path

import pandas as pd
from langdetect import detect, LangDetectException

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Keywords that a relevant SPY/S&P 500 article should contain (in title or description)
FINANCE_KEYWORDS = [
    "s&p", "spy", "stock", "market", "investor", "trading", "wall street",
    "index", "etf", "rally", "bull", "bear", "dow", "nasdaq", "fed",
    "inflation", "recession", "earnings", "hedge", "portfolio", "equity",
    "treasury", "bond", "yield", "oil", "economy", "gdp", "rate",
    "sector", "fund", "dividend", "futures", "options", "crypto", "bitcoin",
]


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def is_relevant(title: str, description: str) -> bool:
    """Check if article is financially relevant based on keywords."""
    combined = f"{title} {description}".lower()
    return any(kw in combined for kw in FINANCE_KEYWORDS)


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
