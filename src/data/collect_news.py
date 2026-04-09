"""Collect financial news articles from NewsAPI and save to data/raw/."""

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from newsapi import NewsApiClient

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def collect_articles(query: str, days_back: int = 30) -> pd.DataFrame:
    """Fetch articles from NewsAPI for a given query."""
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise ValueError("NEWSAPI_KEY not found in .env")

    newsapi = NewsApiClient(api_key=api_key)

    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")

    response = newsapi.get_everything(
        q=query,
        from_param=from_date,
        to=to_date,
        language="en",
        sort_by="publishedAt",
        page_size=100,
    )

    articles = response.get("articles", [])
    print(f"Fetched {len(articles)} articles for '{query}'")

    rows = []
    for a in articles:
        published = a["publishedAt"]  # full ISO timestamp e.g. 2026-04-06T14:21:00Z
        rows.append({
            "datetime": published,
            "date": published[:10],
            "title": a["title"],
            "description": a.get("description", ""),
            "source": a["source"]["name"],
            "url": a["url"],
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    new_df = collect_articles("SPY OR S&P 500")
    output = RAW_DIR / "spy_newsapi_articles.csv"

    # Append to existing file, dedupe by URL
    if output.exists():
        existing = pd.read_csv(output)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)
        added = len(combined) - len(existing)
        combined.to_csv(output, index=False)
        print(f"Added {added} new articles (total: {len(combined)}) to {output}")
    else:
        new_df.to_csv(output, index=False)
        print(f"Saved {len(new_df)} articles to {output}")
