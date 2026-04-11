"""Run FinBERT sentiment analysis on cleaned articles and save results."""

from pathlib import Path

import pandas as pd
from src.nlp.sentiment import SentimentAnalyzer, DEFAULT_OUTPUT
from src.nlp.summarizer import get_article_summary

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "spy_articles_clean.csv"

articles = pd.read_csv(CLEAN_PATH)

# Skip articles already scored (match by URL)
if DEFAULT_OUTPUT.exists():
    existing = pd.read_csv(DEFAULT_OUTPUT)
    scored_titles = set(existing["title"].dropna())
    new_articles = articles[~articles["title"].isin(scored_titles)].reset_index(drop=True)
    print(f"Already scored: {len(articles) - len(new_articles)}, new to score: {len(new_articles)}")
else:
    new_articles = articles
    print(f"No existing scores, scoring all {len(new_articles)} articles")

if len(new_articles) == 0:
    print("Nothing new to score.")
else:
    analyzer = SentimentAnalyzer()

    # Fetch full articles and summarize to fit FinBERT's 512 token limit
    print("Fetching and summarizing articles...")
    texts = []
    for i, row in new_articles.iterrows():
        print(f"  [{i+1}/{len(new_articles)}] {row['title'][:60]}...")
        summary = get_article_summary(str(row["url"]), str(row["title"]), str(row.get("description", "")))
        texts.append(summary)

    results = analyzer.analyze_batch(texts)
    results["datetime"] = new_articles["datetime"].values if "datetime" in new_articles.columns else new_articles["date"].values
    results["date"] = new_articles["date"].values
    results["source"] = new_articles["source"].values
    results["tickers"] = new_articles["tickers"].values if "tickers" in new_articles.columns else ""
    results["title"] = new_articles["title"].values

    # Print results
    print("=== New Sentiment Results ===\n")
    for _, row in results.iterrows():
        print(f"  [{row['label']:>8}] {row['numeric_score']:+.3f}  |  {row['text'][:80]}")

    # Append to existing scores
    analyzer.save(results)

    # Show full daily aggregation
    all_scores = pd.read_csv(DEFAULT_OUTPUT)
    print("\n=== Full Daily Aggregation ===")
    daily = analyzer.aggregate_daily(all_scores)
    print(daily.to_string(index=False))
