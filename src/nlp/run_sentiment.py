"""Run FinBERT sentiment analysis on cleaned articles and save results."""

from pathlib import Path

import pandas as pd
from src.nlp.sentiment import SentimentAnalyzer
from src.nlp.summarizer import get_article_summary

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "spy_articles_clean.csv"

articles = pd.read_csv(CLEAN_PATH)
analyzer = SentimentAnalyzer()

# Fetch full articles and summarize to fit FinBERT's 512 token limit
# Falls back to title + description if scraping fails
print("Fetching and summarizing articles...")
texts = []
for i, row in articles.iterrows():
    print(f"  [{i+1}/{len(articles)}] {row['title'][:60]}...")
    summary = get_article_summary(str(row["url"]), str(row["title"]), str(row.get("description", "")))
    texts.append(summary)

results = analyzer.analyze_batch(texts)
results["datetime"] = articles["datetime"].values if "datetime" in articles.columns else articles["date"].values
results["date"] = articles["date"].values
results["source"] = articles["source"].values
results["tickers"] = articles["tickers"].values if "tickers" in articles.columns else ""
results["title"] = articles["title"].values

# Print results
print("=== Sentiment Results ===\n")
for _, row in results.iterrows():
    print(f"  [{row['label']:>8}] {row['numeric_score']:+.3f}  |  {row['text'][:80]}")

# Daily aggregation
print("\n=== Daily Aggregation ===")
daily = analyzer.aggregate_daily(results)
print(daily.to_string(index=False))

# Save
analyzer.save(results)
