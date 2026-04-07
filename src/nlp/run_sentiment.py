"""Run FinBERT sentiment analysis on cleaned articles and save results."""

from pathlib import Path

import pandas as pd
from src.nlp.sentiment import SentimentAnalyzer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "spy_articles_clean.csv"

articles = pd.read_csv(CLEAN_PATH)
analyzer = SentimentAnalyzer()

# Combine title + description for maximum context (FinBERT truncates at 512 tokens)
texts = (articles["title"].fillna("") + ". " + articles["description"].fillna("")).tolist()

results = analyzer.analyze_batch(texts)
results["date"] = articles["date"].values
results["source"] = articles["source"].values

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
