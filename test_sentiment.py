import pandas as pd
from src.nlp.sentiment import SentimentAnalyzer

articles = pd.read_csv("data/raw/spy_articles.csv")
analyzer = SentimentAnalyzer()

# Analyze article titles
results = analyzer.analyze_batch(articles["title"].tolist())
results["date"] = articles["date"].values
results["source"] = articles["source"].values

# Print results
for _, row in results.iterrows():
    print(f"  [{row['label']:>8}] {row['numeric_score']:+.3f}  |  {row['text'][:80]}")

# Show daily aggregation
print("\n=== Daily Aggregation ===")
print(analyzer.aggregate_daily(results).to_string(index=False))

# Save to data/processed/sentiment_scores.csv
analyzer.save(results)
