"""
FinBERT sentiment analysis pipeline for financial news.

Usage:
    from nlp.sentiment import SentimentAnalyzer

    analyzer = SentimentAnalyzer()
    score = analyzer.analyze("Tesla stock surges after strong Q3 earnings")
    daily = analyzer.analyze_batch(list_of_headlines)
"""

from pathlib import Path

import pandas as pd
from transformers import pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "sentiment_scores.csv"


class SentimentAnalyzer:
    """Wrapper around FinBERT for financial sentiment scoring."""

    def __init__(self, device=-1):
        """
        Args:
            device: -1 for CPU, 0 for GPU. Use 0 if running on Colab with GPU.
        """
        self.pipe = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=device,
            truncation=True,
        )

    def analyze(self, text: str) -> dict:
        """Analyze a single text and return sentiment dict.

        Returns:
            dict with keys: label, score, numeric_score
            numeric_score: +1 (positive), 0 (neutral), -1 (negative), weighted by confidence.
        """
        if not text or not text.strip():
            return {"label": "neutral", "score": 0.0, "numeric_score": 0.0}

        result = self.pipe(text)[0]
        label = result["label"]
        confidence = result["score"]

        label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        numeric_score = label_map[label] * confidence

        return {
            "label": label,
            "score": confidence,
            "numeric_score": numeric_score,
        }

    def analyze_batch(self, texts: list[str]) -> pd.DataFrame:
        """Analyze a list of texts and return a DataFrame with sentiment scores.

        Returns:
            DataFrame with columns: text, label, score, numeric_score
        """
        results = [self.analyze(t) for t in texts]
        df = pd.DataFrame(results)
        df.insert(0, "text", texts)
        return df

    def save(self, df: pd.DataFrame, path: Path = DEFAULT_OUTPUT) -> None:
        """Save sentiment results to CSV. Appends if the file already exists."""
        header = not path.exists()
        df.to_csv(path, mode="a", header=header, index=False)
        print(f"Saved {len(df)} rows to {path}")

    def aggregate_daily(self, df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
        """Aggregate article-level sentiments into daily scores.

        Expects a DataFrame with at least 'date' and 'numeric_score' columns.

        Returns:
            DataFrame with columns: date, avg_sentiment, sentiment_std, num_articles
        """
        daily = (
            df.groupby(date_col)["numeric_score"]
            .agg(
                avg_sentiment="mean",
                sentiment_std="std",
                num_articles="count",
            )
            .reset_index()
        )
        daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)
        return daily


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()

    test_headlines = [
        "Fed raises interest rates by 0.5%, markets tumble",
        "Apple reports record quarterly revenue, stock jumps 5%",
        "Unemployment remains steady at 3.7%",
        "S&P 500 closes at all-time high on strong earnings season",
        "Oil prices crash amid global demand concerns",
    ]

    print("=== Single analysis ===")
    result = analyzer.analyze(test_headlines[0])
    print(f"{result['label']:>10} (confidence: {result['score']:.2f}, numeric: {result['numeric_score']:+.2f})")
    print(f"  -> {test_headlines[0]}")

    print("\n=== Batch analysis ===")
    df = analyzer.analyze_batch(test_headlines)
    print(df.to_string(index=False))

    print(f"\n=== Summary ===")
    print(f"Mean sentiment: {df['numeric_score'].mean():+.3f}")
    print(f"Articles analyzed: {len(df)}")

    analyzer.save(df)