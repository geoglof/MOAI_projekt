# Claude Code Context

## Project
News-driven stock prediction (SPY/TSLA) — academic project, 4-week timeline. See PLAN.md for full details.

## What's been done
- **src/nlp/sentiment.py** — FinBERT sentiment pipeline (SentimentAnalyzer class)
  - `analyze(text)` — scores one headline, returns label + numeric score (-1 to +1)
  - `analyze_batch(texts)` — scores a list, returns DataFrame
  - `aggregate_daily(df)` — groups scores by date into daily averages
  - Test it with: `cd src && python -m nlp.sentiment`

## What's next (Jakub's tasks from PLAN.md)
- Week 1: Test sentiment.py on 10-20 manual examples to verify it works
- Week 2: Run FinBERT on all collected articles → save to data/processed/sentiment_scores.csv
- Week 2: Handle edge cases (non-English text, very long articles)

## Team
- Jakub: NLP pipeline + project coordination (that's you)
- Kacper: Data pipeline + statistical analysis (Granger causality)
- Maja: Data collection + visualizations

## Setup
```bash
pip install -r requirements.txt
# Then test:
cd src && python -m nlp.sentiment
```
