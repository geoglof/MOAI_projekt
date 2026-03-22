# MOAI Projekt - News-Driven Stock Prediction

Academic research project investigating whether financial news sentiment can predict stock price movements.

## Research Question

**Does sentiment extracted from financial news predict short-term stock price movements, or do price movements cause the news?**

## Team

- **Jakub** - NLP pipeline, project coordination
- **Kacper** - Data pipeline, statistical analysis
- **Maja** - Data collection, visualization, presentation

## Project Structure

```
MOAI_projekt/
├── data/
│   ├── raw/           # Downloaded raw data (gitignored)
│   └── processed/     # Cleaned, merged datasets
├── notebooks/         # Jupyter exploration notebooks
├── src/
│   ├── data_collection/   # Scripts for fetching prices and news
│   ├── nlp/               # Sentiment analysis pipeline (FinBERT)
│   └── analysis/          # Statistical tests and prediction models
├── visualizations/    # Saved plots and charts
├── report/            # Final report and presentation
└── notes/             # Research notes and reference materials
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Key Methods

- **FinBERT** for financial text sentiment analysis
- **Granger causality test** to determine directionality (news → price or price → news)
- **Lagged correlation analysis** between sentiment and returns
- **Contrarian signal detection** (volume spike + positive news flood = exit signal)
