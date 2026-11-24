# TriageLens

A Streamlit dashboard that shows when an intent classifier accepts a prediction or sends it for review.

[**Try the live demo**](https://triagelens-portfolio-cudxhgqfsks5j2gjabkf2u.streamlit.app/)

**Held-out result:** The seven-signal policy reduced modeled cost by 8.75% compared with the confidence-only baseline under the study’s cost assumptions.

![TriageLens dashboard analyzing a query](docs/live-prediction.png)

## What it does

- Inspects the predicted intent and seven signals used to score prediction reliability.
- Compares a confidence-only policy with a learned trust policy on in-scope and out-of-scope queries.
- Visualizes the trade-off between accepting more predictions and making more errors.

## Held-out result

In the local reproduction, the seven-signal policy reduced modeled expected cost from 0.728 to 0.664 (8.75%) across 4,500 known-intent and 1,000 out-of-scope test queries. The calculation assumes 18% out-of-scope prevalence and a 10:1 cost ratio for wrong automation versus review.

## How it works

Query → TF-IDF and LSA representations → intent classifiers → reliability signals → accept or review.

- [Research notebook](notebooks/research.ipynb) contains saved charts and metrics.
- [Dashboard](app.py) is the interface shown above. The original Streamlit entry point is [triagelens_app.py](triagelens_app.py).
- [Python package](triagelens/) implements preprocessing, inference, and evaluation.

## Run locally

From the repository folder, use Python 3.12 and PowerShell.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Live inference requires trusted exported files in `triagelens_models/` and `triagelens_results/experiment_config.json`. Keep them beside `app.py` or set `TRIAGELENS_PROJECT_DIR` to the folder containing them. Some dashboard views also require the original `evidence/` exports. The deployment copy includes the trained model files, frozen configuration, and saved evidence. The raw dataset is optional and is needed to recompute the held-out evaluation.
