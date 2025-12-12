# soccer-predictor
Top 5 League + UEFA (UCL/UEL/UECL) match predictor using FBref schedules.

## What this does
- Scrapes **Scores & Fixtures** tables from `fbref.com` for:
  - Premier League, La Liga, Serie A, Bundesliga, Ligue 1
  - Champions League, Europa League, Europa Conference League
- Builds a match dataset and trains a **multiclass** ML model to predict **H/D/A**.
- Generates probabilities for upcoming fixtures (rows without scores yet).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
Fetch (cached + rate-limited):
```bash
python3 scripts/run_soccer_predictor.py fetch --max-seasons 6
```

If FBref returns **403 (Cloudflare challenge)** in your environment, use the browser fetcher:
```bash
pip install -r requirements-browser.txt
python3 -m playwright install chromium
python3 scripts/run_soccer_predictor.py fetch --use-playwright --max-seasons 6
```

Train:
```bash
python3 scripts/run_soccer_predictor.py train --data-csv data/matches_raw.csv --model-path models/outcome_model.joblib
```

Predict upcoming:
```bash
python3 scripts/run_soccer_predictor.py predict --data-csv data/matches_raw.csv --model-path models/outcome_model.joblib --out-csv data/predictions.csv
```

## Optional: Quantum ML (experimental)
This repo includes a tiny QML demo (not production training).

```bash
pip install -r requirements-quantum.txt
python3 scripts/run_soccer_predictor.py train --quantum
```

## Notes
- Please scrape responsibly (this project caches HTML and sleeps between requests).
- FBref pages sometimes embed tables in HTML comments; the scraper handles that.
