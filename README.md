
# Montpellier Apartment Price Prediction

An automated valuation model (AVM) for apartments in Montpellier, France, built on real government transaction data. Give it a street address, apartment surface, and room count — it returns an estimated market price.

## What this project does

Predicts apartment sale prices from a real street address using structural and location features, and investigates *where and why* the model's predictions fail — using domain knowledge of the local real estate market to interpret the errors, not just report them.

## Try it

```python
estimate_price_by_address("Place de la Comedie, Montpellier", surface=45, rooms=2)
```
Returns an estimated price and price/m², automatically converting the address to coordinates via France's official geocoding API — no manual lookup needed.

## Data

- **Source:** [DVF (Demandes de valeurs foncières)](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres-geolocalisees) — France's open government registry of real estate transactions, published by the DGFiP (tax authority).
- **Scope:** Montpellier apartments, 2023–2025.
- **Raw data:** 42,689 transaction rows → **11,331 clean, individual apartment sales** after filtering.

## Data cleaning

- Removed bulk/developer transactions (single sale record spanning dozens or hundreds of apartment rows — not representative of individual buyer pricing).
- Filtered non-market transfers (symbolic €1 sales, inheritance transfers) using a €/m² floor set from local market knowledge, combined with a statistical trim (1st/99th percentile) as a secondary check.
- Full process documented step by step in [`notebooks/01_explore_clean.py`](notebooks/01_explore_clean.py).

## Model

XGBoost regressor trained on: living surface, room count, postal code, latitude/longitude, year of sale.

## Results

| Metric | Value |
|---|---|
| R² | 0.78 |
| MAE | €29,763 |
| RMSE | €47,632 |

## Key finding

The model's largest errors cluster on large apartments (100m²+). Investigation traced part of this to a **geocoding precision limitation**: DVF's rounded coordinates can make two genuinely different streets appear identical to the model. Verified case: two apartments on different streets (Rue de Syracuse and Rue de Chio) shared identical rounded coordinates, despite selling for €160,000 and €425,870. A production system would need house-number-level geocoding to fix this — a concrete, specific limitation rather than a vague "the model isn't perfect."

## Limitations

- DVF does not capture renovation state, floor, orientation, or building condition — all known to meaningfully affect price in this market.
- DVF's surface figure excludes terraces, balconies, and loggias by definition — a large terrace adds real value the model cannot see.
- The model cannot extrapolate price trends beyond the years it was trained on (2023–2025); passing a future year does not project the trend forward.
- Outlier thresholds were computed on the full dataset before the train/test split, a minor form of data leakage — future iteration would derive thresholds from training data only.
- Scope restricted to residential apartments; houses and commercial property excluded (commercial/office types aren't reliably separable in DVF's category schema).

## How to run

```bash
pip install -r requirements.txt
python notebooks/01_explore_clean.py
python notebooks/02_model_baseline.py
```

## Author

Yan Kostsov — [LinkedIn](www.linkedin.com/in/yankostsov) — background in real estate and asset management, transitioning into applied AI/data (AI4CI Master's, Avignon Université).# Montpellier Apartment Price Prediction

An automated valuation model (AVM) for apartments in Montpellier, France, built on real government transaction data. Give it a street address, apartment surface, and room count — it returns an estimated market price.

## What this project does

Predicts apartment sale prices from a real street address using structural and location features, and investigates *where and why* the model's predictions fail — using domain knowledge of the local real estate market to interpret the errors, not just report them.

## Try it

```python
estimate_price_by_address("Place de la Comedie, Montpellier", surface=45, rooms=2)
```
Returns an estimated price and price/m², automatically converting the address to coordinates via France's official geocoding API — no manual lookup needed.

## Data

- **Source:** [DVF (Demandes de valeurs foncières)](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres-geolocalisees) — France's open government registry of real estate transactions, published by the DGFiP (tax authority).
- **Scope:** Montpellier apartments, 2023–2025.
- **Raw data:** 42,689 transaction rows → **11,331 clean, individual apartment sales** after filtering.

## Data cleaning

- Removed bulk/developer transactions (single sale record spanning dozens or hundreds of apartment rows — not representative of individual buyer pricing).
- Filtered non-market transfers (symbolic €1 sales, inheritance transfers) using a €/m² floor set from local market knowledge, combined with a statistical trim (1st/99th percentile) as a secondary check.
- Full process documented step by step in [`notebooks/01_explore_clean.py`](notebooks/01_explore_clean.py).

## Model

XGBoost regressor trained on: living surface, room count, postal code, latitude/longitude, year of sale.

## Results

| Metric | Value |
|---|---|
| R² | 0.78 |
| MAE | €29,763 |
| RMSE | €47,632 |

## Key finding

The model's largest errors cluster on large apartments (100m²+). Investigation traced part of this to a **geocoding precision limitation**: DVF's rounded coordinates can make two genuinely different streets appear identical to the model. Verified case: two apartments on different streets (Rue de Syracuse and Rue de Chio) shared identical rounded coordinates, despite selling for €160,000 and €425,870. A production system would need house-number-level geocoding to fix this — a concrete, specific limitation rather than a vague "the model isn't perfect."

## Limitations

- DVF does not capture renovation state, floor, orientation, or building condition — all known to meaningfully affect price in this market.
- DVF's surface figure excludes terraces, balconies, and loggias by definition — a large terrace adds real value the model cannot see.
- The model cannot extrapolate price trends beyond the years it was trained on (2023–2025); passing a future year does not project the trend forward.
- Outlier thresholds were computed on the full dataset before the train/test split, a minor form of data leakage — future iteration would derive thresholds from training data only.
- Scope restricted to residential apartments; houses and commercial property excluded (commercial/office types aren't reliably separable in DVF's category schema).

## How to run

```bash
pip install -r requirements.txt
python notebooks/01_explore_clean.py
python notebooks/02_model_baseline.py
```

## Author

Yan Kostsov — [LinkedIn](www.linkedin.com/in/yankostsov) — background in real estate and asset management, transitioning into applied AI/data (AI4CI Master's, Avignon Université).