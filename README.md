# CSCI 4022 — Final Project Showcase

Self-contained Python code extracted from **`TPS_trading/strategies/strategy_nobec.py`** (class NobecStrategy (NOBEC stands for NO BECAUSE)): **KNN confidence** and **GMM (Gaussian mixture / EM) confidence**.

## What is included

| Module | Role |
|--------|------|
| `showcase/features.py` | 7-D feature vector (`gap_abs`, `gap_pct`, `volume_ratio`, …) |
| `showcase/knn_confidence.py` | k-nearest neighbor confidence score |
| `showcase/gmm_confidence.py` | `sklearn.mixture.GaussianMixture` log-likelihood confidence |

## What is not included
- Most of the pipeline code from TPS Trading isn't included due to not wanting to bloat this repo (there is a ton of code related to these tests) but also honestly we did not want to share the entire inner workings of TPS for privacy reasons. I hope that makes sense. 

## Notes
- Generative AI was utilized in the creation of TPS trading, not for any of the concepts or ideas, but the mechanical side of it.

- The data used to run all of these tests is sourced from Alpaca (an API broker-dealer) and it is all minute-bar OHLCV (open,high,low,close,volume) SIP (Securities Information Processor) data, we keep it stored in parquet files to reduce storage costs. We have around 70 symbols downloaded.
- They look like this: 
```
timestamp,open,high,low,close,volume
2026-04-21T09:30:00Z,150.10,150.50,149.90,150.25,12000
2026-04-21T09:31:00Z,150.25,150.40,150.10,150.35,8500
2026-04-21T09:32:00Z,150.35,150.60,150.30,150.55,15200
2026-04-21T09:33:00Z,150.55,150.70,150.45,150.50,9100
2026-04-21T09:34:00Z,150.50,150.55,150.10,150.20,11400
```

- For the record, we are attempting to use KNN and GMM for science. Not because it generates the most profitable trading strategies. Any historical data pattern based methods tend to lean towards unprofitability in our experience. KNN/GMM are much more suited to bollinger band swing trading, or for optimizing paramaters of gap_fade strategies, but unfortunately we didn't have the time to pull something like that off. It would also help if we could optimize the input paramaters of these methods but again, resources didn't permit that.

- What we learned: confidence in a trade should be based on current market signals, not comparing opening breakouts to the past.

- **GMM confidence** uses **Expectation–Maximization** via scikit-learn’s `GaussianMixture`; **direction** “gmm” in the engine is a **lightweight centroid heuristic**, not a second mixture fit.

## High level dataflow of TPS

One backtest job (single test run), from start to finish:

1. A client calls the trading server (for example `POST /backtest-by-nobec-profile` with `strategy_id`, `profile_name`, and optional `confidence_method`). The server creates a row in `job_records` and queues work.

2. The backtest worker process starts. It sets `DATA_DIR`, opens the SQLite DB and DuckDB bridge for parquet, and builds a `StrategyRunner` from the job config.

3. Phase 0: `StrategyRunner.verify_required_data` checks that cached parquet (and related inputs) cover the date window needed for the profile.

4. Phase 1: `run_optimization` runs if the profile enables optimization. Output is per-symbol parameters used in the test window.

5. Phase 2: `run_standard_backtest` runs. The backtest engine resolves test dates and capital, then drives simulation per strategy.

6. For each symbol and each trading day in the test window, the engine uses loaded minute bars and a window simulator. The simulator steps through configured selection times and asks the strategy to evaluate that day.

7. Once trades opportunities are identified, we apply filters from the profile, then computes confidence (equation, KNN, or GMM depending on `confidence_method`). If filters and confidence pass, it may form a trade; results feed into `CleanRoom` for execution logic, fills and exits.

8. The runner aggregates PnL, trade lists, and metrics across strategies. The worker converts that to rows the DB can store.

9. `save_result` writes a `runner_results` row (summary metrics and `config_snapshot`) and associated `trades` rows. The job record is updated with status completed and `result_id` pointing at that run.

10. The API returns job status to the client; the UI or archive browser can read the same tables for display or export.