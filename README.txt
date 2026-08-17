================================================================================
README - CatBoost Financial Model: Predictive Directional Trading Strategy
================================================================================

PROJECT OVERVIEW
================
This notebook implements an end-to-end machine learning pipeline for predicting
short-term stock price direction (5-day forward returns) using a CatBoost
classifier. It constructs a diversified portfolio from highly liquid S&P 500
stocks and evaluates the strategy using a walk-forward, cross-sectional
approach.

The goal is to generate directional signals that can be used to construct a
long-short portfolio, outperforming a naive baseline. The model incorporates
extensive technical indicators, cross-sectional features, and commodity data.

NOTE: Originally designed to use the Alpaca API for live/paper trading. Due to
technical difficulties, the notebook uses historical data from Yahoo Finance
(yfinance) instead.

KEY FEATURES
============
- Portfolio selection based on liquidity (Average Dollar Volume) and sector
  diversification (55 stocks across 11 GICS sectors).
- Rich feature engineering including:
    * Technical indicators (RSI, MACD, Bollinger Bands, ADX, OBV, etc.)
    * Multi-window volatility, momentum, and trend regime features
    * Cross-sectional features (relative strength, market breadth, volatility
      percentile)
    * Global macro data from commodity prices (merged via as-of join)
- Walk-forward validation with periodic retraining (every 20 trading days).
- CatBoost classifier with early stopping and GPU-friendly parameters.
- Comprehensive evaluation including pooled accuracy, AUC, long-short portfolio
  backtest, and feature importance analysis.

DEPENDENCIES & ENVIRONMENT
==========================
All required packages are installed in the notebook using `!pip install`:

Core Libraries:
- Python 3.x
- pandas, numpy
- matplotlib
- scikit-learn
- statsmodels

Financial Data & Indicators:
- yfinance (data download - used as fallback for Alpaca)
- alpaca-py (installed, but not used due to technical issues)
- pandas-ta (technical indicator generation)

Machine Learning:
- catboost (CatBoostClassifier)
- scipy (for hierarchical clustering)

Data Files:
- The notebook expects:
    - sp500.parquet (pre-downloaded S&P 500 daily data from yfinance)
    - CMO-Historical-Data-Monthly.csv (commodity data)

DATA ACQUISITION & PREPARATION
==============================
1. Universe Selection:
   - Downloads S&P 500 constituent list from Wikipedia.
   - Filters stocks by liquidity (Average Dollar Volume over the entire period).
   - Selects the 5 most liquid stocks per GICS sector, resulting in a final
     universe of 55 stocks.

2. Data Source:
   - The notebook reads pre-downloaded OHLCV data from `sp500.parquet`.
   - The data covers the period from 2021-01-01 to 2026-07-01.
   - Commodity data is read from `CMO-Historical-Data-Monthly.csv`.

3. Feature Engineering:
   - A comprehensive set of features is generated for each stock separately:
     * Price-based indicators (returns over various windows).
     * Technical indicators (RSI, MACD, Bollinger Bands, ADX, OBV).
     * Regime-aware features (volatility ratios, trend strength, RSI on
       multiple windows).
     * Cross-sectional features (relative strength, market breadth,
       volatility percentile).
   - Commodity data is merged using a forward-fill and as-of-join to align
     monthly macro data with daily stock data.
   - All features are normalized using cross-sectional Z-scores (per date,
     across stocks).

LABEL DEFINITION
================
- Target: Binary label indicating whether a stock will outperform the median
  stock in the universe over the next 5 trading days.
- Calculation:
    1. Compute the 5-day forward return for each stock.
    2. Rank stocks by forward return within each date.
    3. Label = 1 if the stock is in the top 50% (above median rank).

MODELING PIPELINE
=================
1. Time Series Split (Walk-Forward Validation):
   - Initial training period: 500 days.
   - Retrain every 20 trading days.
   - Each training set is purged of any data that overlaps with the forward
     returns used in the test set (to avoid data leakage).
   - 20% of the training data is held out for early stopping validation.

2. Model: CatBoostClassifier
   - Objective: Logloss (binary classification).
   - Early stopping with patience of 20 rounds.
   - Shallow trees (depth=4) and regularization (l2_leaf_reg, random_strength)
     to prevent overfitting.
   - Subsampling (80% of data, 70% of features per level) for speed and
     robustness.

EVALUATION
==========
The model is evaluated on each test period. Pooled results across all periods
are reported:

1. **Pooled Accuracy & AUC**: Overall out-of-sample performance.
2. **Local Baseline**: Accuracy of always predicting the majority class for
   that period.
3. **Long-Short Portfolio**:
   - Each day, stocks are ranked by the model's predicted probability.
   - A long position is taken in the top quintile, and a short position in the
     bottom quintile.
   - Returns of the long-short portfolio (mean daily spread, Sharpe ratio,
     win rate) are reported.
4. **Feature Importance**: Both CatBoost built-in importance and permutation
   importance are computed and visualized.

RESULTS (FROM SAMPLE OUTPUT)
=============================
- Pooled Out-of-Sample Accuracy: ~0.518 (baseline: ~0.506)
- Pooled Out-of-Sample AUC: ~0.534
- Long-Short Sharpe (Daily): ~0.041 (Annualized: ~0.650)
- Feature Importance: The most important features are technical indicators
  (volatility, ADX, MACD, returns, price vs. moving averages). Commodity
  features showed negligible importance in this implementation.

FILES
=====
- Catboost_Financial_Model.ipynb : The main Jupyter Notebook.
- sp500.parquet : Pre-downloaded OHLCV data for all S&P 500 stocks.
- CMO-Historical-Data-Monthly.csv : Monthly commodity price data.
- README.txt : This file.

POTENTIAL IMPROVEMENTS / FUTURE WORK
====================================
1. **Feature Engineering**:
   - Investigate non-linear transformations or interaction features.
   - Include sector/industry-specific features.
2. **Model Selection**:
   - Experiment with other tree-based models (XGBoost, LightGBM) or deep
     learning.
   - Use a more extensive hyperparameter tuning strategy (e.g., Optuna,
     grid search).
3. **Portfolio Construction**:
   - Incorporate risk management (position sizing, stop-losses).
   - Use a transaction cost model to make the backtest more realistic.
4. **Data**:
   - Incorporate alternative data (sentiment, earnings, economic indicators).
   - Use intraday data for more granular signals.
5. **Live Trading**:
   - Re-integrate Alpaca API for live/paper trading.
   - Build a robust production pipeline for daily retraining and signal
     generation.
