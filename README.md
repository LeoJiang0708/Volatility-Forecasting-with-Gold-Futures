# Gold Futures Volatility Forecasting: LSTM vs. GARCH

## Project Overview

This project investigates the effectiveness of deep learning and traditional econometric models in forecasting gold futures volatility.

The primary objective is to predict **21-trading-day forward annualized realized volatility** and evaluate whether Long Short-Term Memory (LSTM) neural networks can improve forecasting performance relative to the GARCH(1,1) model.

The project explores how historical price movements, volatility clustering, and market indicators contribute to volatility predictability.

## Data and Feature Engineering

**Primary asset:** Gold Futures (GC=F from yfinance)

**Data period:** January 2006 – August 2026

**Sources:** Yahoo Finance

The dataset incorporates:

- **Price-derived features:** Daily logarithmic, absolute, and squared returns, and rolling 21-, 42-, and 63-day realized volatility.
- **Market indicators:** USD index, S&P 500 returns, VIX levels and changes, and crude oil and silver returns.

### Prediction Target

Daily logarithmic returns are calculated as:

\(r_t=100\ln(P_t/P_{t-1})\)

The forward 21-day realized volatility target is:

\[
RV_{t,t+21}=\sqrt{\frac{252}{21}\sum_{j=1}^{21}r_{t+j}^{2}}
\]

The target is expressed as annualized volatility in percentage terms.

## Methodology

### 1. GARCH(1,1)

A GARCH(1,1) model serves as the primary econometric benchmark.

- Constant conditional mean
- Student's t-distributed innovations
- Expanding-window walk-forward estimation
- 21-step-ahead conditional variance forecasts aggregated into annualized volatility

### 2. Long Short-Term Memory (LSTM)

A multivariate LSTM network is developed to model nonlinear relationships and temporal dependence in financial time series.

**Architecture:**
- Employed with Keras
- Maximum of three stacked LSTM layers with 64, 32, and 16 units
- Dropout and L2 regularization to constrain overfitting
- Dense output layer for final volatility prediction
- 84-trading-day (i.e. 4 months) input sequences

**Training and tuning:**
- Standardization using training data only
- Hyperparameter grid search over architectures, learning rates, dropouts, L2 penalties, optimizers, and batch sizes
- Time-series cross-validation
- Out-of-sample testing with the best model

## Experimental Design

The GARCH model divides the data chronologically, and uses expanding-window forecast:

| Split | Period |
|---|---|
| Training | 2006–2022 |  
| Testing | 2023–2026 |

(The training set extends into the test set by incorporating data up to the date the forecast is made)

The LSTM model adds a cross-validation phase to help improve out-of-sample accuracy (fixed windows, unlike that for GARCH):

| Split | Period |
|---|---|
| Training | 2006–2022 |
| Validation | 2023–2024 |
| Testing | 2025–2026 |

Forward-target endpoint checks are used at the training and validation boundaries to reduce look-ahead leakage.

## Model Evaluation

Forecast accuracy will be evaluated using:

- **RMSE:** Penalizes larger forecasting errors.
- **MAE:** Measures average absolute forecasting error.
- **QLIKE:** Evaluates conditional variance forecast accuracy.
- - **BIAS:** Records the absolute deviations from the actual realized volatility.

### Results

*To be completed following final out-of-sample evaluation.*

| Model | RMSE | MAE | QLIKE |
|---|---|---|---|
| LSTM | TBD | TBD | TBD |
| GARCH(1,1) | TBD | TBD | TBD |
| EWMA benchmark | TBD | TBD | TBD |

### Forecast Visualization

*Insert finalized out-of-sample forecast plots here.*

## Key Findings

*To be completed following final evaluation.*

This section will summarize:

- The relative accuracy of LSTM and GARCH forecasts.
- Forecast performance during periods of elevated or changing volatility.
- The value of including macroeconomic and financial market indicators.
- The strengths and weaknesses of econometric and deep learning approaches.

## Installation and Usage

The main analysis is implemented in Python using Jupyter Notebook.

**Core libraries:** NumPy, pandas, TensorFlow/Keras, scikit-learn, ARCH, statsmodels, yfinance, Matplotlib, and Seaborn.

To reproduce the analysis:

1. Clone the GitHub repository.
2. Install dependencies listed in `requirements.txt` (once added).
3. Open `Gold Futures Forecast.ipynb` in Jupyter or VS Code.
4. Run the notebook to download market data, construct features, train models, and evaluate forecasts.

Internet access is required for the Yahoo Finance and FRED data downloads.

## Limitations and Future Work

Potential extensions include:

- Evaluating models across high- and low-volatility regimes.
- Incorporating additional macroeconomic and alternative data sources.
- Performing feature ablation studies to evaluate individual predictors.
- Investigating hybrid GARCH–LSTM architectures.
- Extending the framework to other financial futures markets.

## Disclaimer

This project is for academic research and educational purposes only. The forecasts are not intended to constitute financial or investment advice.
