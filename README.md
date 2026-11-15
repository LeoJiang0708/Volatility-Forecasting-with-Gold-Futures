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

## Prediction Target

Daily logarithmic returns are calculated as:

$$
r_t = 100 \ln\left(\frac{P_t}{P_{t-1}}\right)
$$

The forward 21-day realized volatility target is:

$$
RV_{t,t+21} = \sqrt{\frac{252}{21}\sum_{j=1}^{21}r_{t+j}^{2}}
$$

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
- **BIAS:** Records the absolute deviations from the actual realized volatility.

### Results

*To be completed following final out-of-sample evaluation.*

| Model | RMSE | MAE | QLIKE | BIAS |
|---|---|---|---|---|
| LSTM | 11.153902 | 7.777844 | 0.617204 | -4.024647 |
| GARCH(1,1) | 10.784047	 | 8.119742	 | 0.421693 | -0.465293 |

### Forecast Visualization

<img width="1617" height="854" alt="GARCH_vs_LSTM_Graph_Comparison" src="https://github.com/user-attachments/assets/0428f7b5-f728-4711-b375-224cc00daf57" />

## Key Findings

- Firstly and most importantly, an intricate neural network model does not necessary beat a traditional time-series model in terms of accuracy. We see that the GARCH(1,1) prototype achieves better results for every error metric compared against LSTM, except for RMSE, and there is the biggest difference in bias.
- From the graph, GARCH(1,1) captures elevated periods of sudden spikes better than LSTM as can be seen at the beginning of 2026. LSTM fluctuates more on average throughout, but fails to capture this spike and instead predicts the opposite direction.
- Recall that we incorporated financial market indicators into the inputs for training the LSTM model. However, it is uncertain whether these features actually improved the forecast or were more or less futile in this attempt. Note that we simply included these features in the dataset. A more robust diagnosis would involve analyzing and modeling different volatility regimes, engineer cross-market interaction features, and capture the volatility of the indicators themselves.
- Every statistical model comes with both advantages and drawbacks. GARCH(1,1) is a traditional time-series model, so it forecasts market values by choosing pre-determined parameters in a rigid parametric model (i.e. a given equation). These models have the advantage of high interpretability and easier engineering process; the disadvantage is that it is less flexible and may fail to model scenarios where financial data exhibit a complex, dynamic, and non-temporal relationship. LSTM, serving as a deep learning learning framework for time-series forecasting, can beat these difficulties. Nevertheless, deep learning approaches often need a vast amount of data for training, and could fail to do its job correctly otherwise, which could be the consequence of overfitting the data (i.e. the model somehow memorized the patterns in the training set and attempts to reproduce it in the out-of-sample forecast). The feature engineering and hyper-parameter tuning phases require more thought and validation as well to avoid similar issues. These essential limitations may have explained the reason LSTM did not outperform GARCH(1,1) in this notebook.

## Installation and Usage

The main analysis is implemented in Python using Jupyter Notebook.

**Core libraries:** NumPy, pandas, TensorFlow/Keras, scikit-learn, ARCH, statsmodels, yfinance, Matplotlib, and Seaborn.

To reproduce the analysis:

1. Clone the GitHub repository.
2. Install dependencies listed in `requirements.txt`
3. Open `Gold Futures Forecast.ipynb` in Jupyter or VS Code.
4. Run the notebook to download market data, construct features, train models, and evaluate forecasts.

Internet access is required for the Yahoo Finance

## Limitations and Future Work

- One key limitation is that we have used past 21-day realized volatility to forecast future 21-day volatility by simply shifting the former one month behind. Therefore, we have somehow used the same data for both input and output. Notice that the GARCH(1,1) forecast seems lag behind the realized volatility line. This is because it sees the same 21-day lagged realized volatility a month after the future 21-day RV, which are the same, and since the model computes a prediction for volatility based on the squared returns, it generates a similar curve to the actual RV and lags about 21 days behind. A better methodology would be to predict intraday volatility patterns, but these data are scarce and require more careful feature engineering to deal with.
- Another limitation is that the grid search algorithm is computationally expensive. Expect a few days for it to figure out the optimal combination of hyper-parameters. But the author believes that the need for grid search can reduced or even discarded if the initial volatility patterns could be analyzed in a more rigorous manner, since the entire purpose here is to improve out-of-sample forecast.

Potential extensions include:

- Evaluating models across high- and low-volatility regimes.
- Incorporating additional macroeconomic and alternative data sources, and analyze their explanatory power more rigorously.
- Performing feature ablation studies to evaluate individual predictors.
- Investigating hybrid GARCH–LSTM architectures.
- Extending the framework to other financial futures markets.

## Disclaimer

This project is for academic research and educational purposes only. The forecasts are not intended to constitute financial or investment advice.
