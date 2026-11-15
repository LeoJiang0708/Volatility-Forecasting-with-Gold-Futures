# Model Theory: LSTM and GARCH for Gold Futures Volatility

> Companion to [`Gold Futures Forecast.ipynb`](Gold%20Futures%20Forecast.ipynb). This page explains the models *as specified in the notebook*, distinguishing what is implemented from improvements to consider. Mathematical expressions use GitHub-compatible `math` code fences.

## 1. Forecasting problem and notation

The objective is to forecast **the annualized realized volatility over the next 21 trading days** of gold futures (`GC=F`), using information observable at the end of day *t*.

Let `P_t` be the gold futures daily close and `r_t` the daily log return in percentage points:

```math
r_t = 100\log\left(\frac{P_t}{P_{t-1}}\right).
```

The target is a *forward-looking* 21-day realized volatility, defined using **future** returns:

```math
RV_{t,t+21} = \sqrt{\frac{252}{21}\sum_{j=1}^{21}r_{t+j}^{2}}.
```

- **21** is the forecast horizon in trading days.
- **252** annualizes daily variance to an approximate trading-year scale.
- Because `r_t` is multiplied by 100, volatility is expressed in **percent per year** (annualized percentage volatility, not a percentage return forecast).
- The notebook calculates this target with `rolling(21).mean().shift(-21)`, which aligns the next 21 returns with the forecast-origin date *t*.

The historical, *trailing* volatility features are different from the target:

```math
RV_{t,L}^{\mathrm{trailing}} = \sqrt{\frac{252}{L}\sum_{j=0}^{L-1} r_{t-j}^{2}}, \qquad L\in\{21,42,63\}.
```

These trailing measures use data already observed through day *t*.

## 2. GARCH(1,1): a conditional-variance model

### 2.1 Core equations

A constant-mean GARCH(1,1) model separates returns into a mean and a shock:

```math
r_t = \mu + \varepsilon_t, \qquad \varepsilon_t = \sigma_t z_t.
```

Its conditional variance evolves as:

```math
\sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2.
```

| Symbol | Interpretation |
|---|---|
| `μ` | Constant conditional mean return |
| `ε_(t−1)^2` | Yesterday's squared unexpected return (new shock) |
| `σ_(t−1)^2` | Yesterday's estimated conditional variance (persistence) |
| `ω` | Baseline variance intercept |
| `α` | Sensitivity to the latest shock (ARCH effect) |
| `β` | Persistence of previously elevated variance (GARCH effect) |

Common restrictions include `ω > 0`, `α ≥ 0`, `β ≥ 0`. The usual sufficient condition for a finite long-run unconditional variance in standard GARCH(1,1) is `α + β < 1`, in which case:

```math
\mathrm{Var}(\varepsilon_t) = \frac{\omega}{1-\alpha-\beta}.
```

The notebook specifies **Student's t innovations**, rather than Gaussian innovations, to allow heavier tails in return shocks. The standardized innovation `z_t` has conditional mean zero and variance one when the t degrees-of-freedom parameter is greater than two.

### 2.2 Mechanism: why GARCH predicts clustering

1. A large absolute unexpected return makes `ε_t²` large.
2. The ARCH term raises the next estimated variance, `σ_(t+1)²`.
3. The GARCH term carries part of that rise into subsequent periods.
4. If the fitted process is mean reverting, the effect decays over time.

GARCH therefore forecasts **time-varying risk**, not the direction of the gold price.

### 2.3 Turning daily variance forecasts into a 21-day forecast

After fitting through date *t*, GARCH produces variance forecasts for horizons `h = 1, …, 21`:

```math
\widehat{\sigma}_{t+h\mid t}^{2} = E_t[\sigma_{t+h}^{2}].
```

The notebook aggregates them as:

```math
\widehat{RV}^{\mathrm{GARCH}}_{t,t+21}
= \sqrt{\frac{252}{21}\sum_{h=1}^{21}\widehat{\sigma}_{t+h\mid t}^{2}}.
```

This corresponds to:

```python
forecast = fitted_result.forecast(horizon=21)
variances = forecast.variance.iloc[-1]
predicted_vol = (252 * variances.mean()) ** 0.5
```

**Technical distinction:** Your realized-volatility target squares the **raw return** `r`, whereas GARCH's `σ²` refers to the variance of the **mean-adjusted innovation**. With a constant nonzero mean `μ`, the conditional second moment of returns is `E_t[r²] = E_t[σ²] + μ²`; the implemented forecast omits this typically small mean-squared contribution. Also, the square root of expected future average variance is **not identical** to the mathematical expectation of future realized volatility (by Jensen's inequality).

### 2.4 Implementation in the notebook

```python
model = arch_model(
    training_data,
    mean="Constant", vol="GARCH", p=1, q=1,
    dist="t", rescale=False,
)
result = model.fit(disp="off")
forecast = result.forecast(horizon=21)
```

The GARCH model is **re-estimated on an expanding window** at successive forecast dates in **2026**. Each forecast uses data through its forecast-origin date, not the subsequently realized target.

## 3. LSTM: a sequence model for nonlinear dynamics

### 3.1 Why recurrent memory may help

An LSTM is a recurrent neural network that processes historical observations sequentially. Unlike the fixed GARCH variance equation, it can learn nonlinear relationships across multiple input features, potentially including interactions among gold returns, trailing volatility, VIX, equities, rates, and oil.

At each historical step *s*, the model receives a feature vector `x_s`, and updates its **cell state** `c_s` and **hidden state** `h_s`.

### 3.2 LSTM cell equations

Here, `σ(·)` is the logistic sigmoid, `tanh` is the hyperbolic tangent, `⊙` means elementwise multiplication, and `W` and `b` are learned weights and biases.

**Forget gate** — what past information to retain:

```math
f_s = \sigma\!\left(W_f [h_{s-1},x_s]+b_f\right).
```

**Input gate and candidate memory** — what new information to write:

```math
i_s = \sigma\!\left(W_i [h_{s-1},x_s]+b_i\right),
\qquad
\widetilde{c}_s = \tanh\!\left(W_c [h_{s-1},x_s]+b_c\right).
```

**Cell-state update** — combine retained memory with newly selected information:

```math
c_s = f_s\odot c_{s-1}+i_s\odot\widetilde{c}_s.
```

**Output gate and hidden state** — select what to expose to the next layer:

```math
o_s = \sigma\!\left(W_o [h_{s-1},x_s]+b_o\right),
\qquad
h_s = o_s\odot\tanh(c_s).
```

The gates are learned from data; they are **not** manually assigned to particular economic indicators. The model's memory can help represent temporal patterns, but does not guarantee that long-range dependencies are learned or improve forecasts.

### 3.3 Sequence-to-one forecast

The notebook uses the past **126 trading days** of selected features to predict one scalar: future 21-day volatility.

```math
\widehat{y}_t = g_{\theta}(x_{t-125},x_{t-124},\ldots,x_t).
```

Here `g_θ` is the complete neural network. It predicts the **next 21-day aggregate volatility directly**; it does *not* recursively forecast 21 individual daily returns.

The selected notebook features number **11**:

- Gold: returns, absolute returns, trailing 21-/42-/63-day realized volatility
- Macroeconomic/market: 2-year Treasury-yield change, S&P 500 return, VIX level and change, 10-year real-yield change, oil return

Dollar-index and silver-return series are constructed but **not included** in the selected feature matrix.

Consequently, each model input has shape `(126, 11)` and the full tensor has shape `(number_of_samples, 126, 11)`.

### 3.4 Architecture and training

The notebook's network is:

```text
126 trading days × 11 features
             │
             ▼
     LSTM(64 units)
       → Dropout
             │
             ▼
     LSTM(32 units)
       → Dropout
             │
             ▼
     LSTM(16 units)
       → Dropout
             │
             ▼
        Dense(1)
             │
             ▼
  Predicted 21-day RV
```

- **Training objective:** mean squared error on *standardized* targets, with L2 kernel penalties included in the model's training loss when `l2 > 0`.
- **Regularization:** dropout between layers and optional L2 weight penalties.
- **Hyperparameter search:** learning rate, dropout, L2 penalty, and batch size, using `GridSearchCV` with `TimeSeriesSplit(n_splits=3, gap=21)` on training sequences.
- **Optimization:** the notebook defines Adam as the model-builder default and separately fits an RMSprop configuration in its later training cell.
- **Output conversion:** predicted standardized volatility is transformed back into annualized percentage units with the target `StandardScaler`.

## 4. How LSTM and GARCH differ

| Aspect | GARCH(1,1) | LSTM |
|---|---|---|
| Model type | Parametric conditional variance equation | Data-driven nonlinear neural network |
| Primary inputs in this notebook | Historical gold returns | 126-day sequences of 11 market and gold features |
| Forecasting mechanism | Recursively predicts future conditional variances | Directly predicts one forward 21-day volatility target |
| Time dependence | Encoded by squared shock and lagged variance | Learned through recurrent states and gates |
| Strength | Parsimonious, interpretable, explicit volatility dynamics | Flexible nonlinear and multivariate relationships |
| Risk | Misspecification, restrictive functional form | Overfitting, sensitive tuning, harder interpretation |

Neither model is automatically superior. A fair comparison uses **the same forecast origins, the same target definition, and strictly out-of-sample predictions**.

## 5. Evaluation and practical cautions

### Forecast error measures

For `n` out-of-sample forecasts of annualized volatility `v_t` and predictions `v̂_t`:

```math
\mathrm{MAE} = \frac1n\sum_{t=1}^{n}|v_t-\widehat v_t|,
\qquad
\mathrm{RMSE}=\sqrt{\frac1n\sum_{t=1}^{n}(v_t-\widehat v_t)^2}.
```

A **variance-based** QLIKE loss uses *realized variance* `v_t²` and *forecast variance* `v̂_t²`:

```math
\mathrm{QLIKE} = \frac1n\sum_{t=1}^{n}
\left(\frac{v_t^2}{\widehat v_t^2}
-\log\!\frac{v_t^2}{\widehat v_t^2}-1\right).
```

This formula requires strictly positive forecast and realized variances (or a documented handling rule for zero values). It should **not** be computed by substituting volatility directly for variance.

### Current notebook details to check before drawing conclusions

1. **Different comparison periods:** LSTM test samples begin in **2025**, but the expanding-window GARCH forecasts are produced for **2026**. Align the same dates before reporting a head-to-head score.
2. **LSTM inference variable:** the late notebook cell assigns `history = create_model(...).fit(...)` but then calls `model.predict(X_test)`. The name `model` was previously used for an ARCH/GARCH model. Save the trained LSTM object explicitly and predict with it.
3. **EWMA is not yet a like-for-like 21-day forecast:** the current EWMA code computes a trailing volatility estimate; it is not converted into a forward 21-day forecast before plotting alongside GARCH.
4. **Data timing:** features must be available at the forecast origin; check market closing times, publication dates, and any forward-filling across different market calendars.
5. **Model selection vs. final testing:** scale on training data only, tune using training/validation periods, and reserve the final test period for reporting rather than iterative feature selection.

## 6. Further reading

- Bollerslev, T. (1986). *Generalized Autoregressive Conditional Heteroskedasticity*. Journal of Econometrics.
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
- Keras documentation: [LSTM layer](https://keras.io/api/layers/recurrent_layers/lstm/).
- `arch` documentation: [Univariate Volatility Modeling](https://arch.readthedocs.io/en/latest/univariate/univariate_volatility_modeling.html) and [Forecasting](https://arch.readthedocs.io/en/latest/univariate/forecasting.html).

---

**Scope:** This document describes the methods as written in the uploaded notebook. It is not a claim that every model run, comparison, or result has already been validated or reproduced.
