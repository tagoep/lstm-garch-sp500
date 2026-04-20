# Hybrid LSTM-GARCH Framework for S&P 500 Volatility Forecasting

> **Note:** This repository contains the code and data pipeline for a research project currently being prepared for journal submission. Detailed numerical results and methodology are withheld pending publication. Please contact the author for collaboration inquiries.

---

## What This Project Is About — Plain English

Every day the stock market moves. Sometimes it barely moves at all. Other times it swings wildly, rising or falling several percent in a single session. The question this project answers is not *which direction* the market will move tomorrow. That is nearly impossible to predict consistently. The question is: **how wild will tomorrow be?**

That wildness, the magnitude of price swings, is called **volatility**. And forecasting it accurately is one of the most consequential problems in quantitative finance.

Here is why it matters:

- **Insurance companies** use volatility forecasts to price financial guarantee products. When you buy an annuity that promises a minimum return no matter what the market does, the insurer is taking on risk. How much they charge you depends on how volatile they expect the market to be.
- **Banks** use volatility to calculate how much money they could lose on a bad day, a number regulators require them to hold in reserve. Underestimate volatility, underestimate risk, hold too little capital. This was a contributing factor to the 2008 financial crisis.
- **Portfolio managers** use volatility forecasts to decide how much protective hedging to buy. A more accurate forecast means cheaper, better-targeted protection.
- **Options traders** price contracts using expected volatility. A better forecast is a genuine market edge.

This project builds and compares three volatility forecasting approaches, a classical statistical model (GARCH), a deep learning model (LSTM), and a hybrid of both, on 31 years of daily S&P 500 data.

**The headline finding: the LSTM neural network substantially outperformed the classical GARCH baseline across all volatility regimes and market conditions, including the COVID-19 market crash. Full results are forthcoming in a journal paper.**

---

## The Two Models — What They Are and Why They Are Different

### Model 1: GARCH(1,1) — The Classical Approach

GARCH stands for Generalized Autoregressive Conditional Heteroskedasticity. The name is intimidating but the idea is simple: **yesterday's volatility predicts today's volatility**.

Markets have a well-documented property called volatility clustering, calm periods tend to be followed by more calm, and turbulent periods tend to be followed by more turbulence. GARCH captures this mathematically with a small number of interpretable parameters representing the baseline variance level, the reaction to market shocks, and the persistence of volatility over time.

GARCH is fast, interpretable, and backed by decades of academic and industry validation. It is the industry standard for financial volatility modeling. We chose GARCH(1,1), specifically one lagged shock term and one lagged variance term, because empirical research consistently shows this specification fits daily financial return data as well as or better than higher-order variants, and our residual diagnostics confirmed no remaining structure after fitting.

**What GARCH cannot do:** It assumes volatility follows a specific linear mathematical structure. Real markets are messier, different regimes, structural breaks, and nonlinear dynamics during crises. GARCH handles the well-understood parts well. The nonlinear parts are what the LSTM tries to capture.

---

### Model 2: LSTM — The Deep Learning Approach

LSTM stands for Long Short-Term Memory — a type of recurrent neural network specifically designed to learn patterns in sequential data over time.

Where GARCH has a handful of parameters and a fixed mathematical formula, the LSTM has tens of thousands of parameters learned entirely from historical data. It makes no assumptions about what structure volatility follows, it discovers whatever patterns exist by seeing thousands of historical sequences and learning what tends to come next.

The LSTM is trained on a 20-day lookback window using features derived from both the raw returns and the GARCH model outputs, including the conditional variance and standardised residuals. This means GARCH does what it does best first, and LSTM then tries to improve on it by learning the nonlinear patterns GARCH leaves behind.

---

### Model 3: The Hybrid

The hybrid forecast combines the LSTM and GARCH predictions. This is a standard ensemble technique, combining two models often produces better results than either alone because their errors tend to cancel out. The results of this combination and its implications are discussed in the forthcoming paper.

---

## The Data

| Property | Value |
|----------|-------|
| Asset | SPY — SPDR S&P 500 ETF Trust |
| Source | Yahoo Finance via `yfinance` |
| Period | January 1993 to December 2024 |
| Trading days | 8,036 |
| Training period | Feb 1993 – Aug 2018 (80%) |
| Test period | Sep 2018 – Dec 2024 (20%) |

The test period was deliberately chosen to include a full range of market conditions, the calm 2018–2019 bull market, the COVID crash of March 2020, the 2022 rate hike turbulence, and the subsequent recovery. This ensures the evaluation is not limited to a single market regime.

---

## Confirmed Statistical Properties

Before building any model, three statistical properties were formally tested and confirmed, each justifying a key modeling decision:

| Property | Test Used | Conclusion | Modeling Implication |
|----------|-----------|------------|----------------------|
| Non-normality | Jarque-Bera | Fat tails confirmed | Student-t distribution used in GARCH |
| Stationarity | Augmented Dickey-Fuller | Series is stationary | Returns modeled directly without differencing |
| Volatility clustering | Engle ARCH LM | ARCH effects confirmed | GARCH modeling is statistically justified |

---

## Architecture

```
SPY Daily Prices (1993–2024)
           ↓
   Calculate daily log returns
           ↓
      GARCH(1,1) model
   (fitted on training data)
           ↓
   ┌─────────────────────────┐
   │ Conditional variance     │  ← GARCH's estimate of daily volatility
   │ Standardised residuals   │  ← what GARCH cannot explain
   └─────────────────────────┘
           ↓
   Build 7-feature matrix
   with 20-day lookback windows
           ↓
      2-layer LSTM neural network
   (trained on GARCH outputs + return features)
           ↓
   LSTM volatility forecast
           ↓
   Hybrid = weighted combination of LSTM and GARCH
           ↓
   Evaluate all three on held-out test set
   across overall and regime-specific metrics
```

---

## Key Findings

Full numerical results are withheld pending journal submission. The following summarises the directional findings:

- The LSTM substantially outperformed the GARCH baseline on all error metrics — RMSE, MAE, MAPE — across the full test period
- The improvement held consistently across all three volatility regimes — calm markets, normal markets, and crisis periods
- The LSTM tracked the COVID-19 volatility spike and subsequent recovery more precisely than the GARCH baseline
- The hybrid model performance and its implications relative to the individual models are discussed in the forthcoming paper
- Error distributions for all three models show no systematic bias — all forecasts are well-calibrated around zero mean error

For a full discussion of methodology, results, and implications please refer to the forthcoming journal paper or contact the author directly.

---

## Connection to Actuarial Science

The methodology in this project connects directly to actuarial risk quantification. The core actuarial concept of **pure risk premium** — expected frequency times expected severity — has a direct financial analog:

```
Wildfire risk:   Pure Risk Premium = avg fires/year × avg acres/fire
Financial risk:  Volatility Risk   = frequency of shocks × magnitude of shocks
```

GARCH models exactly this structure, the ARCH parameter captures shock magnitude (severity) and the GARCH parameter captures persistence (frequency of elevated risk periods). LSTM then extends this framework by learning nonlinear dynamics that classical actuarial and econometric models cannot capture, the same motivation behind modern catastrophe models that go beyond simple frequency-severity products.

This project sits at the intersection of actuarial science, financial econometrics, and machine learning, three disciplines that are increasingly converging in industry practice.

---

## LSTM Architecture

| Layer | Configuration |
|-------|--------------|
| LSTM (return_sequences=True) | 64 units |
| Dropout | rate = 0.2 |
| LSTM | 32 units |
| Dropout | rate = 0.2 |
| Dense (ReLU activation) | 16 units |
| Dense (linear output) | 1 unit |

- **Lookback window:** 20 trading days
- **Input features:** 7 (log return, GARCH conditional vol, GARCH residual, squared return, absolute return, 5-day rolling vol, 21-day rolling vol)
- **Optimizer:** Adam with learning rate reduction on plateau
- **Regularization:** Dropout + early stopping
- **Reproducibility:** Fixed random seeds (numpy and TensorFlow)

---

## How to Reproduce This Project

```bash
# Install all dependencies
pip install yfinance arch tensorflow scikit-learn pandas numpy matplotlib seaborn statsmodels scipy

# Run scripts in order
python scripts/01_data_download.py       # Download SPY data from Yahoo Finance
python scripts/02_eda.py                 # EDA and statistical tests
python scripts/03_garch_model.py         # Fit GARCH(1,1) and extract conditional volatility
python scripts/04_lstm_model.py          # Build and train LSTM on GARCH outputs
python scripts/05_hybrid_evaluation.py  # Compare all three models on test set
```

All random seeds are fixed — results are fully reproducible.

---

## Repository Structure

```
lstm-garch-sp500/
├── data/
│   ├── spy_data.csv                  # Raw SPY price data
│   ├── garch_outputs.csv             # GARCH conditional volatility and residuals
│   ├── lstm_predictions.csv          # LSTM test set predictions
│   └── model_comparison.csv          # Evaluation metrics (summary only)
├── scripts/
│   ├── 01_data_download.py
│   ├── 02_eda.py
│   ├── 03_garch_model.py
│   ├── 04_lstm_model.py
│   └── 05_hybrid_evaluation.py
├── outputs/
│   ├── 01_spy_price_returns.png
│   ├── 02_return_statistics.png
│   ├── 03_rolling_volatility.png
│   ├── 04_garch_volatility.png
│   ├── 05_garch_residual_diagnostics.png
│   ├── 06_lstm_training.png
│   ├── 07_lstm_forecast.png
│   ├── 08_model_comparison.png
│   └── 09_error_distributions.png
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `yfinance` | Download SPY price data from Yahoo Finance |
| `arch` | GARCH model fitting and diagnostics |
| `tensorflow` / `keras` | LSTM neural network construction and training |
| `scikit-learn` | Feature scaling and evaluation metrics |
| `pandas` / `numpy` | Data manipulation and numerical computation |
| `matplotlib` | All visualisations |
| `statsmodels` / `scipy` | Statistical tests — Jarque-Bera, ADF, ARCH LM |

---

## Related Projects

This project is part of a broader portfolio applying rigorous statistical methods to real-world risk quantification:

- **[Wildfire Risk Analysis](https://github.com/tagoep/Wildfire-Analysis)** — Spatio-temporal analysis of 1.88 million U.S. wildfire records with actuarial risk metrics and an interactive dashboard
- **[ILI Flu Forecasting](https://github.com/tagoep/flu-ili-forecasting)** — ARIMA and STL-based forecasting of U.S. influenza activity using 29 years of CDC surveillance data

---

## Author

**Princess Tagoe**
Statistical Consultant · Data Scientist · Actuarial Science Background

[GitHub](https://github.com/tagoep) · [Medium](https://medium.com/@princesstagoe24)

---

## Citation

If you use this code in your research, please cite as:

```
Tagoe, P. (2026). Hybrid LSTM-GARCH Framework for S&P 500 Volatility Forecasting.
GitHub repository. https://github.com/tagoep/lstm-garch-sp500
```

A formal citation will be updated upon journal publication.

---

## License

MIT License 
