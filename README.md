# M4-Forecasting

Deep learning & transformer benchmarks on the M4 Competition time series dataset (100,000 series).

- **Dataset**: M4 Competition (Yearly, Quarterly, Monthly, Weekly, Daily, Hourly)
- **Baselines**: Naive, ETS, ARIMA
- **Deep Learning**: N-BEATS, PatchTST, iTransformer, TimesNet, Autoformer, DLinear and more
- **Evaluation**: sMAPE, MASE (official M4 metrics)
- **Goal**: Compare classic and modern architectures, from stat models to SOTA transformers

## Setup

- Requirements: numpy, pandas, matplotlib, scikit-learn, torch, tqdm, requests, jupyter
- Download data:
    python src/data/download.py
- Run notebooks:
    jupyter lab notebooks/

## References

- M4 Competition: github.com/M4Competition/M4-methods
- Modern models: github.com/thuml/Time-Series-Library

---

Production-ready code for advanced time series modeling. See notebooks for details and concrete implementation of every architecture.
