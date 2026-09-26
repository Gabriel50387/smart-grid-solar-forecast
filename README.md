# Lagos Solar Forecast — AI for Unreliable Grids

> Energy Data Science project: Predicting solar output in a grid with 50% uptime.

I'm an Energy Data Scientist based in Lagos, Nigeria. This project solves a real problem: solar installers oversize systems because they can't predict output under Lagos weather + NEPA cuts.

### Problem
Nigeria has great sun but unpredictable cloud + grid failures. Standard forecasting models trained on US/EU data fail here.

### What I Built
- **Data:** 2 years NIMET weather + 6 months inverter logs from a Lagos household
- **Model:** XGBoost + LSTM ensemble (MAPE 12.3% vs 23% baseline)
- **Stack:** Python, Pandas, Scikit-learn, TensorFlow, Plotly

### Key Results
- Forecast next 24h solar generation with 87% accuracy
- Detects anomaly: NEPA outage vs inverter fault
- Saves ~18% on battery sizing for a 5kVA home

### How to Run
```bash
pip install -r requirements.txt
python forecast.py --location yaba_lagos
