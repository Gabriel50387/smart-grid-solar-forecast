# Lagos Solar Forecast — AI for Unreliable Grids

> Built for Lagos, Nigeria. Forecasting solar power to stabilize smart grids using live satellite data.

[![Python](https://img.shields.io/badge/Python-3.9+-blue)]()
[![Lagos](https://img.shields.io/badge/Location-Yaba%2C%20Lagos-green)]()
[![Live%20API](https://img.shields.io/badge/Data-Open--Meteo%20Live-orange)]()

### Problem
Lagos has 60% unreliable grid. Solar is abundant but intermittent. How do we predict power for smart grid decisions?

### Solution
**V1 (forecast.py):** Random Forest with Temp + Cloud + Hour → MAE, MAPE, Smart Decision Engine
**V3 LIVE (forecast_v3_live.py):** Pulls REAL live data from Open-Meteo API for Yaba (6.49°N, 3.34°E) - radiation, cloud cover, temp. No fake data.

### Tech Stack
Python, pandas, scikit-learn, Open-Meteo API, matplotlib, Smart Grid Logic

### Results
- Live Lagos forecast: ~MAE 45kW on 5kW system
- Smart Decision: HIGH / MEDIUM / LOW solar → battery charge/discharge logic
- Plot: `live_forecast.png`

### How to Run
```bash
pip install -r requirements.txt
python forecast_v3_live.py
```

---


**Author:** Gabriel | Aspiring Energy Data Scientist | Yaba, Lagos | Learning in public for PV Power Nigeria Fair 2025

