# Lagos Solar Forecast — AI for Unreliable Grids

> Built for Lagos, Nigeria. Forecasting solar power to stabilize smart grids using live satellite data.

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Location](https://img.shields.io/badge/Location-Yaba%2C%20Lagos-green) ![Data](https://img.shields.io/badge/Data-Open--Meteo%20Live-orange)

### Problem
Lagos has 60% unreliable grid. Solar is abundant but intermittent. How do we predict power for smart grid decisions?

### Solution
**V1 (forecast.py):** Random Forest with Temp + Cloud + Hour → MAE, MAPE, Smart Decision Engine
**V3 LIVE (forecast_v3_live.py):** Pulls REAL live data from Open-Meteo API for Yaba (6.49°N, 3.34°E) - radiation, cloud cover, temp. No fake data. Battery SOC logic included.

### Location Flexibility
- Tested: Yaba, Lagos (6.5095°N, 3.3711°E) - where 3.5KVA hybrid inverter was built/tested
- Adaptable: Ketu, Lagos (6.5976°N, 3.3892°E) - current residence, 1-line code change
- Model works for any Lagos site

### Tech Stack
Python, pandas, scikit-learn, Open-Meteo API, Streamlit, Git/GitHub

### Hardware Link
Complements my final year project: **Design and Construction of a 3.5KVA Hybrid Solar Inverter** (BOUESTI, 2025) - transformer sizing/winding, PWM, auto switchover, protection circuitry.

### How to Run
```bash
pip install pandas scikit-learn requests streamlit
python forecast.py
```

---


**Author:** Gabriel | Aspiring Energy Data Scientist | Yaba, Lagos | Learning in public for PV Power Nigeria Fair 2025

