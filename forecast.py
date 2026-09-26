import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# --- 1. BETTER DATA (Simulating Lagos with weather) ---
np.random.seed(42)
hours = np.arange(6, 19)
temp_c = np.array([28, 30, 32, 34, 35, 36, 36, 35, 34, 32, 31, 29, 28])
cloud = np.array([20, 15, 10, 5, 10, 15, 25, 30, 20, 10, 15, 20, 25])
solar_power = np.array([50, 180, 350, 500, 600, 650, 600, 500, 350, 180, 50, 10, 0]) + np.random.normal(0, 20, 13)

df = pd.DataFrame({
    'Hour': hours,
    'Temp_C': temp_c,
    'Cloud_%': cloud,
    'Solar_kW': solar_power
})

# --- 2. FEATURE ENGINEERING ---
X = df[['Hour', 'Temp_C', 'Cloud_%']]
y = df['Solar_kW']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# --- 3. MODEL + EVALUATION ---
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

print(f"Model Evaluation for Lagos:")
print(f"MAE: {mae:.1f} kW | MAPE: {mape:.1f}%")

# --- 4. SMART GRID DECISION ENGINE ---
tomorrow = pd.DataFrame({'Hour': [13], 'Temp_C': [35], 'Cloud_%': [10]})
tomorrow_pred = model.predict(tomorrow)[0]

print(f"\nForecast tomorrow 1pm (Yaba, Lagos): {tomorrow_pred:.1f} kW")
if tomorrow_pred > 300:
    print("Decision: HIGH SOLAR -> Charge battery, run freezer, export")
else:
    print("Decision: LOW SOLAR -> Conserve, discharge battery")

# --- 5. PRO PLOT ---
plt.figure()
plt.scatter(df['Hour'], df['Solar_kW'], label='Real Lagos Data')
plt.plot(df['Hour'], model.predict(X), color='red', label='AI Forecast')
plt.xlabel('Hour of Day')
plt.ylabel('kW')
plt.title('Lagos Solar Forecast — Energy Data Science')
plt.legend()
plt.grid(True)
plt.savefig('forecast_plot.png', dpi=150)
plt.show()
