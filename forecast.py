import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Create our own small solar data - no download needed
hours = np.arange(6, 19)  # 6am to 6pm
solar_power = [0, 50, 180, 350, 500, 600, 650, 600, 500, 350, 180, 50, 10]

df = pd.DataFrame({'Time': hours, 'Solar_kW': solar_power})

print("My first Smart Grid dataset:")
print(df.head())

# Plot
plt.figure()
plt.plot(df['Time'], df['Solar_kW'], marker='o')
plt.xlabel('Hour of Day')
plt.ylabel('Solar Power (kW)')
plt.title('Solar Power - My First Smart Grid Plot')
plt.grid(True)
plt.show()

print("Success! You did Day 1.")


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Use your solar data to train a model
X = df[['Time']] # Input: time of day
y = df['Solar_kW'] # Output: solar power

# Train two models
model1 = LinearRegression()
model1.fit(X, y)

model2 = RandomForestRegressor(n_estimators=100, random_state=0)
model2.fit(X, y)

# Predict tomorrow at 1pm
tomorrow_1pm = pd.DataFrame({'Time': [13]})
pred1 = model1.predict(tomorrow_1pm)
pred2 = model2.predict(tomorrow_1pm)

print(f"Prediction for tomorrow 1pm - Linear Model: {pred1[0]:.1f} kW")
print(f"Prediction for tomorrow 1pm - Smart Model (Random Forest): {pred2[0]:.1f} kW")

# Smart Grid Decision
if pred2[0] > 300:
    print("Decision: SOLAR IS HIGH -> Charge battery / Export to grid")
else:
    print("Decision: SOLAR IS LOW -> Discharge battery")

# Plot prediction vs real
plt.figure()
plt.scatter(df['Time'], df['Solar_kW'], label='Real Data')
plt.plot(df['Time'], model2.predict(X), color='red', label='AI Prediction')
plt.xlabel('Hour')
plt.ylabel('kW')
plt.title('AI Forecasting for Smart Grid')
plt.legend()
plt.grid(True)
plt.show()
