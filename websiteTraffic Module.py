#websiteTraffic forecasting
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# Step 12: Split the data into training and testing sets
split_point = int(len(rescaledX_df) * 0.92)
train_df = rescaledX_df[:split_point]
test_df = rescaledX_df[split_point:]

time_series = train_df.set_index('Date')['Views']
plt.style.use('fivethirtyeight')
plt.figure(figsize=(15, 10))
plt.plot(time_series)
plt.title('Daily Traffic of website.com')
plt.show()

p, d, q = 5,1,0
traffic_model=sm.tsa.statespace.SARIMAX(time_series,order=(p, d, q),seasonal_order=(1, 1, 0, 12))
traffic_model=traffic_model.fit()
predictions = traffic_model.predict(start=len(train_df), end=len(train_df)+len(test_df)-1)
print("Predictions")
print(predictions)

rescaledX_df.set_index('Date')['Views'].plot(legend=True, label="Training Data", figsize=(15, 10))
predictions.plot(legend=True, label="Predictions")