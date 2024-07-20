#Evaluation
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
# Forecast accuracy metrics
actual_demand = test_df['Demand']
predicted_demand = d_predictions
pred_demand = dm_pred

mae = mean_absolute_error(actual_demand, predicted_demand)
mse = mean_squared_error(actual_demand, predicted_demand)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((actual_demand - predicted_demand) / actual_demand)) * 100
print("TAKING VIEWS")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")

mae = mean_absolute_error(actual_demand, pred_demand)
mse = mean_squared_error(actual_demand, pred_demand)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((actual_demand - pred_demand) / actual_demand)) * 100
print("WITHOUT VIEWS")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")

# Fit a simple ARIMA model for comparison
arima_model = ARIMA(d_time_series, order=(5, 1, 0)).fit()
arima_predictions = arima_model.predict(start=len(train_df), end=len(train_df) + len(test_df) - 1, dynamic=False)

mae_arima = mean_absolute_error(actual_demand, arima_predictions)
mse_arima = mean_squared_error(actual_demand, arima_predictions)
rmse_arima = np.sqrt(mse_arima)
mape_arima = np.mean(np.abs((actual_demand - arima_predictions) / actual_demand)) * 100

print(f"ARIMA Model - \nMAE: {mae_arima}\n MSE: {mse_arima}\n RMSE: {rmse_arima}\n MAPE: {mape_arima}%")

# Plot actual vs predicted demand
plt.figure(figsize=(12, 6))
plt.plot(test_df['Date'], actual_demand, label='Actual Demand')
plt.plot(test_df['Date'], predicted_demand, label='Predicted Demand(Taking views)', linestyle='--')
plt.plot(test_df['Date'], pred_demand, label='Predicted Demand(Without views)', linestyle=':')
plt.plot(test_df['Date'], arima_predictions, label='ARIMA Model', linestyle='-.')
plt.xlabel('Date')
plt.ylabel('Demand')
plt.title('Actual vs Predicted Demand')
plt.legend()
plt.show()

# Calculate residuals
residuals = actual_demand.values - predicted_demand.values

# Ensure dates are aligned with residuals
dates = test_df['Date']

# Plot residuals
plt.figure(figsize=(12, 6))
plt.plot(dates, residuals, label='Residuals')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.legend()
plt.show()

