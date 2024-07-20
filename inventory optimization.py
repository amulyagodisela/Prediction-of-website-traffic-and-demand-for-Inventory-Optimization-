#inventory optimisation
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA

# Fit a simple ARIMA model for comparison
arima_model = ARIMA(d_time_series, order=(5, 1, 0)).fit()
arima_predictions = arima_model.predict(start=len(train_df), end=len(train_df) + len(test_df) - 1, dynamic=False)


# Inventory management metrics
initial_inventory = 5500
lead_time = 1  # Example value
service_level = 0.95  # Example value
holding_cost = 0.1  # Example value
stockout_cost = 10  # Example value

#sarimax(views)
z = np.abs(np.percentile(d_predictions, 100 * (1 - service_level)))
order_quantity = np.ceil(d_predictions.mean() + z).astype(int)
reorder_point = d_predictions.mean() * lead_time + z
safety_stock = reorder_point - d_predictions.mean() * lead_time
total_holding_cost = holding_cost * (initial_inventory + 0.5 * order_quantity)
total_stockout_cost = stockout_cost * np.maximum(0, d_predictions.mean() * lead_time - initial_inventory)
total_cost = total_holding_cost + total_stockout_cost
print("TAKING VIEWS")
print("Optimal Order Quantity:", order_quantity)
print("Reorder Point:", reorder_point)
print("Safety Stock:", safety_stock)
print("Total Cost:", total_cost)

#sarimax(without views)
z = np.abs(np.percentile(dm_pred, 100 * (1 - service_level)))
order_quantity = np.ceil(dm_pred.mean() + z).astype(int)
reorder_point = dm_pred.mean() * lead_time + z
safety_stock = reorder_point - dm_pred.mean() * lead_time
total_holding_cost = holding_cost * (initial_inventory + 0.5 * order_quantity)
total_stockout_cost = stockout_cost * np.maximum(0, dm_pred.mean() * lead_time - initial_inventory)
total_cost = total_holding_cost + total_stockout_cost
print("WITHOUT VIEWS")
print("Optimal Order Quantity:", order_quantity)
print("Reorder Point:", reorder_point)
print("Safety Stock:", safety_stock)
print("Total Cost:", total_cost)

#actual demand
z = np.abs(np.percentile(test_df['Demand'], 100 * (1 - service_level)))
order_quantity = np.ceil(test_df['Demand'].mean() + z).astype(int)
reorder_point = test_df['Demand'].mean() * lead_time + z
safety_stock = reorder_point - test_df['Demand'].mean() * lead_time
total_holding_cost = holding_cost * (initial_inventory + 0.5 * order_quantity)
total_stockout_cost = stockout_cost * np.maximum(0, test_df['Demand'].mean() * lead_time - initial_inventory)
total_cost = total_holding_cost + total_stockout_cost
print("ACTUAL INVENTORY IN TEST PERIOD")
print("Optimal Order Quantity:", order_quantity)
print("Reorder Point:", reorder_point)
print("Safety Stock:", safety_stock)
print("Total Cost:", total_cost)

#arima
z = np.abs(np.percentile(arima_predictions, 100 * (1 - service_level)))
order_quantity = np.ceil(arima_predictions.mean() + z).astype(int)
reorder_point = arima_predictions.mean() * lead_time + z
safety_stock = reorder_point - arima_predictions.mean() * lead_time
total_holding_cost = holding_cost * (initial_inventory + 0.5 * order_quantity)
total_stockout_cost = stockout_cost * np.maximum(0, arima_predictions.mean() * lead_time - initial_inventory)
total_cost = total_holding_cost + total_stockout_cost
print("ARIMA")
print("Optimal Order Quantity:", order_quantity)
print("Reorder Point:", reorder_point)
print("Safety Stock:", safety_stock)
print("Total Cost:", total_cost)