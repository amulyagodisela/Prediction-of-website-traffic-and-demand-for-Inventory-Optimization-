#Demand Module
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Train SARIMAX model for demand prediction
d_time_series = train_df.set_index('Date')['Demand']
exog_views = train_df.set_index('Date')['Views']
exog_pred = test_df.set_index('Date')['Views']
order = (7, 1, 0)
seasonal_order = (5, 1, 0, 12)
#sarimax(views)
demand_model = SARIMAX(d_time_series, exog=exog_views, order=order, seasonal_order=seasonal_order)
demand_model_fit = demand_model.fit()
#sarimax(no views)
d_model=SARIMAX(d_time_series, order=order, seasonal_order=seasonal_order)
d_model_fit = d_model.fit()

# Predict demand
d_predictions = demand_model_fit.predict(len(train_df), len(train_df) + len(test_df) - 1, exog=exog_pred)
d_predictions = d_predictions.astype(int)
print(d_predictions)
dm_pred= d_model_fit.predict(len(train_df), len(train_df) + len(test_df) - 1)
dm_pred = dm_pred.astype(int)
print(dm_pred)