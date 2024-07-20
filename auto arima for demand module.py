!pip install pmdarima
from pmdarima.arima import auto_arima
model1 = auto_arima(train_df.set_index('Date')['Demand'],start_p=1, start_q=1, d=1, max_p=5, max_q=5, max_d=5,m=12,start_P=0,D=1,start_Q=0,max_P=5,max_D=5,seasonal=True,information_criteria = 'AIC' )
model1.summary()