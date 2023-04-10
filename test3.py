import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# Load historical GDP data from FRED database
url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=GDP&scale=left&co'
df = pd.read_csv(url, index_col='DATE')
df.index = pd.to_datetime(df.index)
df = df.sort_index()
df.columns = ['gdp']

# Load exogenous variables data
inflation_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo"
# Load exogenous variables data
inflation_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=FPCPITOTLZGUSA&scale=left&cos"
inflation_df = pd.read_csv(inflation_url, index_col='DATE')
inflation_df.index = pd.to_datetime(inflation_df.index)
inflation_df.columns = ['inflation_rate']

interest_rate_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=DFF&scale=left&cos"
interest_rate_df = pd.read_csv(interest_rate_url, index_col='DATE')
interest_rate_df.index = pd.to_datetime(interest_rate_df.index)
interest_rate_df.columns = ['interest_rate']

population_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=POPTHM&scale=left&cos"
population_df = pd.read_csv(population_url, index_col='DATE')
population_df.index = pd.to_datetime(population_df.index)
population_df.columns = ['population']

# Resample exogenous varuables to annual frequency
inflation_df = inflation_df.resample('A').mean()
interest_rate_df = interest_rate_df.resample('A').mean()
population_df = population_df.resample('A').mean()

#Merge exogenous variables data with GDP data
df = df.join(inflation_df).join(interest_rate_df).join(population_df)   
# Create a train-test split for the data
train_end = pd.to_datetime('2000-12-31')
train_data = df.loc[:train_end]
test_data = df.loc[train_end:]

# Seasonal decomposition of GDP data to check for trend and seasonality
result = seasonal_decompose(train_data['gdp'], model='multiplicative')
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize=(15,12))
result.observed.plot(ax=ax1)
ax1.set_ylabel('Observed')
result.trend.plot(ax=ax2)
ax2.set_ylabel('Trend')
result.seasonal.plot(ax=ax3)
ax3.set_ylabel('Seasonal')
result.resid.plot(ax=ax4)
ax4.set_ylabel('Residual')
plt.tight_layout()
plt.show()

# Fit an arima model for each exogenous variable

# Inflation rate
inflation_model = SARIMAX(train_data['inflation_rate'], order=(1,1,1), seasonal_order=(1,1,1,4))
inflation_fit = inflation_model.fit()

# Interest rate
interest_rate_model = SARIMAX(train_data['interest_rate'], order=(1,1,1), seasonal_order=(1,1,1,4))
interest_rate_fit = interest_rate_model.fit()

# Population
population_model = SARIMAX(train_data['population'], order=(1,1,1), seasonal_order=(1,1,1,4))
population_fit = population_model.fit()

# Generate predictions for the test data using the fitted models

inflation_forecast = inflation_fit.predict(start=pd.to_datetime('2011-01-01'), end=pd.to_datetime('2100-12-31'))
interest_rate_forecast = interest_rate_fit.predict(start=pd.to_datetime('2011-01-01'), end=pd.to_datetime('2100-12-31'))
population_forecast = population_fit.predict(start=pd.to_datetime('2011-01-01'), end=pd.to_datetime('2100-12-31'))

# Display the predictions
print(inflation_forecast)
print(interest_rate_forecast)
print(population_forecast)

# Since predictions are all 0, we will use the mean of the train data for the predictions
inflation_forecast = [train_data['inflation_rate'].mean()]*90
interest_rate_forecast = [train_data['interest_rate'].mean()]*90
population_forecast = [train_data['population'].mean()]*90


# create a sarimax model to predict gdp using the forecasted exogenous variable values

exog = pd.DataFrame({'inflation_rate': inflation_forecast, 'interest_rate': interest_rate_forecast, 'population': population_forecast})

exog.index = pd.to_datetime(exog.index, origin='2011-01-01', unit='D')
exog.index = exog.index + pd.DateOffset(years=1)

# Filter exog data to remove NaN or Inf. values
exog = exog[exog['inflation_rate'] != np.inf]
exog = exog[exog['interest_rate'] != np.inf]
exog = exog[exog['population'] != np.inf]

model = SARIMAX(train_data['gdp'], exog=train_data[['inflation_rate', 'interest_rate', 'population']], order=(1,1,1), seasonal_order=(1,1,1,4))
fit = model.fit()

# Generate predictions for the test data using the fitted model
predictions = fit.predict(start=pd.to_datetime('2011-01-01'), end=pd.to_datetime('2100-12-31'), exog=exog)

# Plot the predictions
plt.figure(figsize=(15,8))
plt.plot(train_data['gdp'], label='Train')
plt.plot(test_data['gdp'], label='Test')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()
