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


df = pd.merge(df, inflation_df, how='left', left_index=True, right_index=True)
df = pd.merge(df, interest_rate_df, how='left', left_index=True, right_index=True)
df = pd.merge(df, population_df, how='left', left_index=True, right_index=True)

# Fill missing values with interpolation

df = df.interpolate()

# Define test and train sets
train = df[:'2020']
test = df['2020':]


# decompose train data into trend, seasonal and residual components

decomposition = seasonal_decompose(train['gdp'], model='multiplicative', period=4)
train['trend'] = decomposition.trend
train['seasonal'] = decomposition.seasonal
train['residual'] = decomposition.resid

# Plot decomposed data
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(train['gdp'], label='Observed')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(train['trend'], label='Trend')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(train['seasonal'], label='Seasonality')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(train['residual'], label='Residuals')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Plot ACF and PACF of residual data
plot_acf(train['residual'].dropna(), lags=50)
plot_pacf(train['residual'].dropna(), lags=50)
plt.show()


if np.isinf(train[['inflation_rate', 'interest_rate', 'population']]).any().any() or np.isnan(train[['inflation_rate', 'interest_rate', 'population']]).any().any():
    # Remove exog variables with missing values or infinite values
    train = train.dropna(subset=['inflation_rate', 'interest_rate', 'population'], how='any')
    test = test.dropna(subset=['inflation_rate', 'interest_rate', 'population'], how='any')
    
# Define exogenous variables
exog_vars = ['inflation_rate', 'interest_rate', 'population']

# Create Sarimax models 


# Create SARIMAX model for inflation rate
model = SARIMAX(train['inflation_rate'], order=(1, 1, 1), seasonal_order=(0, 1, 1, 4))
inflation_results = model.fit(maxiter=100000)
# Use model to predict inflation rate
future_exog = test[exog_vars]

# Create SARIMAX model for interest rate
model = SARIMAX(train['interest_rate'], order=(1, 1, 1), seasonal_order=(0, 1, 1, 4))
interest_results = model.fit(maxiter=100000)
# Use model to predict interest rate
future_exog = test[exog_vars]

# Create SARIMAX model for population
model = SARIMAX(train['population'], order=(1, 1, 1), seasonal_order=(0, 1, 1, 4))
population_results = model.fit(maxiter=100000)
# Use model to predict population
future_exog = test[exog_vars]

# Use the SARIMAX models to forecast exogenous variables for the additional periods

inflation_forecast = inflation_results.forecast(steps=80, exog=future_exog)
interest_forecast = interest_results.forecast(steps=80, exog=future_exog)
population_forecast = population_results.forecast(steps=80, exog=future_exog)

# Create a new DataFrame for the additional periods
index = pd.date_range(start=test.index[-1], periods=80+1, freq='Q')[1:]
additional_periods = pd.DataFrame(index=index)

# Add the forecasted values to the new DataFrame
additional_periods['inflation_rate'] = inflation_forecast
additional_periods['interest_rate'] = interest_forecast
additional_periods['population'] = population_forecast

# Append the new DataFrame to the original test DataFrame
test = pd.concat([test, additional_periods])

# Remove exog variables with missing values or infinite values
train = train.dropna(subset=['inflation_rate', 'interest_rate', 'population'], how='any')
test = test.dropna(subset=['inflation_rate', 'interest_rate', 'population'], how='any')




# Fit SARIMAX model with exogenous variables, use predicted values for exogenous variables
model = SARIMAX(train['gdp'], exog=test[exog_vars], order=(1, 1, 1), seasonal_order=(0, 1, 1, 4))
results = model.fit(maxiter=100000)




print("The length of test is {}".format(len(test)))

# Forecast GDP using fitted model and exogenous variables
forecast = results.get_forecast(steps=80, exog=test[['inflation_rate', 'interest_rate', 'population']])


# Plot actual and forecasted GDP
plt.figure(figsize=(12, 8))
plt.plot(train['gdp'], label='Train')
plt.plot(test['gdp'], label='Test')
plt.plot(forecast.predicted_mean, label='Forecast')
plt.legend(loc='upper left')
plt.show()
