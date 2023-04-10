import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load historical GDP data from FRED database
url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=GDP&scale=left&co'
gdp = pd.read_csv(url, index_col='DATE', parse_dates=['DATE'], infer_datetime_format=True)
gdp = gdp.resample('A').last()

# Load exogenous variable data
url_inflation = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=FPCPITOTLZGUSA'
inflation = pd.read_csv(url_inflation, index_col='DATE', parse_dates=['DATE'], infer_datetime_format=True)
inflation = inflation.resample('A').last()

# Select relevant columns and rename them
inflation = inflation[['FPCPITOTLZGUSA']]
inflation.columns = ['inflation_rate']


url_interest = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS'
interest = pd.read_csv(url_interest, index_col='DATE', parse_dates=['DATE'], infer_datetime_format=True)
interest = interest.resample('A').last()

# Select relevant columns and rename them
interest = interest[['FEDFUNDS']]
interest.columns = ['interest_rate']


url_population = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=POPTHM'
population = pd.read_csv(url_population, index_col='DATE', parse_dates=['DATE'], infer_datetime_format=True)
population = population.resample('A').last()

# Select relevant columns and rename them
population = population[['POPTHM']]
population.columns = ['population']


# Merge GDP data with exogenous variable data
data = pd.concat([gdp, inflation, interest, population], axis=1, join='inner')
data.columns = ['gdp', 'inflation', 'interest', 'population']

# Calculate the annual percentage change in GDP
data['gdp_growth_rate'] = data['gdp'].pct_change(periods=1) * 100

# Remove the first row containing NaN
data = data.iloc[1:]

# Determine the appropriate order of the SARIMAX model
fig, ax = plt.subplots(figsize=(12, 8))
plot_acf(data['gdp_growth_rate'], ax=ax)
plot_pacf(data['gdp_growth_rate'], ax=ax)
plt.show()

# Based on the ACF and PACF plots, we can set p=1 and q=0
p = 1
q = 0



# Fit a SARIMAX model to the historical GDP growth rate data with exogenous variables
model = SARIMAX(data['gdp_growth_rate'], exog=data[['inflation', 'interest', 'population']], order=(p, 1, q))
model_fit = model.fit()


# Generate a new set of dates from 2021 to 2100
future_years = pd.date_range(start='2021-01-01', end='2100-01-01',
                                 freq='A', closed='left')

# Create a new DataFrame with the missing years
future_data = pd.DataFrame(index=future_years, columns=data.columns)

# Only include the new years in future_data
future_data = future_data.loc['2021':'2100']

# Concatenate the existing exogenous variable data with the new DataFrame
exog = pd.concat([data[['inflation', 'interest', 'population']], future_data[['inflation', 'interest', 'population']][:79]])

# Ensure exogenous variable data length matches the length of the forecasted data
assert len(exog) == len(future_data) + len(data)
# If not equal, then the exogenous variable data is missing some years
# to fix this, we can use the interpolate() method to fill in the missing values
exog = exog.interpolate()



# Generate the forecasted GDP growth rates for the years 2021-2100
forecast = model_fit.get_forecast(steps=len(future_years), exog=exog)
forecasted_gdp_growth_rate = forecast.predicted_mean

# Plot the forecasted GDP growth rates
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(future_years, forecasted_gdp_growth_rate, color='red')
ax.set_xlabel('Year')
ax.set_ylabel('GDP Growth Rate (%)')
ax.set_title('Forecasted GDP Growth Rates')
plt.show()

# Calculate the forecasted GDP values for the years 2021-2100
forecasted_gdp = data['gdp'].iloc[-1] * (1 + forecasted_gdp_growth_rate / 100).cumprod()

# Plot the forecasted GDP values
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(future_years, forecasted_gdp, color='red')
ax.set_xlabel('Year')
ax.set_ylabel('GDP ($)')
ax.set_title('Forecasted GDP Values')
plt.show()
