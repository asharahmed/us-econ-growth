import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import itertools


# Load historical GDP data from FRED database
url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=GDP&scale=left&co'
df = pd.read_csv(url, index_col='DATE')
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# Convert quarterly data to annual data
df = df.resample('A').last()

# Calculate the annual percentage change in GDP
df['GDP_growth_rate'] = df['GDP'].pct_change(periods=1) * 100

# Remove the first row containing NaN
df = df.iloc[1:]

# Determine the appropriate order of the ARIMA model
fig, ax = plt.subplots(figsize=(12, 8))
plot_acf(df['GDP_growth_rate'], ax=ax)
plot_pacf(df['GDP_growth_rate'], ax=ax)
plt.show()



# Based on the ACF and PACF plots, we can set p=1 and q=0
p = 2
q = 3

# Fit an ARIMA model to the historical GDP growth rate data
model = ARIMA(df['GDP_growth_rate'], order=(p, 1, q))
model_fit = model.fit()

# Generate the forecasted GDP growth rates for the years 2021-2100
forecast_years = pd.date_range(start='2021-01-01', end='2100-01-01', freq='A')
num_years = len(forecast_years)
forecasted_growth_rates = model_fit.forecast(steps=num_years)[0]

# Create a DataFrame with the forecasted GDP growth rates
forecast_df = pd.DataFrame(index=forecast_years, columns=['GDP_growth_rate'])
forecast_df['GDP_growth_rate'] = forecasted_growth_rates

# Combine the historical GDP data and the forecasted data
combined_df = pd.concat([df, forecast_df])

# Calculate the cumulative GDP growth over time
cumulative_growth = (1 + combined_df['GDP_growth_rate'] / 100).cumprod()

# Calculate the simulated GDP values
initial_gdp = df['GDP'].iloc[-1]
simulated_gdp = initial_gdp * cumulative_growth
simulated_gdp[0] = initial_gdp

# Plot the simulated GDP growth over time
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(simulated_gdp)
ax.set_xlabel('Year')
ax.set_ylabel('GDP (in trillions of dollars)')
ax.set_title('Simulated GDP growth until 2100')
plt.show()