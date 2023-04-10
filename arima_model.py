import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

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

# Fit an ARIMA model to the historical GDP growth rate data
model = ARIMA(df['GDP_growth_rate'], order=(1, 1, 0))
model_fit = model.fit()

# Simulate the GDP growth rate until 2100
num_years = 2100 - df.index.year[-1]
predicted_growth_rate = model_fit.forecast(num_years)

# Generate stochastic element
mean_expansion = 0
std_expansion = 0.5
expansion = np.random.normal(mean_expansion, std_expansion, num_years)

# Add stochastic element to predicted growth rate
simulated_growth_rate = predicted_growth_rate + expansion

# Combine historical and simulated data
simulated_index = pd.date_range(start=df.index[-1] + pd.DateOffset(years=1), end='2100-12-31', freq='A')
simulated_df = pd.DataFrame({'GDP_growth_rate': simulated_growth_rate}, index=simulated_index)
combined_df = pd.concat([df, simulated_df])

# Calculate the cumulative GDP growth
cumulative_growth = (1 + combined_df['GDP_growth_rate'] / 100).cumprod()

# Plot the simulated GDP growth
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(cumulative_growth)
ax.set_xlabel('Year')
ax.set_ylabel('Cumulative GDP growth')
ax.set_title('Simulated GDP growth')
plt.show()
