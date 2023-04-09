import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Plot the GDP growth rate
plt.plot(df.index, df['GDP_growth_rate'])
plt.title('US GDP Growth Rate')
plt.xlabel('Year')
plt.ylabel('Growth Rate (%)')
plt.show()

# Calculate the mean and standard deviation of the GDP growth rate
mean_growth_rate = df['GDP_growth_rate'].mean()
std_dev_growth_rate = df['GDP_growth_rate'].std()

# Simulate the GDP growth rate until 2100
num_years = 2100 - df.index.year[-1]
simulated_growth_rate = np.random.normal(mean_growth_rate, std_dev_growth_rate, num_years)

# Combine historical and simulated data
simulated_index = pd.date_range(start=df.index[-1] + pd.DateOffset(years=1), end='2100-12-31', freq='A')
simulated_df = pd.DataFrame({'GDP_growth_rate': simulated_growth_rate}, index=simulated_index)
combined_df = pd.concat([df, simulated_df])

# Calculate the cumulative GDP growth
cumulative_growth = (1 + combined_df['GDP_growth_rate'] / 100).cumprod() * df['GDP'][-1]

# Plot the cumulative GDP growth
plt.plot(combined_df.index, cumulative_growth)
plt.title('SIMULATED US GDP Growth Rate') 
plt.xlabel('Year')
plt.ylabel('GDP (trillions of dollars)')
plt.show()
