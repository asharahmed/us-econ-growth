import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load historical GDP data from FRED database
url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=GDP&scale=left&co'
df = pd.read_csv(url, index_col='DATE')
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# Load exogenous variables data
inflation_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=FPCPITOTLZGUSA&scale=left&cos'
inflation_df = pd.read_csv(inflation_url, index_col='DATE')
inflation_df.index = pd.to_datetime(inflation_df.index)
inflation_df.columns = ['inflation_rate']

interest_rate_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=DFF&scale=left&cos'
interest_rate_df = pd.read_csv(interest_rate_url, index_col='DATE')
interest_rate_df.index = pd.to_datetime(interest_rate_df.index)
interest_rate_df.columns = ['interest_rate']

population_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1318&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id=POPTHM&scale=left&cos'
population_df = pd.read_csv(population_url, index_col='DATE')
population_df.index = pd.to_datetime(population_df.index)
population_df.columns = ['population']

# Merge dataframes
df = df.merge(inflation_df, left_index=True, right_index=True)
df = df.merge(interest_rate_df, left_index=True, right_index=True)
df = df.merge(population_df, left_index=True, right_index=True)

# Plot GDP data along with exogenous variables on a single plot
fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].plot(df['GDP'])
ax[0, 0].set_title('GDP')
ax[0, 1].plot(df['inflation_rate'])
ax[0, 1].set_title('Inflation Rate')
ax[1, 0].plot(df['interest_rate'])
ax[1, 0].set_title('Interest Rate')
ax[1, 1].plot(df['population'])
ax[1, 1].set_title('Population')
plt.tight_layout()
plt.show()

