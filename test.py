import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pomegranate import HiddenMarkovModel, NormalDistribution, StudentTDistribution

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

# Define the Hidden Markov Model
model = HiddenMarkovModel()

# Define the emission distributions for each state
d1 = StudentTDistribution(3, 0, 1)
d2 = StudentTDistribution(3, 0, 1)
d3 = StudentTDistribution(3, 0, 1)

# Add the states and emission distributions to the model
model.add_states(['state1', 'state2', 'state3'])
model.add_distribution(d1)
model.add_distribution(d2)
model.add_distribution(d3)

# Define the starting probabilities and transition matrix
model.add_transition(model.start, 'state1', 0.5)
model.add_transition(model.start, 'state2', 0.25)
model.add_transition(model.start, 'state3', 0.25)
model.add_transition('state1', 'state1', 0.8)
model.add_transition('state1', 'state2', 0.1)
model.add_transition('state1', 'state3', 0.1)
model.add_transition('state2', 'state1', 0.2)
model.add_transition('state2', 'state2', 0.6)
model.add_transition('state2', 'state3', 0.2)
model.add_transition('state3', 'state1', 0.1)
model.add_transition('state3', 'state2', 0.2)
model.add_transition('state3', 'state3', 0.7)

# Finalize the model
model.bake()

# Fit the model to the historical GDP growth rate data
X = np.array(df['GDP_growth_rate']).reshape(-1, 1)
model.fit(X)

# Simulate the GDP growth rate until 2100
num_years = 2100 - df.index.year[-1]
simulated_growth_rate = []
state_sequence = model.predict(X)
current_state = state_sequence[-1]

for i in range(num_years):
    if current_state == 0:
        simulated_growth_rate.append(d1.sample()[0])
    elif current_state == 1:
        simulated_growth_rate.append(d2.sample()[0])
    else:
        simulated_growth_rate.append(d3.sample()[0])

    simulated_gdp = simulated_gdp * (1 + simulated_growth_rate[-1] / 100)
    
# Create a new dataframe containing the simulated GDP growth rate and GDP
dates = pd.date_range(start=df.index[-1], periods=num_years, freq='A')
simulated_df = pd.DataFrame({'GDP_growth_rate': simulated_growth_rate, 'GDP': simulated_gdp}, index=dates)

# Plot the simulated GDP growth rate and GDP
fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plt.plot(simulated_df.index, simulated_df['GDP'], label='Simulated GDP')
plt.legend()
plt.show()


