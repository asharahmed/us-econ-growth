import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm 
import theano.tensor as tt

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

# Define the number of states for the HMM
n_states = 3

# Define the prior probabilities for the initial state
p_initial = np.array([0.6, 0.2, 0.2])

# Define the transition probabilities between states
p_transitions = np.array([[0.8, 0.1, 0.1],
                          [0.2, 0.6, 0.2],
                          [0.1, 0.1, 0.8]])

# Define the means and standard deviations for each state
means = np.array([1.5, 2.5, 3.5])
std_devs = np.array([0.5, 1.0, 1.5])

# Define the HMM model
with pm.Model() as model:
    # Define the latent state variable
    states = pm.Categorical('states', p=p_initial, shape=len(df) + 1)
    print(states[:-1])

    # Define the observed variable for the GDP growth rate
    obs = pm.Normal('obs', mu=tt.switch(states[:-1] < n_states, means[states[:-1]], np.nan),
                    sd=tt.switch(states[:-1] < n_states, std_devs[states[:-1]], np.nan),
                    observed=df['GDP_growth_rate'].values)

    # Define the transition probabilities
    transitions = pm.Deterministic('transitions', p_transitions[states[:-1], :, np.newaxis][:-1])
    
    # Run the HMM sampler
    trace = pm.sample(2000, tune=1000, cores=2, target_accept=0.95)


    

# Extract the simulated GDP growth rate from the trace
simulated_growth_rate = trace['obs'][-1, :]

# Combine historical and simulated data
simulated_index = pd.date_range(start=df.index[-1] + pd.DateOffset(years=1), end='2100-12-31', freq='A')
simulated_df = pd.DataFrame({'GDP_growth_rate': simulated_growth_rate}, index=simulated_index)
combined_df = pd.concat([df, simulated_df])


# Calculate the cumulative GDP growth
cumulative_growth = (1 + combined_df['GDP_growth_rate'] / 100).cumprod()


# Plot the simulated GDP growth rate
plt.plot(simulated_df.index, simulated_df['GDP_growth_rate'])
plt.title('SIMULATED US GDP Growth Rate')
plt.xlabel('Year')
plt.ylabel('Growth Rate (%)')
plt.show()

