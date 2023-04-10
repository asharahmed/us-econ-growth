import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM



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

# Fit a Hidden Markov Model to the historical GDP growth rate data
model = GaussianHMM(n_components=4)  # modified from n_components=2

# Estimate the mean and standard deviation of the GDP growth rate in each state
X = np.array(df['GDP_growth_rate']).reshape(-1, 1)
model.fit(X)
mean_states, std_states = model.means_, np.sqrt(model.covars_)

# Estimate the transition probabilities between the states
startprob = np.ones(4) / 4  # equal probability of starting in any state
transmat = np.array([
    [0.6, 0.2, 0.1, 0.1],
    [0.1, 0.6, 0.2, 0.1],
    [0.1, 0.1, 0.6, 0.2],
    [0.2, 0.1, 0.2, 0.5]
])  # arbitrary transition probabilities, should be tuned
model.startprob_ = startprob
model.transmat_ = transmat
model.fit(X)

# Simulate the GDP growth rate until 2100
num_years = 2100 - df.index.year[-1]
simulated_growth_rate = []
state_sequence = model.predict(X)[-1]

for i in range(num_years):
    if state_sequence == 0:
        simulated_growth_rate.append(np.random.normal(mean_states[0], std_states[0]))
    elif state_sequence == 1:
        simulated_growth_rate.append(np.random.normal(mean_states[1], std_states[1]))
    elif state_sequence == 2:
        simulated_growth_rate.append(np.random.normal(mean_states[2], std_states[2]))
    else:
        simulated_growth_rate.append(np.random.normal(mean_states[3], std_states[3]))
    state_sequence = np.random.choice(range(4), p=model.transmat_[state_sequence])

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
