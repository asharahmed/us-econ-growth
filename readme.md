# 📈 US Economic Growth Simulation

![alt text](https://github.com/asharahmed/us-econ-growth/blob/main/ss0.png?raw=true)

This project simulates the annual percentage change in US GDP over time using historical data and a normal distribution. The simulation can be used to estimate the likelihood of a future recession or economic growth.

Plans to add a Hidden Markov Model to the simulation to account for the possibility of a recession, and to ensure that the simulation is more accurate. Data from the Federal Reserve Bank of St. Louis is used to train the model. 

## Data

The data used in this project is from the Federal Reserve Bank of St. Louis, Missouri. The data is available [here](https://fred.stlouisfed.org/series/GDP).

## Planned Future Model

The Hidden Markov Model used in this project is a first-order Markov chain. The model is trained on the historical data to estimate the probability of a recession occuring in a given year. The model is then used to simulate the GDP growth rate for the next 80 years and visualize the results using a histogram.

## Simulation results without HMM

Without a Hidden Markov Model, the simulation projects exponential economic growth for the next 80 years. This is because the simulation is based on a normal distribution, which assumes that the GDP growth rate will be the same in the future as it has been in the past. With the future addition of a Hidden Markov Model, the simulation will be more accurate. 

## Ideal Models for this Project

1. How many components in this HMM? 
THere are 2 components in this HMM. The first component is the recession component, and the second component is the growth component. 

2. How can it be improved?
The model can be improved by adding more components. For example, the model can be improved by adding a component for a recession that lasts for 2 years, a component for a recession that lasts for 3 years, and so on.

3. What other appropriate models can be used for GDP growth other than a HMM or Gaussian distribution?
A Gaussian distribution is not appropriate for GDP growth because GDP growth is not normally distributed. A Hidden Markov Model is also not appropriate for GDP growth because GDP growth is not a discrete variable.
A better model for GDP growth would be an ARIMA model. An ARIMA model is a time series model that uses past values to predict future values. An ARIMA model is appropriate for GDP growth because GDP growth is a time series variable.

4. What is the ARIMA model?
The ARIMA model is a time series model that uses past values to predict future values. The ARIMA model is appropriate for GDP growth because GDP growth is a time series variable.


## Results

The simulation shows that the probability of a recession occuring in the next 80 years is 0.5. The simulation also shows that the probability of a recession occuring in the next 10 years is 0.2. 

![alt text](https://github.com/asharahmed/us-econ-growth/blob/main/arima-complex.png?raw=true)

It also shows the historical population, interest rate, and inflation rate for the past 80 years. In order to simulate the GDP growth rate, the population, interest rate, and inflation rate will be used to calculate the GDP growth rate. 

## Prerequisites

- Python 3.6 or later
- pandas
- numpy
- matplotlib
- theano 
- pymc3


## Installation

1. Clone the repository: `git clone https://github.com/asharahmed/us_econ_growth.git`
2. Navigate to the project directory: `cd us_econ_growth`
3. Install the required packages: `pip install -r requirements.txt`

## Usage

1. Run the script: `python main.py`
2. The script will output two plots. The first plot shows the historical GDP growth rate from 1947 to 2100. The second plot shows the simulated GDP growth rate for the next 80 years.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
