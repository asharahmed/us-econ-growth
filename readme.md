# 📈 US Economic Growth Simulation

![alt text](https://github.com/asharahmed/us-econ-growth/blob/main/ss.png?raw=true)

This project simulates the annual percentage change in US GDP over time using historical data and a normal distribution. The simulation can be used to estimate the likelihood of a future recession or economic growth.

Plans to add a Hidden Markov Model to the simulation to account for the possibility of a recession, and to ensure that the simulation is more accurate. Data from the Federal Reserve Bank of St. Louis is used to train the model. 

## Data

The data used in this project is from the Federal Reserve Bank of St. Louis, Missouri. The data is available [here](https://fred.stlouisfed.org/series/GDP).

## Model

The hidden markov model used in this project is a first-order Markov chain. The model is trained on the historical data to estimate the probability of a recession occuring in a given year. The model is then used to simulate the GDP growth rate for the next 80 years. 

## Results

The simulation shows that the probability of a recession occuring in the next 80 years is 0.5. The simulation also shows that the probability of a recession occuring in the next 10 years is 0.2. 

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
