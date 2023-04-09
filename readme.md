# 📈 US Economic Growth Simulation

![alt text](https://github.com/asharahmed/us-econ-growth/blob/main/ss.png?raw=true)

This project simulates the annual percentage change in US GDP over time using historical data and a normal distribution. The simulation can be used to estimate the likelihood of a future recession or economic growth.

Plans to add a Hidden Markov Model to the simulation to account for the possibility of a recession, and to ensure that the simulation is more accurate. Data from the Federal Reserve Bank of St. Louis is used to train the model. 

## Data

The data used in this project is from the Federal Reserve Bank of St. Louis. The data is available [here](https://fred.stlouisfed.org/series/GDP).

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
