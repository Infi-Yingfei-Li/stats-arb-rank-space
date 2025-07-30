## Requirements

- Python 3.10
- numpy==1.24.3
- pandas==2.0.3
- torch==2.2.0+cu118
- matplotlib>=3.7.1

## Data preparation

Please ensure the following file is present before running the code: 

(1) data/equity_data/Fama_French_3factor_19900101_20221231.csv
This CSV file contains the following columns with name:
- date: the format of date is %m/%d/%y
- mktrf: the risk-free rate
- smb: Fama-French factor
- hml: Fama-French factor
- umd: Fama-French factor

The CSV file can be downloaded from the Kenneth R. French Data Library at https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html.

(2) data/equity_data/equity_data_19900101_20221231.csv
This CSV file contains the following columns with name:
- PERMNO: the stock identifier in CRSP database
- Ticker: the stock ticker
- DlyCalDt: the date of the stock price, in the format of %Y-%m-%d
- DlyPrc: the daily price of the stock
- ShrOut: the number of shares outstanding
- DlyCap: the daily market capitalization of the stock
- DlyRet: the daily return of the stock

Comments:
1. The data is from CRSP and covers all trading days from 1970-01-01 to 2022-12-31. 
2. The order of stocks index depends on the order of the stock identifier in the PERMNO column.

(3) data/equity_data/equity_data_high_frequency/equity_data_Polygon_high_freq_5_min_{}.npz
The npz files contain high-frequency data for all stocks on every trading days, where `{}` is the date in the format of %Y-%m-%d for all trading days from 2023-09-10 to 2022-12-30.

Each file contains the following items:
- 'time_axis_high_freq': 1D numpy array with shape (T, ), the intra-day time axis for high-frequency data, in the format of datetime timestamp
- 'price_high_freq': 2D numpy array with shape (N, T). N is the number of stocks. The high-frequency price data for each stock.
- 'capitalization_high_freq': 2D numpy array with shape (N, T). N is the number of stocks. The high-frequency market capitalization data for each stock.
- 'rank_high_freq': 2D numpy array with shape (N, T). The rank in capitalization for each stock at each time point. The rank in capitalization is directly derived from the capitalization data. Specifically, ar[i, j] = k means that the stock with index i occupies the k-th rank at time j.
- 'equity_idx_by_rank_high_freq': 2D numpy array with shape (N, T). The equity index by rank for each stock at each time point. Specifically, ar[k, j] = i means that the stock with index i occupies the k-th rank at time j.

Comments:
1. The high-frequency capitalization data is derived from the high-frequency price data from Polygon.io and the number of shares outstanding from CRSP. 
2. We note that the price data from Polygon.io may not be consistent with the CRSP data. In most cases, the inconsistency originates from the historical price data not being correctly adjusted for stock splits and dividends, which requires correction before using the data for analysis.
3. The order of the stocks in the high-frequency data is consistent with the order in the equity_data.csv file. 

## Run the code
To run the code, please follow these steps:
1. Set up a Python environment with the required packages.

2. Prepare the data files mentioned above and place them in the correct directories.

3. Run the python script by the following command:
```bash
python main_name.py
python main_rank.py
python main_rank_high_freq.py
```

4. Run the Jupyter notebook "notebook/portfolio_performance_PnL/portfolio_performance_PnL.ipynb" to backtest the portfolio performance.




