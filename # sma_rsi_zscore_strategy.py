# sma_rsi_zscore_strategy.py

import yfinance as yf
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# To Fetch historical stock data
df = yf.download('GAIL.NS', start='2010-05-01', end='2025-02-25', auto_adjust=True)

# To Clean and format the dataframe
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.dropna(inplace=True)
df.index.name = 'Date'

# To Define custom strategy using SMA, RSI, and Z-score
class SMA_RSI_ZScore_Strategy(Strategy):
    upper_bound_RSI = 68
    lower_bound_RSI = 30
    window_short_sma = 20
    window_long_sma = 90
    window_rsi = 14
    zscore_window = 20

    def init(self):
        close = self.data.Close

        # Calculate rolling mean and std for z-score
        self.mean = self.I(lambda x: pd.Series(x).rolling(self.zscore_window).mean(), close)
        self.std = self.I(lambda x: pd.Series(x).rolling(self.zscore_window).std(), close)
        self.zscore = self.I(lambda x, m, s: (x - m) / s, close, self.mean, self.std)

        # RSI and SMAs
        self.rsi = self.I(talib.RSI, close, self.window_rsi)
        self.long_sma = self.I(talib.SMA, close, self.window_long_sma)
        self.short_sma = self.I(talib.SMA, close, self.window_short_sma)

    def next(self):
        z = self.zscore[-1]

        # Exit conditions
        if crossover(self.rsi, self.upper_bound_RSI) or crossover(self.long_sma, self.short_sma) or z > 1:
            self.position.close()

        # Entry conditions
        elif crossover(self.lower_bound_RSI, self.rsi) or crossover(self.short_sma, self.long_sma) or z < -1:
            self.buy()

# To Run the backtest
bt = Backtest(df, SMA_RSI_ZScore_Strategy, cash=100_000, commission=0.002)
stats = bt.run()
print(stats)

# To Plot equity curve and trade signals
bt.plot()
