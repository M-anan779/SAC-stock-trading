import pandas as pd
import numpy as np
import pandas_ta as ta

class TechnicalIndicators:
    def __init__(self, df):
        self.df = df

    @staticmethod
    def _get_slope(y):
        if y.isnull().any():
            return np.nan
        x = np.arange(len(y))
        return np.polyfit(x, y, 1)[0]

    def _add_price(self):
            self.df['market_price'] = (self.df['high'] + self.df['low'] + self.df['close']) / 3

    def _add_atr(self):
            self.df['atr'] = ta.atr(self.df['high'], self.df['low'], self.df['close'])

    def _add_volatility(self):
            self.df['volatility'] = np.log(self.df['close'] / self.df['close'].shift(1)).rolling(20).std()

    def _add_roc(self):
            self.df['roc'] = ta.roc(self.df['close'], length=14)

    def _add_price_zscore(self):
            rolling_mean = self.df['close'].rolling(20).mean()
            rolling_std = self.df['close'].rolling(20).std()
            self.df['price_zscore'] = (self.df['close'] - rolling_mean) / rolling_std

    def _add_vwap_dev(self):
            vwap = ta.vwap(self.df['high'], self.df['low'], self.df['close'], self.df['volume'])
            self.df['vwap_dev'] = self.df['close'] - vwap

    def _add_adx(self):
            adx = ta.adx(self.df['high'], self.df['low'], self.df['close'], length=14)
            self.df['adx'] = adx['ADX_14'] if 'ADX_14' in adx else np.nan

    def _add_slope_regression(self):
            self.df['slope'] = self.df['close'].rolling(20).apply(self._get_slope, raw=False)

    def _add_volume_zscore(self):
            rolling_mean = self.df['volume'].rolling(20).mean()
            rolling_std = self.df['volume'].rolling(20).std()
            self.df['volume_zscore'] = (self.df['volume'] - rolling_mean) / rolling_std

    def generate_all_indicators(self):
        self._add_price()
        self._add_atr()
        self._add_volatility()
        self._add_roc()
        self._add_price_zscore()
        self._add_vwap_dev()
        self._add_adx()
        self._add_slope_regression()
        self._add_volume_zscore()
        return self.df
