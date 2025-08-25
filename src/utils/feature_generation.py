import numpy as np
import pandas as pd
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
    
    def _add_slope_regression(self):
        self.df['slope'] = self.df['close'].rolling(20).apply(self._get_slope, raw=False)

    def _add_market_price(self):
        self.df['market_price'] = (self.df['high'] + self.df['low'] + self.df['close']) / 3

    def _add_direction_acceleration_features(self):
        adx = ta.adx(self.df['high'], self.df['low'], self.df['close'], length=14, scalar=100, mamode='ema')
        dmp = adx['DMP_14']
        dmn = adx['DMN_14']

        dmi_dev = dmp - dmn
        dev_delta = dmi_dev - dmi_dev.shift(axis=0, periods=1).fillna(0)
        
        self.df['DMI_dev'] = dmi_dev
        self.df['dev_delta'] = dev_delta
    
    def _add_trend_confirmation_features(self):
        K_length = 10
        D_length = 3
        ema_length = 3

        highest_high = self.df['high'].rolling(K_length).max()    # nan filled later
        lowest_low = self.df['low'].rolling(K_length).min()
        highlow_range = highest_high - lowest_low
        relative_range = self.df['close'] - (highest_high + lowest_low) / 2

        smooth_relative_range = ta.ema(ta.ema(relative_range, D_length), D_length)
        smooth_highlow_range = ta.ema(ta.ema(highlow_range, D_length), D_length)
        smi = 200 * (smooth_relative_range / smooth_highlow_range)

        self.df['time_since_reversal'] = self._time_since_reversal_helper(series=smi, upper_thresh=40, lower_thresh=-40)
        
        smi_dev = smi - ta.ema(smi, ema_length)
        smi_ema_dev_delta = smi_dev - smi_dev.shift(axis=0, periods=1).fillna(0)
        self.df['SMI_EMA_dev_delta'] = smi_ema_dev_delta
    
    @staticmethod
    def _time_since_reversal_helper(series, upper_thresh, lower_thresh):
        smi= series.to_numpy()
        result = np.zeros_like(smi, dtype=np.float32)

        count = 0
        for i in range(1, len(smi)):
            if (smi[i-1] > upper_thresh and smi[i] < upper_thresh) or (smi[i-1] < lower_thresh and smi[i] > lower_thresh):
                count = 0
            else:
                count += 1
            
            result[i] = count
        
        return pd.Series(result, index=series.index)

    def _add_volume_zscore(self):
        rolling_mean = self.df['volume'].rolling(20).mean()
        rolling_std = self.df['volume'].rolling(20).std()
        self.df['volume_zscore'] = (self.df['volume'] - rolling_mean) / rolling_std
    
    def _add_price_zscore(self):
        rolling_mean = self.df['close'].rolling(20).mean()
        rolling_std = self.df['close'].rolling(20).std()
        self.df['price_zscore'] = (self.df['close'] - rolling_mean) / rolling_std

    def _add_candle_strength_features(self):
        self.df['price_movement'] = (self.df['close'] - self.df['open'] / self.df['close'] - self.df['open'].shift(axis=0, periods=1).fillna(1)) - 1
        self.df['price_efficiency'] = self.df['close'] - self.df['open'] / abs(self.df['volume_zscore'])

    def _normalize(self):
        self.df['time_since_reversal'] = (self.df['time_since_reversal'] / 78).clip(0,1)
        
        cols = ['DMI_dev', 'dev_delta', 'SMI_EMA_dev_delta', 'price_movement', 'price_efficiency']
        rolling_mean = self.df[cols].shift(1).rolling(20).mean()
        rolling_std = self.df[cols].shift(1).rolling(20).std(ddof=0)
        eps = 1e-8
        self.df[cols] = ((self.df[cols] - rolling_mean) / np.maximum(rolling_std, eps)).clip(-10, 10)
        
        cols.append('price_zscore')
        cols.append('volume_zscore')
        self.df[cols] = np.tanh(self.df[cols])
        

    def generate_all_indicators(self):
        self._add_direction_acceleration_features()
        self._add_trend_confirmation_features()
        self._add_volume_zscore()
        self._add_price_zscore()
        self._add_market_price()
        self._add_candle_strength_features()
        self._normalize()

        return self.df
