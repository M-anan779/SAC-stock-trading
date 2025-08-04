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
    
    def _add_candle_strength_features(self):
        self.df['candle_strength'] = abs(self.df['close'] - self.df['open']) / (self.df['high'] - self.df['low']).replace(0.0, np.nan)
        self.df['candle_strength'] = self.df['candle_strength'].replace(0.0, np.nan)
        self.df['strength_against_prev'] = self.df['candle_strength'] / self.df['candle_strength'].shift(axis=0, periods=1)
        self.df['price_efficiency'] = abs(self.df['close'] - self.df['open']) / self.df['volume']

    def _add_volume_zscore(self):
        rolling_mean = self.df['volume'].rolling(20).mean()
        rolling_std = self.df['volume'].rolling(20).std()
        self.df['volume_zscore'] = (self.df['volume'] - rolling_mean) / rolling_std
    
    def _add_price_zscore(self):
        rolling_mean = self.df['close'].rolling(20).mean()
        rolling_std = self.df['close'].rolling(20).std()
        self.df['price_zscore'] = (self.df['close'] - rolling_mean) / rolling_std

    def generate_all_indicators(self):
        self._add_direction_acceleration_features()
        self._add_trend_confirmation_features()
        self._add_candle_strength_features()
        self._add_volume_zscore()
        self._add_price_zscore()
        self._add_market_price()

        return self.df
