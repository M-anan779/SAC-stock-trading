import numpy as np
import pandas as pd
import pandas_ta as ta

class TechnicalIndicators:
    def __init__(self, df):
        self.df = df

    def _add_market_price(self):
        self.df["market_price"] = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
    
    def _add_directional_features(self):
        adx = ta.adx(self.df["high"], self.df["low"], self.df["close"], length=14, scalar=100, mamode="ema")
        dmp = adx["DMP_14"]
        dmn = adx["DMN_14"]

        dmi_dev = dmp - dmn
        self.df["DMI_dev"] = dmi_dev
        self.df["ADX"] = adx["ADX_14"]
    
    def _add_trend_confirmation_features(self):
        K_length = 10
        D_length = 3
        
        highest_high = self.df["high"].rolling(K_length).max()
        lowest_low = self.df["low"].rolling(K_length).min()
        highlow_range = highest_high - lowest_low
        relative_range = self.df["close"] - (highest_high + lowest_low) / 2

        smooth_relative_range = ta.ema(ta.ema(relative_range, D_length), D_length)
        smooth_highlow_range = ta.ema(ta.ema(highlow_range, D_length), D_length)
        smi = 200 * (smooth_relative_range / (smooth_highlow_range.replace(0, 1e-8)))

        self.df["time_since_reversal"] = self._time_since_reversal_helper(series=smi, upper_thresh=40, lower_thresh=-40)

        ema_length = 3
        smi_dev = smi - ta.ema(smi, ema_length)
        smi_dev_delta = smi_dev - smi_dev.shift(axis=0, periods=1).fillna(0)
        self.df["SMI_dev_delta"] = smi_dev_delta

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
    
    def _add_price_zscore(self):
        rolling_mean = self.df["close"].rolling(30).mean()
        rolling_std = self.df["close"].rolling(30).std(ddof=0)
        self.df["price_zscore"] = ((self.df["close"] - rolling_mean) / rolling_std.replace(0, 1e-8)).clip(-5, 5)

    def _add_candle_strength_features(self):
        true_range = (self.df["high"] - self.df["low"]).replace(0, 1e-8)  # True range per bar
        body = self.df["close"] - self.df["open"]

        # candle body size normalized by range
        self.df["price_movement"] = (body / true_range).clip(-5, 5)

    def _add_volatility_zscore(self):
        returns = self.df["close"].pct_change()
        vol = returns.rolling(30).std()
        rolling_mean = vol.rolling(60).mean()
        rolling_std = vol.rolling(60).std(ddof=0)
        self.df["volatility_zscore"] = ((vol - rolling_mean) / rolling_std.replace(0, 1e-8)).clip(-5, 5)

    def _normalize(self):
        self.df["time_since_reversal"] = (self.df["time_since_reversal"] / 78).clip(0,1)
        self.df["ADX"] = (self.df["ADX"] / 50)
        
        cols = ["DMI_dev", "SMI_dev_delta", "price_movement"]
        rolling_mean = self.df[cols].shift(1).rolling(30).mean()
        rolling_std = self.df[cols].shift(1).rolling(30).std(ddof=0)
        self.df[cols] = ((self.df[cols] - rolling_mean) / rolling_std.replace(0, 1e-8)).clip(-5, 5)
        
        cols.append("price_zscore")
        cols.append("volatility_zscore")
        self.df[cols] = np.tanh(self.df[cols])
        
    def generate_all_indicators(self):
        self._add_market_price()
        self._add_directional_features()
        self._add_trend_confirmation_features()
        self._add_price_zscore()
        self._add_candle_strength_features()
        self._add_volatility_zscore()
        self._normalize()

        return self.df
