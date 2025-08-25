import pandas as pd
from pathlib import Path

class Analyzer:
    def __init__(self, log_file):
        self.log_df = pd.read_csv(log_file)
    
    def get_summary(self):
        summary = self._get_trading_stats(self.log_df)
        print(summary)

    @staticmethod
    def _get_trading_stats(episode_df):
        episode_df = episode_df[['flag', 'pnl']]
        sell_series = episode_df[episode_df['flag'] == 'LS'].drop(columns=['flag'])
        buy_series = episode_df[episode_df['flag'] == 'LB'].drop(columns=['flag'])
        hold_series = episode_df[episode_df['flag'] == 'H'].drop(columns=['flag'])

        sell_count = len(sell_series)
        buy_count = len(buy_series)
        hold_count = len(hold_series)
        total_count = sell_count + buy_count + hold_count
        
        buy_rate = buy_count / total_count
        sell_rate = sell_count / total_count
        hold_rate = hold_count / total_count

        profit_series = sell_series[sell_series['pnl'] > 0]
        loss_series = sell_series[sell_series['pnl'] < 0]

        profit_total = profit_series['pnl'].sum()
        loss_total = loss_series['pnl'].sum()
        total_diff = profit_total - abs(loss_total)

        profit_avg = profit_series['pnl'].mean()
        loss_avg = loss_series['pnl'].mean()

        profit_max = profit_series['pnl'].max()
        loss_max = loss_series['pnl'].min()

        win_count = len(profit_series)
        lose_count = len(loss_series)
        winlose_total = win_count + lose_count

        win_rate = win_count / winlose_total
        lose_rate = lose_count / winlose_total

        winlose_ratio = win_count/lose_count
        avg_pl_ratio = profit_avg / abs(loss_avg)
        total_pl_ratio = profit_total / abs(loss_total)

        expected_value = (win_rate * profit_avg) + (lose_rate * loss_avg)

        row = {
            "sell_count":       float(sell_count),
            "buy_count":        float(buy_count),
            "hold_count":       float(hold_count),
            "buy_rate":         float(buy_rate),
            "sell_rate":        float(sell_rate),
            "hold_rate":        float(hold_rate),
            "profit_total":     float(profit_total),
            "loss_total":       float(loss_total),
            "total_diff":       float(total_diff),
            "profit_avg":       float(profit_avg),
            "loss_avg":         float(loss_avg),
            "profit_max":       float(profit_max),
            "loss_max":         float(loss_max),
            "win_count":        float(win_count),
            "lose_count":       float(lose_count),
            "win_rate":         float(win_rate),
            "lose_rate":        float(lose_rate),
            "winlose_ratio":    float(winlose_ratio),
            "avg_pl_ratio":     float(avg_pl_ratio),
            "total_pl_rato":    float(total_pl_ratio),
            "expected_value":   float(expected_value)
        }

        return row

# Usage
file = Path('logs/trades-0825-0313.csv')
analyzer = Analyzer(file)
analyzer.get_summary()

