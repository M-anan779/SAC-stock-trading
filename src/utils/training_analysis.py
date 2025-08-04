import pandas as pd
from pathlib import Path

import pandas as pd
from pathlib import Path

class Analyzer:
    def __init__(self, log_file, output_file):
        self.log_df = pd.read_csv(log_file)
        self.output_file = output_file
        self.summary_df = self._summarize_episodes()

        self.summary_df.to_csv(self.output_file, index=False)
        print(f"Episode summary saved to: {self.output_file}")

    def _summarize_episodes(self):
        summary_rows = []

        grouped = self.log_df.groupby(['date', 'ticker'])

        for (date, ticker), group in grouped:
            stats = {}
            mean, min_, max_, range_ = self._get_trade_stats(group)

            # Build flattened stats row
            stats['date'] = date
            stats['ticker'] = ticker

            for col in mean.index:
                stats[f'mean_{col}'] = mean[col]
                stats[f'min_{col}'] = min_[col]
                stats[f'max_{col}'] = max_[col]
                stats[f'range_{col}'] = range_[col]

            summary_rows.append(stats)

        return pd.DataFrame(summary_rows)

    @staticmethod
    def _get_trade_stats(episode_df):
        df = episode_df.drop(columns=['date', 'ticker'])
        mean = df.mean()
        min_ = df.min()
        max_ = df.max()
        range_ = abs(max_) - abs(min_)
        return mean, min_, max_, range_

# Usage
file = Path('logs/trades_log_best.csv')
analyzer = Analyzer(file, 'analysis.csv')

