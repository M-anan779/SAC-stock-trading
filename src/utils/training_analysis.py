import pandas as pd
from pathlib import Path

class Analyzer:

    # load log csv to be analyzed
    def __init__(self, file_path):
        file_path = Path(file_path)
        self.log_df = pd.read_csv(file_path)
    
    # main method for printing all computed stats. rounded to 3 decimals
    def get_summary(self):
        summary = self._get_trading_stats(self.log_df)
        print("\nPerformance Stats:")
        for key in summary:
            print(f'{key}: {round(summary[key], 3)}')

    @staticmethod
    def _get_trading_stats(episode_df):
        # filter data from csv
        episode_df = episode_df[['flag', 'pnl', 'avl_cash', 'date']]
        buy_series = episode_df[episode_df['flag'].isin(['LB', 'SB'])].drop(columns=['flag'])
        sell_series = episode_df[episode_df['flag'].isin(['LS', 'SS'])].drop(columns=['flag'])
        no_position_series = episode_df[episode_df['flag'] == 'NP'].drop(columns=['flag'])
        hold_position_series = episode_df[episode_df['flag'] == 'HP'].drop(columns=['flag'])
        total_years = episode_df.groupby('date').ngroups / 260

        # action counts
        buy_count = len(buy_series)
        sell_count = len(sell_series)
        no_position_count = len(no_position_series)
        hold_position_count = len(hold_position_series)
        total_count = sell_count + buy_count + no_position_count + hold_position_count
        
        # action % rates
        buy_rate = buy_count / total_count
        sell_rate = sell_count / total_count
        hold_position_rate = hold_position_count / total_count
        no_position_rate = no_position_count / total_count

        profit_series = sell_series[sell_series['pnl'] > 0]
        loss_series = sell_series[sell_series['pnl'] < 0]

        # profit/loss totals
        profit_total = profit_series['pnl'].sum()
        loss_total = loss_series['pnl'].sum()
        pnl_total = profit_total - abs(loss_total)

        # profit/loss avg
        profit_avg = profit_series['pnl'].mean()
        loss_avg = loss_series['pnl'].mean()

        # maximums
        profit_max = profit_series['pnl'].max()
        loss_max = loss_series['pnl'].min()

        #win/loss stats
        win_count = len(profit_series)
        lose_count = len(loss_series)
        winlose_total = win_count + lose_count
        win_rate = win_count / winlose_total
        lose_rate = lose_count / winlose_total
        pl_ratio = profit_avg / abs(loss_avg)
        
        # misc. stats
        pnl_per_year = pnl_total / total_years
        expected_value = (win_rate * profit_avg) + (lose_rate * loss_avg)

        # compute final dict for all stats
        dict = {
            'buy count':                    float(buy_count),
            'sell count':                   float(sell_count),
            'hold position count':          float(hold_position_count),
            'no position count':            float(no_position_count),
            'win_count':                    float(win_count),
            'lose_count':                   float(lose_count),
            
            'buy rate':                     float(buy_rate),
            'sell rate':                    float(sell_rate),
            'hold rate':                    float(hold_position_rate),
            'no position rate':             float(no_position_rate),
            'win_rate':                     float(win_rate),
            'lose_rate':                    float(lose_rate),
            
            'profit_total':                 float(profit_total),
            'loss_total':                   float(loss_total),
            'pnl_total':                    float(pnl_total),
            'profit_max':                   float(profit_max),
            'loss_max':                     float(loss_max),
            'profit_avg':                   float(profit_avg),
            'loss_avg':                     float(loss_avg),
            'pl_ratio':                     float(pl_ratio),
            
            'pnl_per_year':                 float(pnl_per_year),
            'expected_value':               float(expected_value)
        }

        return dict

