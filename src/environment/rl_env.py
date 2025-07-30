import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
from pathlib import Path
from math import log
from torch.utils.tensorboard import SummaryWriter

project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / 'data'
aggregated_dir = data_dir / 'aggregated'
processed_dir = data_dir / 'processed'

class PositionNode:
    # position data
    def __init__(self, delta_a, cash, position_price):
        self.delta_a = delta_a
        self.cash = cash
        self.position_price = position_price

class PositionLedger:
    def __init__(self, init_cash):
        self.ledger = []
        self.cash = init_cash
    
    # pop top of stack
    def _pop(self):
        return self.ledger.pop()
    
    # push to top of stack
    def _push(self, node):
        self.ledger.append(node)
    
    # add when entering a new position
    def add_position(self, buy_delta, position_price):
        node = PositionNode(buy_delta, self.cash, position_price)
        self._push(node)
    
    # return shares needed to exit position
    def sell_position(self, sell_delta):
        total_delta = 0
        total_shares = 0

        # get latest position
        while self.ledger and total_delta < sell_delta:
            top = self._pop()
            position_price = top.position_price
            current_delta = top.delta_a
            projected_delta = total_delta + current_delta

            if projected_delta > sell_delta:
                # partial sale
                used_delta = sell_delta - total_delta
                total_shares += used_delta * self.cash / position_price

                remaining_delta = current_delta - used_delta
                if remaining_delta > 0:
                    top.delta_a = remaining_delta
                    self._push(top)     # if a position is not completely sold

                total_delta += used_delta
            
            # full sale
            else:
                total_shares += sell_delta * self.cash / position_price
                total_delta += sell_delta

        return total_shares

class TradingEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # initialization
        self.current_step = 0
        self.current_episode = 0
        self.init_cash = 100000
        self.timesteps = 0
        self.states = None 
        self.window_size = 12
        self.action_space = spaces.Box(low=-1, high=1, shape = (1, ), dtype = np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape = (self.window_size, 14), dtype = np.float32)
        self.log_df = pd.DataFrame(columns=[
            'date', 'ticker', 'avl_cash', 'market_price', 'shares', 'action', 'nav', 'reward'
            ])

        self.writer = SummaryWriter(log_dir="logs/nav_logging")

        # meta vector index order
        self.avl_cash_idx = 0
        self.marketprice_idx = 1
        self.shares_idx = 2
        self.action_idx = 3
        self.nav_idx = 4
        self.reward_idx = 5

        # load ticker files
        self.ticker_file_paths = [f for f in processed_dir.iterdir()]
        ticker_df = pd.read_csv(self.ticker_file_paths[0])

        # compute global min, max per column for normalization later
        df = ticker_df.drop(columns = ['date', 'window_start', 'ticker'])
        self.global_max = df.max()
        self.global_min = df.min()
        
        # group ticker files by date
        self.episode_df_list = []
        for _, group in ticker_df.groupby('date'):
            self.episode_df_list.append(group)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # initialization of new training episode
        self.positions = PositionLedger(self.init_cash)
        self.current_step = 0
        t = self.current_step        
        self.current_episode = np.random.randint(0, len(self.episode_df_list))

        # reset previous states and compute state transitions
        self.states = []
        self.obs = []        
        group_df = self.episode_df_list[self.current_episode]
        self._window_helper(group_df)       # handles normalization internally
        
        # return normalized obs
        observation = self._get_obs(t)

        return observation, {}      # return empty info, handled internally
    
    def step(self, action):
        # store action
        action = np.clip(action, -1, 1)
        t = self.current_step
        self.states[t][self.action_idx] = action

        # compute trade and retrieve reward
        self._trade_helper(t)
        reward = self.states[t][self.reward_idx]

        # check for episode end and log results
        terminate = False
        if t == len(self.states)-1:
            terminate = True
            df = pd.DataFrame(self.states, columns=[
                    "avl_cash", "market_price", "shares", "action", "nav", "reward"
                ])
            
            df.insert(0, "ticker", self.current_ep_ticker)
            df.insert(0, "date", self.current_ep_date)
            self.log_df = pd.concat([self.log_df, df], ignore_index=True)

            csv_path = "logs/trades_log.csv"
            df.to_csv(csv_path, mode="a", index=False, header=not os.path.exists(csv_path))

            eod_nav = self.states[t][self.nav_idx]      # log nav on tensorboard
            self.writer.add_scalar('env/eod_nav', eod_nav, self.current_episode)
            self.writer.flush() 
        
        # increment step
        self.current_step += 1
        observation = self._get_obs(t)
        truncated = False
        
        return observation, reward, terminate, truncated, {}
    
    def _get_obs(self, t):
        return self.obs[t]
   
    def _get_reward(self, t):
        # check t == 0 and adjust t accordingly to prevent out of bounds error
        prev_t = self._check_index(t)

        # initialization and partial computation
        avl_cash = self.states[t][self.avl_cash_idx]
        shares = self.states[t][self.shares_idx]
        market_price = self.states[t][self.marketprice_idx]

        nav = avl_cash + abs(shares * market_price)
        prev_nav = self.states[prev_t][self.nav_idx] 
        self.states[t][self.nav_idx] = nav              # store current nav for next timestep

        # compute reward
        curr_reward = log(nav / prev_nav)
        prev_reward = self.states[prev_t][self.reward_idx]
        self.states[t][self.reward_idx] = curr_reward + prev_reward    # rewards are cumulative

    def _window_helper(self, episode_df):
        # store date and ticker info for logging at episode end
        self.current_ep_date = episode_df['date'].iloc[0]
        self.current_ep_ticker = episode_df['ticker'].iloc[0]
        market_price = episode_df['market_price']
        
        # drop unnecessary info and normalize data
        episode_df = episode_df.drop(columns = ['date', 'window_start', 'ticker'])
        normalized_group = self._normalize_df(episode_df)
        
        # compute and store obs and meta (info) vector
        for i in range(len(episode_df) - self.window_size + 1):
            obs_window = normalized_group.iloc[i:i+self.window_size].to_numpy(dtype=np.float32)

            meta = np.zeros(6, dtype=np.float32)
            meta[self.avl_cash_idx] = self.init_cash 
            meta[self.marketprice_idx] = market_price.iloc[i+self.window_size-1]  # price of last candle of current state where trades are taken
            
            self.states.append(meta)
            self.obs.append(obs_window)
        
        self.states = np.array(self.states, dtype=np.float32)       # info for logging
        self.obs = np.array(self.obs, dtype=np.float32)             # obs for training 
    
    def _normalize_df(self, df):
        # min-max normalization per column for current episode
        df_normalized = (df - self.global_min) / (self.global_max - self.global_min)
        df_normalized = df_normalized.clip(0, 1)

        return df_normalized

    def _trade_helper(self, t):
        action = self.states[t][self.action_idx]

        if t == 0:
            prev_action = 0
        else:
            prev_action = self.states[t-1][self.action_idx]

        # determine whether action is "suy short position (SB)" or "sell short position (SS)"
        if action <= 0 and prev_action <= 0:
            if action < prev_action:
                self._buy(action, prev_action, 'SB', t, reversal=False)
            elif action > prev_action:
                self._sell(action, prev_action, 'SS', t)

        # determine whether action is "sell long position (LS)" or "buy long position (LB)"
        elif action >= 0 and prev_action >= 0:
            if action < prev_action:
                self._sell(action, prev_action, 'LS', t)
            elif action > prev_action:
                self._buy(action, prev_action, 'LB', t, reversal=False)

        # determine whether action is "short reversal-> sell short position + buy new long position" 
        # or "long reversal-> sell long position + buy new short position"
        else:
            if action > 0 and prev_action < 0:
                self._sell(0, prev_action, 'SS', t)
                self._buy(action, 0, 'LB', t, reversal=True)
            else:
                self._sell(0, prev_action, 'LS', t)
                self._buy(action, 0, 'SB', t, reversal=True)
        
        # compute reward for step()
        self._get_reward(t)
        
    def _buy(self, action, prev_action, flag, t, reversal):
        prev_t = self._check_index(t)       # check t == 0

        # change in action (delta) used to determine shares for a position
        delta_a = action - prev_action
        market_price = self.states[t][self.marketprice_idx]
        new_shares = abs(delta_a * self.init_cash / market_price)       # compute new shares

        current_shares = self.states[t][self.shares_idx]

        # if this operation is part of second leg of reversal, then use the current timestep instead to access info for calculations
        # else initialize as normal
        if (reversal):  
            avl_cash = self.states[t][self.avl_cash_idx]
            prev_shares = self.states[t][self.shares_idx]
        else:
            avl_cash = self.states[prev_t][self.avl_cash_idx]
            prev_shares = self.states[prev_t][self.shares_idx]

        # calculate available cash and current share holdings for new trade
        if flag == 'LB':                            # buy long position
            avl_cash -= new_shares * market_price
            current_shares = prev_shares + new_shares

        elif flag == 'SB':                          # buy short position
            avl_cash -= new_shares * market_price
            current_shares = prev_shares - new_shares
        
        # update info to finalize trade
        self.states[t][self.avl_cash_idx] = avl_cash
        self.states[t][self.shares_idx] = current_shares
        
        # add new position for future sale calculations
        self.positions.add_position(abs(delta_a), market_price)

    def _sell(self, action, prev_action, flag, t):
        prev_t = self._check_index(t)       # check t == 0
        
        delta_a = action - prev_action
        new_shares = self.positions.sell_position(abs(delta_a)) # retrieve correct amount of shares to sell depending on position price and delete the position
        current_shares = self.states[t][self.shares_idx]
        prev_shares = self.states[prev_t][self.shares_idx]

        market_price = self.states[t][self.marketprice_idx]
        avl_cash = self.states[prev_t][self.avl_cash_idx]

        # calculate available cash and current share holdings for new trade
        if flag == 'LS':                                # sell long position
            avl_cash += new_shares * market_price
            current_shares = prev_shares - new_shares

        elif flag == 'SS':                              # sell short position
            avl_cash += new_shares * market_price
            current_shares = prev_shares + new_shares
        
        # update info to finalize trade
        self.states[t][self.avl_cash_idx] = avl_cash
        self.states[t][self.shares_idx] = current_shares
    
    def _check_index(self, t):
        if (t == 0):
            return t
        else:
            return t - 1



     

    

    