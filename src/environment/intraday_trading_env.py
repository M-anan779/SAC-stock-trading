import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
import math
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / 'data' / 'processed_tickers_5-minute'

class PositionNode:
    # position data
    def __init__(self, delta_a, cash, position_price):
        self.delta_a = delta_a
        self.cash = cash
        self.position_price = position_price
        self.shares = (delta_a * cash) / position_price

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
    def add_position(self, buy_delta, flag, position_price):
        if (buy_delta == 0.0):
            market_price = position_price
            top = self._pop()
            position_price = top.position_price
            shares = top.shares    
            pnl = self.get_pnl(flag, market_price, position_price, shares)
            
            self._push(top)
            
            return pnl
        else: 
            node = PositionNode(buy_delta, self.cash, position_price)
            self._push(node)
            
            return 0.0
    
    # return shares needed to exit position
    def sell_position(self, sell_delta, t, flag, market_price):
        if (sell_delta == 0.0):
            return 0.0, 0.0
        total_shares = 0
        total_pnl = 0
        top = self._pop()
        
        position_price = top.position_price     
    
        current_delta = top.delta_a
        shares_to_sell = sell_delta * self.cash / position_price
        
        remaining = current_delta - sell_delta
        if remaining < 0:
            total_shares, total_pnl = self.sell_position(remaining, t, flag, market_price)
        else:
            top.delta_a = remaining
            self._push(top)
        
        shares_to_sell = (sell_delta * self.cash) / position_price
        total_shares += shares_to_sell

        total_pnl += self.get_pnl(flag, market_price, position_price, shares_to_sell)

        return total_shares, total_pnl
    
    def get_pnl(self, flag, market_price, position_price, shares):
        if (flag == "L"):
            pnl = (market_price - position_price) * shares
        elif (flag == "S"):
            pnl = -1 * (market_price - position_price) * shares
        else:
            pnl = 0.0
        return pnl
    

class TradingEnv(gym.Env):

    def __init__(self):
        super().__init__()

        # initialization
        self.current_step = 0
        self.global_step = 0
        self.current_episode = 0
        self.init_cash = 50000
        self.states = None 
        self.window_size = 12
        self.action_space = spaces.Box(low=-1, high=1, shape = (1, ), dtype = np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape = (self.window_size, 9), dtype = np.float32)
        self.log_df = pd.DataFrame(columns=[
            'date', 'ticker', 'avl_cash', 'market_price', 'shares', 'action', 'pnl', 'reward'
            ])

        self.start_time = datetime.now().strftime("%m%d-%H%M")

        # meta vector index order
        self.avl_cash_idx = 0
        self.marketprice_idx = 1
        self.shares_idx = 2
        self.action_idx = 3
        self.pnl_idx = 4
        self.reward_idx = 5

        self.minmax_dict = {}

        # load ticker files and group episodes
        self.ticker_file_paths = [f for f in data_dir.iterdir()]
        self.episode_df_list = []
        for file in data_dir.iterdir():
            df = pd.read_csv(file)
            
            if (df['ticker'].iloc[0] != 'AAPL'):
                continue
            
            temp_df = df.drop(columns=['date', 'ticker'])
            
            minmax = {}
            for col in temp_df.columns.tolist():
                minmax[col] = (df[col].min(), df[col].max())
            self.minmax_dict[file.stem] = minmax
            
            for _, group in df.groupby('date'):
                self.episode_df_list.append(group)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # initialization of new training episode
        self.positions = PositionLedger(self.init_cash)
        self.current_step = 0
        t = self.current_step        

        # reset previous states and compute state transitions
        while True:
            self.states = []
            self.obs = []
            self.current_episode = np.random.randint(0, len(self.episode_df_list))
            self._window_helper(self.episode_df_list[self.current_episode])
            if len(self.obs) > 15:
                break
            print(f"[WARNING] Skipping empty episode index {self.current_episode}")
        
        # return normalized obs
        observation = self._get_obs(t)
        
        return observation, {}      # return empty info, handled internally
    
    def step(self, action):       
        t = self.current_step

        # clean and store action
        action = np.clip(action, -1, 1)
        action = np.round(action, 1)
        action = abs(action) 
        if action <= 0.5:
            action = 0.0
        else:
            action = 1
        self.states[t][self.action_idx] = action

        # compute trade and retrieve reward
        self._trade_helper(t)
        reward = self.states[t][self.reward_idx]

        # increment step
        self.current_step += 1
        self.global_step += 1
        observation = self._get_obs(t)
        truncated = False

        # check for episode end and log results
        terminate = False
        if t == len(self.states)-1:
            terminate = True
            df = pd.DataFrame(self.states, columns=[
                    "avl_cash", "market_price", "shares", "action", "pnl", "reward"
                ])
            
            df.insert(0, "ticker", self.current_ep_ticker)
            df.insert(0, "date", self.current_ep_date)
            self.log_df = pd.concat([self.log_df, df], ignore_index=True)

            ep_pnl = df["pnl"].sum()
            info = {"episode_pnl": ep_pnl}
            
            csv_path = os.path.join("logs", "trades-" + self.start_time + ".csv")
            df.to_csv(csv_path, mode="a", index=False, header=not os.path.exists(csv_path))
        else:
            info = {}
        
        return observation, reward, terminate, truncated, info
    
    def _get_obs(self, t):
        return self.obs[t]
   
    def _get_reward(self, t):
        reward = 0
        if t != 0:
            pnl = self.states[t][self.pnl_idx]
            curr_action = self.states[t][self.action_idx]
            prev_action = self.states[t-1][self.action_idx]
            target = 0.005 * self.init_cash      # profit target: 0.5% of portfolio value
            
            relative = (pnl / target)

            if curr_action == prev_action:
                # holding no position
                if curr_action == 0:
                    reward = 0.0
                
                # holding position
                else:
                    if pnl > 0:
                        reward = relative * 1.25     # greater emphasis on sustaining short term profit
                    else:
                        if relative <= -1:
                            reward = relative * 1.5
                        else:
                            reward = relative
            else:
                # sold position
                if curr_action ==  0:
                    
                    # profit
                    if pnl > 0:
                        reward = relative * 2.5
                    
                    # loss
                    else:
                        reward = relative * 3    # greater loss ratio
                
                # bought position
                else:
                    reward = 0.0
        
        reward = np.tanh(reward) * 2
        self.states[t][self.reward_idx] += reward


    def _window_helper(self, episode_df):
        # store date for logging at episode end
        self.current_ep_ticker = episode_df['ticker'].iloc[0]
        self.current_ep_date = episode_df['date'].iloc[0]
        market_price = episode_df['market_price']
        
        # drop unnecessary info and normalize data
        episode_df = episode_df.drop(columns = ['date', 'ticker', 'market_price'])
        normalized_group = self._normalize_df(episode_df, self.current_ep_ticker)
        
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
    
    def _normalize_df(self, df, ticker):
        df_normalized = df.copy() 
        # min-max normalization per column for current episode
        for col in df.columns.tolist():
            if col == 'time_since_reversal':
                df_normalized[col] = 2 * (df[col] / 78) - 1
                df_normalized[col] = norm_col.clip(-1, 1)
            else:
                min, max = self.minmax_dict[ticker][col]
                norm_col = 2 * ((df[col] - min) / (max - min)) - 1
                df_normalized[col] = norm_col.clip(-1, 1)

        return df_normalized

    def _trade_helper(self, t):
        action = self.states[t][self.action_idx]

        if t == 0:
            prev_action = 0
        else:
            prev_action = self.states[t-1][self.action_idx]

        if action == 0 and prev_action == 0:
            self.states[t][self.pnl_idx] = 0.0
            self.states[t][self.avl_cash_idx] = self.states[t-1][self.avl_cash_idx]

        # determine whether action is "buy short position (SB)" or "sell short position (SS)"
        elif action <= 0 and prev_action <= 0:
            if action <= prev_action:
                self._buy(action, prev_action, 'SB', t, reversal=False)
            elif action > prev_action:
                self._sell(action, prev_action, 'SS', t)

        # determine whether action is "sell long position (LS)" or "buy long position (LB)"
        elif action >= 0 and prev_action >= 0:
            if action < prev_action:
                self._sell(action, prev_action, 'LS', t)
            elif action >= prev_action:
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
            pnl = self.positions.add_position(abs(delta_a), 'L', market_price)

        elif flag == 'SB':                          # buy short position
            avl_cash -= new_shares * market_price
            current_shares = prev_shares - new_shares
            pnl = self.positions.add_position(abs(delta_a), 'S', market_price)
        
        # update info to finalize trade
        self.states[t][self.avl_cash_idx] = avl_cash
        self.states[t][self.shares_idx] = current_shares
        
        # add new position for future sale calculations
        self.states[t][self.pnl_idx] = pnl

    def _sell(self, action, prev_action, flag, t):
        prev_t = self._check_index(t)       # check t == 0
        
        delta_a = action - prev_action
        current_shares = self.states[t][self.shares_idx]
        prev_shares = self.states[prev_t][self.shares_idx]

        market_price = self.states[t][self.marketprice_idx]
        avl_cash = self.states[prev_t][self.avl_cash_idx]

        # calculate available cash and current share holdings for new trade
        if flag == 'LS':                                # sell long position
            new_shares, pnl = self.positions.sell_position(abs(delta_a), t, 'L', market_price) # retrieve correct amount of shares to sell depending on position price and delete the position
            avl_cash += new_shares * market_price
            current_shares = prev_shares - new_shares

        elif flag == 'SS':                              # sell short position
            new_shares, pnl = self.positions.sell_position(abs(delta_a), t, 'S', market_price)
            avl_cash += new_shares * market_price
            current_shares = prev_shares + new_shares
        
        # update info to finalize trade
        self.states[t][self.avl_cash_idx] = avl_cash
        self.states[t][self.shares_idx] = current_shares
        self.states[t][self.pnl_idx] = pnl
    
    def _check_index(self, t):
        if (t == 0):
            return t
        else:
            return t - 1



     

    

    