import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / 'data' / 'processed_tickers_5-minute'

# custom env class inherting from base gym class
class TradingEnv(gym.Env):

    # __init__() class constructor, required for gym
    def __init__(self):
        super().__init__()

        # initialization
        self.start_time = datetime.now().strftime("%m%d-%H%M")
        self.current_step = 0
        self.global_step = 0
        self.current_episode = 0
        self.states = None 
        self.window_size = 20
        self.features = 10
        self.init_cash = 25000
        
        # action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape = (1, ), dtype = np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape = (self.window_size, 10), dtype = np.float32)

        # meta vector index order (the vector contains state transition info produced by the env at a single timestep)
        self.avl_cash_idx = 0
        self.marketprice_idx = 1
        self.action_idx = 2
        self.p_size_idx = 3
        self.pnl_idx = 4
        self.reward_idx = 5

        self.meta_len = 6                                                       # increment by 1 if adding an index
        self.flag_arr = []                                                      # since flags are strings they can't be added to the initial np array used to create meta
        self.shares_arr = []

        # load ticker files and group episodes by date and ticker (each episode is a single trading day involving one ticker)
        self.ticker_file_paths = [f for f in data_dir.iterdir()]
        self.episode_df_list = []
        for file in data_dir.iterdir():
            df = pd.read_csv(file)
            
            if df['ticker'].iloc[0] not in ['AAPL', 'MSFT']:
                continue
            
            for _, group in df.groupby('date'):
                self.episode_df_list.append(group)
        
        np.random.shuffle(self.episode_df_list)

    # reset() function, called once per start of new episode, required for gyms
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
            if (self.current_episode == len(self.episode_df_list)):
                self.current_episode = 0
                np.random.shuffle(self.episode_df_list)

            self._window_helper(self.episode_df_list[self.current_episode])
            self.current_episode += 1
            
            if len(self.obs) > 20:
                break
            print(f'[WARNING] Skipping empty episode index {self.current_episode}')
            
        # return normalized obs
        observation = self._get_obs(t)
        
        return observation, {}      # return empty info, handled internally
    
    # step() function, called once per state transtion to return the next state and reward (calls internal gym helpers), required for gyms
    def step(self, action):       
        t = self.current_step

        # clean and store action
        action = np.clip(action, -1, 1)
        action = np.round(action, 1)
        if action >= -0.1 and action <= 0.1:
            action = 0 

        self.states[t][self.action_idx] = action
        self.obs[t][-1] = action                                                    # update obs to include current action (so that it is available for the next)

        # compute trade and retrieve reward
        self._trade_helper(t)
        reward = self.states[t][self.reward_idx]

        # increment step
        self.current_step += 1
        self.global_step += 1
        observation = self._get_obs(t)
        truncated = False

        # check if episode has ended, then log results to csv
        terminate = False
        if t == len(self.states)-1:
            terminate = True
            meta_log_df = pd.DataFrame(self.states, columns=[
                    'avl_cash', 'market_price', 'action', 'position_size', 'pnl', 'reward'
                ])
            meta_log_df.insert(0, 'ticker', self.current_ep_ticker)
            meta_log_df.insert(0, 'date', self.current_ep_date)
            meta_log_df.insert(len(meta_log_df.columns), 'flag', pd.Series(self.flag_arr))
            meta_log_df.insert(len(meta_log_df.columns), 'shares', pd.Series(self.shares_arr))

            ep_pnl = meta_log_df['pnl'].sum()
            info = {'episode_pnl': ep_pnl}
            
            csv_path = os.path.join('logs', 'trades-' + self.start_time + '.csv')
            meta_log_df = np.round(meta_log_df, 2)
            meta_log_df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
        else:
            info = {}
        
        return observation, reward, terminate, truncated, info
    
    def _window_helper(self, episode_df):
        # store date for logging at episode end
        self.current_ep_ticker = episode_df['ticker'].iloc[0]
        self.current_ep_date = episode_df['date'].iloc[0]
        market_price = episode_df['market_price']
        
        # drop unnecessary info and normalize data
        episode_df = episode_df.drop(columns = ['date', 'ticker', 'market_price'])
        episode_df['p_size'] = 0
        episode_df['action'] = 0
        
        # compute and store obs and meta (info) vector
        for i in range(len(episode_df) - self.window_size + 1):
            obs_window = episode_df.iloc[i:i+self.window_size]                          # 20 x 5 min candle obs with 8 + 2(below) features each
            obs_window.to_numpy(dtype=np.float32) 
            self.obs.append(obs_window)

            meta = np.zeros(self.meta_len, dtype=np.float32)
            meta[self.avl_cash_idx] = self.init_cash 
            meta[self.marketprice_idx] = market_price.iloc[i+self.window_size-1]        # price of last candle of current state where trades are taken
            self.flag_arr.append('H')
            self.shares_arr.append(0)
            self.states.append(meta)
        
        # np array containing np arrays
        self.states = np.array(self.states, dtype=np.float32)                   # info for logging
        self.obs = np.array(self.obs, dtype=np.float32)                         # obs for training 

    # returns the obs matrix at time step t
    def _get_obs(self, t):
        return self.obs[t]

   # reward calculations for reversals are handled impliclitly
    def _calculate_reward(self, t):
        reward = 0
        if t != 0:
            pnl = self.states[t][self.pnl_idx]
            curr_action = self.states[t][self.action_idx]
            prev_action = self.states[t-1][self.action_idx]
            target = 0.005 * self.init_cash                         # profit target: 0.5% of portfolio value                          
            scaled_cost = (0.0005 * self.init_cash) / target       # 0.05% of position size, used for slippage and fees              
            relative = pnl / target

            if curr_action == prev_action:
                # holding no position
                if curr_action == 0:
                    reward = 0.075
                
                # holding same position (no change in delta_a)
                else:
                    if pnl > 0:
                        reward = relative * 1.25   
                    else:
                        if relative <= -1:
                            reward = relative * 1.5
                        else:
                            reward = relative      
            else:
                flag = self.flag_arr[t]
                
                # sold position
                relative *= 2
                if flag not in ['LB', 'SB']:
                    # profit
                    if pnl > 0:
                        reward = relative
                    # loss
                    else:
                        if relative <= -1:
                            reward = relative * 1.5 
                        else:
                            reward = relative
                    
                    # slippage cost
                    reward -= scaled_cost

                # bought position
                else:
                    # transaction cost 
                    reward -= scaled_cost

        self.states[t][self.reward_idx] += reward * 3

    def _trade_helper(self, t):
        action = self.states[t][self.action_idx]

        if t == 0:
            prev_action = 0
        else:
            prev_action = self.states[t-1][self.action_idx]

        if action == 0 and prev_action == 0:
            self.states[t][self.pnl_idx] = 0.0
            self.states[t][self.avl_cash_idx] = self.states[t-1][self.avl_cash_idx]
            flag = 'H'

        # determine whether action is 'buy short position (SB)' or 'sell short position (SS)'
        elif action <= 0 and prev_action <= 0:
            if action <= prev_action:
                self._buy(action, prev_action, 'SB', t, reversal=False)
                flag = 'SB'
            elif action > prev_action:
                self._sell(action, prev_action, 'SS', t)
                flag = 'SS'

        # determine whether action is 'sell long position (LS)' or 'buy long position (LB)'
        elif action >= 0 and prev_action >= 0:
            if action < prev_action:
                self._sell(action, prev_action, 'LS', t)
                flag = 'LS'
            elif action >= prev_action:
                self._buy(action, prev_action, 'LB', t, reversal=False)
                flag = 'LB'

        # determine whether action is 'short reversal-> sell short position + buy new long position' 
        # or 'long reversal-> sell long position + buy new short position'
        else:
            if action > 0 and prev_action < 0:
                self._sell(0, prev_action, 'SS', t)
                self._calculate_reward(t)                                                          # additional call is necessary for reversals since it's technically two trading actions
                self._buy(action, 0, 'LB', t, reversal=True)
                flag = 'SR'
            else:
                self._sell(0, prev_action, 'LS', t)
                self._calculate_reward(t)
                self._buy(action, 0, 'SB', t, reversal=True)
                flag = 'LR'
        
        self.flag_arr[t] = flag
        # compute reward for step()
        self._calculate_reward(t)
        
    def _buy(self, action, prev_action, flag, t, reversal):
        prev_t = self._check_index(t)                                                              # check t == 0

        delta_a = action - prev_action                                                             # change in action (delta) used to determine shares for a position
        market_price = self.states[t][self.marketprice_idx]
        
        current_size = 0
        multiplier = 1

        if delta_a != 0:
            pnl = 0
            if (reversal):
                # if this operation is part of second leg of reversal, then use the current timestep instead to access info for calculations  
                avl_cash = self.states[t][self.avl_cash_idx]
                prev_shares = 0
                prev_size = 0
            else:
                # else initialize as normal
                avl_cash = self.states[prev_t][self.avl_cash_idx]
                prev_shares = int(self.shares_arr[prev_t])
                prev_size = self.states[prev_t][self.p_size_idx]

            # calculate shares and pnl
            if flag == 'LB':                                                                                            # buy long position
                new_shares, position_size = self.positions.add_position(abs(delta_a), 'L', market_price)
                current_shares = prev_shares + new_shares
                current_size = prev_size + position_size 

            elif flag == 'SB':                                                                                          # buy short position
                new_shares, position_size = self.positions.add_position(abs(delta_a), 'S', market_price)
                current_shares = prev_shares - new_shares
                current_size = prev_size - position_size
                multiplier = -1                        
            
            avl_cash -= position_size
        
        else:
            current_shares, pnl, position_size = self.positions.hold_position(market_price)
            avl_cash = self.states[prev_t][self.avl_cash_idx]
        
        self.states[t][self.pnl_idx] += pnl
        self.states[t][self.avl_cash_idx] = avl_cash
        self.shares_arr[t] = int(current_shares)
        self.obs[t][-2] = current_size / self.init_cash * multiplier                                                   # multiplier helps distinguish between long and short (negative) 
        self.states[t][self.p_size_idx] = current_size

    def _sell(self, action, prev_action, flag, t):
        prev_t = self._check_index(t)                                                           # check t == 0 and corrects t if it is
        
        delta_a = action - prev_action
        market_price = self.states[t][self.marketprice_idx]
        avl_cash = self.states[prev_t][self.avl_cash_idx]
        prev_shares = int(self.shares_arr[prev_t])
        prev_size = self.states[prev_t][self.p_size_idx]

        current_size = 0
        multiplier = 1

        # calculate shares and pnl
        if flag == 'LS':                                                                            
            new_shares, pnl, position_size = self.positions.sell_position(abs(delta_a), 'L', market_price)
            current_shares = prev_shares - new_shares
            current_size = prev_size - position_size

        elif flag == 'SS':
            new_shares, pnl, position_size = self.positions.sell_position(abs(delta_a), 'S', market_price)          # helper returns positive shares
            current_shares = prev_shares + new_shares
            current_size = prev_size + position_size
            multiplier = -1
        
        # update position size and avl_cash
        avl_cash += position_size

        # update info to finalize trade
        self.states[t][self.pnl_idx] += pnl
        self.states[t][self.avl_cash_idx] = avl_cash
        self.shares_arr[t] = int(current_shares)
        self.obs[t][-2] = (current_size / self.init_cash) * multiplier                                             # multiplier helps distinguish between long and short (negative)
        self.states[t][self.p_size_idx] = current_size
    
    # helper for t == 0 edge case (out of bounds error at start of new episode)
    def _check_index(self, t):
        if (t == 0):
            return t
        else:
            return t - 1

# data object class to store trade position related attributes
class PositionNode:
    # position data
    def __init__(self, delta, cash, position_price, flag):
        self.delta = delta
        self.cash = cash
        self.position_price = position_price
        self.shares = int((delta * cash) / position_price)
        self.size = self.shares * position_price
        self.flag = flag
    
    def partial_sell(self, delta, market_price):
        output_shares = (delta * self.cash) / self.position_price
        output_size = output_shares * self.position_price
        
        output_pnl = 0
        if (self.flag == 'L'):
            output_pnl = (market_price - self.position_price) * output_shares
        elif (self.flag == 'S'):
            output_pnl = -1 * ((market_price - self.position_price) * output_shares)
        
        self.delta = np.round((self.delta - delta), 1)
        self.shares -= int(output_shares)
        self.size -= int(output_size)

        return int(output_shares), output_pnl, output_size

    def get_pnl(self, market_price):
        if (self.flag == 'L'):
            return (market_price - self.position_price) * self.shares
        elif (self.flag == 'S'):
            return -1 * (market_price - self.position_price) * self.shares

# helper class to abstract position tracking and managemement (LIFO structure), used to return appropriate shares and pnl after trading actions
class PositionLedger:
    def __init__(self, init_cash):
        self.ledger = []
        self.cash = init_cash
    
    # pop top of stack
    def _pop(self):
        if len(self.ledger) != 0:
            return self.ledger.pop()
        else:
            None
    
    # push to top of stack
    def _push(self, node):
        self.ledger.append(node)
    
    def hold_position(self, market_price):
        total_pnl = 0
        total_shares = 0
        total_size = 0

        for position in self.ledger:
            total_pnl += position.get_pnl(market_price)
            total_shares += position.shares
            total_size += position.size
            
        return int(total_shares), total_pnl, total_size
    
    # open new position and return shares, pnl, and size
    def add_position(self, delta, flag, market_price):
        delta = np.round(abs(delta), 1)     
     
        if delta != 0: 
            node = PositionNode(delta, self.cash, market_price, flag)
            self._push(node)
            
            return int(node.shares), node.size

    # close position and return shares, pnl, and size 
    def sell_position(self, delta, flag, market_price):
        total_shares = 0
        total_pnl = 0
        total_size = 0
        
        delta = np.round(abs(delta), 1)
        top = self._pop() 

        if top is not None:
            remaining = np.round(np.float64(top.delta - delta), 1)
            if remaining < 0:
                total_shares += top.shares
                total_pnl += top.get_pnl(market_price)
                total_size += top.size
                
                shares, pnl, size = self.sell_position(abs(remaining), flag, market_price)
                total_shares += shares
                total_pnl += pnl
                total_size += size
            
            elif remaining > 0:
                shares, pnl, size = top.partial_sell(abs(delta), market_price)
                total_shares += shares
                total_pnl += pnl
                total_size += size
                self._push(top)
            
            else:
                total_shares += top.shares
                total_pnl += top.get_pnl(market_price)
                total_size += top.size
            
        else:
            return 0.0, 0.0, 0.0

        return int(total_shares), total_pnl, total_size
        
    


     

    

    