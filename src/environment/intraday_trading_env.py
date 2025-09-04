import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os
from pathlib import Path

# custom env class inherting from base gym class
class TradingEnv(gym.Env):

    # __init__() class constructor, required for gym
    def __init__(self, data_dir, tickers, model_path, validation, num, steps):
        super().__init__()

        # initialization
        self.num = num
        self.tickers = tickers
        self.data_dir = Path(data_dir)
        self.model_path = model_path
        self.validation = validation
        self.current_step = 0
        self.global_step = 0
        self.current_ep = 0
        self.ep_len = 0
        self.meta = None 
        self.window_size = 12
        self.features = 10
        self.init_cash = 50000
        self.total_ep_pnl = 0
        self.obs_buffer = []
        self.log_buffer = []
        
        # action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape = (1, ), dtype = np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape = (self.window_size, self.features), dtype = np.float32)

        # obs matrix index order (these values are computed after every timestep as they are related to preceding agent actions)
        self.o_action_idx = None
        self.o_pnltarget_idx = None

        # meta vector index order (the vector contains state transition info produced by the env at a single timestep)
        self.avl_cash_idx = 0
        self.marketprice_idx = 1
        self.action_idx = 2
        self.psize_idx = 3
        self.realized_pnl_idx = 4
        self.unrealized_pnl_idx = 5
        self.reward_idx = 6

        self.meta_len = self.reward_idx + 1                                                       
        self.flag_arr = []                                            # since flags are strings and not np.float32, they cannot be directly added to the numpy array 
        self.shares_arr = []                                          # since shares are ints and not np.float32                              

        # load ticker files and group episodes by date and ticker (each episode is a single trading day involving one ticker)
        self.ticker_file_paths = [f for f in self.data_dir.iterdir()]
        self.episode_df_list = []
        for file in self.data_dir.iterdir():
            df = pd.read_csv(file)
            
            if df["ticker"].iloc[0] not in self.tickers:
                continue
            
            for _, group in df.groupby("date"):
                self.episode_df_list.append(group)
        
        if (self.validation != True):
            np.random.shuffle(self.episode_df_list)

    # reset() function, called once per start of new episode, required for gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # initialization of new training episode
        self.positions = PositionLedger(self.init_cash)
        self.current_step = 0
        self.total_ep_pnl = 0
        t = self.current_step        

        # reset previous states/obs and compute transitions for new episode
        while True:
            self.meta = []
            self.flag_arr = []
            self.shares_arr = []
            self.obs_df = None
            
            # check whether episodes have ran out and shuffle the list to continue training
            if (self.current_ep == len(self.episode_df_list)):
                self.current_ep = 0
                if (self.validation != True):
                    np.random.shuffle(self.episode_df_list)

            # helper to initialize obs and info (the meta vector)
            self._window_helper(self.episode_df_list[self.current_ep])
            self.current_ep += 1
            
            # skip episodes that have fewer than window size amount of candles
            if len(self.obs_df) < self.window_size:
                print(f"[WARNING] Skipping empty episode index {self.current_ep}")
            else:
                break
            
        # return obs
        observation = self._get_obs(t)
        info = {}
        
        return observation, info      # return empty info, since info is handled internally
    
    # step() function, called once per state transtion to return the next state and reward (calls internal gym helpers), required for gyms
    def step(self, action):       
        t = self.current_step

        # clean and store action
        action = np.clip(action, -1, 1)
        action = np.round(action, 1)                                                    # round to one decimal to prevent minute changes in position size
        if action >= -0.4 and action <= 0.4:                                            # increase "dead zone" for selling/holding no position (outputting action = 0)
            action = 0 
        
        # store action as part of env info and obs (agent sees it's previous action for context regarding it's trading position)
        self.meta[t][self.action_idx] = action
        self.obs_matrix[t + self.window_size - 1, self.o_action_idx] = action

        # compute trade and retrieve reward
        self._trade_helper(t)
        reward = self.meta[t][self.reward_idx]

        # increment step
        self.current_step += 1
        self.global_step += 1
        observation = self._get_obs(t)
        truncated = False

        # check if episode has ended, then log results to CSV
        terminate = False
        info = {"episode_pnl": 0}
        if t == self.ep_len - 1:
            terminate = True
            
            # construct meta into dataframe
            meta_log_df = pd.DataFrame(self.meta, columns=[
                    "avl_cash", "market_price", "action", "position_size", "realized_pnl", "unrealized_pnl", "reward"
                ])
            meta_log_df.insert(0, "ticker", self.current_ep_ticker)
            meta_log_df.insert(0, "date", self.current_ep_date)
            meta_log_df.insert(0, "time", self.current_ep_timeseries)
            meta_log_df.insert(0, "flag", pd.Series(self.flag_arr))
            meta_log_df.insert(0, "shares", pd.Series(self.shares_arr))
            meta_log_df = meta_log_df[["ticker", "date", "time", "avl_cash", "shares", "position_size", "market_price", "action", "flag", "unrealized_pnl", "realized_pnl", "reward"]]
            
            # sum of episode pnl data used later for tensorboard logging                                                        
            info = {"episode_pnl": self.total_ep_pnl}

            # default file name when environment is being used for training
            obs_path = f"{self.model_path}-obs-t-{self.num}.csv"
            csv_path = f"{self.model_path}-training-{self.num}.csv"
            
            # round float values for presentation
            meta_log_df = np.round(meta_log_df, 3)
            self.obs_df = np.round(pd.DataFrame(self.obs_matrix, columns=self.obs_df.columns.tolist()), 2)
            self.log_buffer.append(meta_log_df)
            self.obs_buffer.append(self.obs_df)
            

            # change file name and logging frequency if running environment for validation of a model
            step = self.global_step
            if self.validation:
                obs_path = f"{self.model_path}-obs-v-{self.num}.csv"
                csv_path = f"{self.model_path}-validation-{self.num}.csv"
            
            # for training multiplied by 12 due to vec_env usage messing up the internal step count 
            else:
                step = self.global_step * 12
            
            # log buffer to CSV once every 10k steps
            if step >= 10000:                                                                            
                self._buffer_helper(csv_path, obs_path)
                self.global_step = 0
                  
        # return values for step()
        return observation, reward, terminate, truncated, info
    
    # helper to dump and reset buffer
    def _buffer_helper(self, csv_path, obs_path):
        logdf = pd.concat(self.log_buffer, ignore_index=True)
        obsdf = pd.concat(self.obs_buffer, ignore_index=True)
        logdf.to_csv(csv_path, mode="a", index=False, header=not os.path.exists(csv_path))
        obsdf.to_csv(obs_path, mode="a", index=False, header=not os.path.exists(obs_path))
        self.log_buffer = []
        self.obs_buffer = []

    # helper to initialize obs matrix and meta vector
    def _window_helper(self, episode_df):

        # store date for logging at episode end
        self.current_ep_ticker = episode_df["ticker"].iloc[0]
        self.current_ep_date = episode_df["date"].iloc[0]
        self.current_ep_timeseries = episode_df["time"].iloc[self.window_size - 1:].reset_index(drop=True)
        market_price = episode_df["market_price"]
        
        # drop unnecessary info and normalize data
        episode_df = episode_df.drop(columns = ["date", "time", "ticker", "market_price"])
        episode_df["action"] = 0.0
        episode_df["pnl_target"] = 0.0
        self.obs_df = episode_df

        # this only needs to run once per init
        if self.o_action_idx is None:
            self.o_action_idx = self.obs_df.columns.get_loc("action")
            self.o_pnltarget_idx = self.obs_df.columns.get_loc("pnl_target")
        
        # numpy array will allow for faster accessing than the dataframe
        self.obs_matrix = episode_df.to_numpy(dtype=np.float32)
        
        # compute and store meta (info) vector
        for i in range(len(episode_df) - self.window_size + 1):
            info = np.zeros(self.meta_len, dtype=np.float32)
            info[self.avl_cash_idx] = self.init_cash 
            info[self.marketprice_idx] = market_price.iloc[i + self.window_size - 1]        # price of last candle of current state where trades are taken
            self.flag_arr.append("_")
            self.shares_arr.append(0)
            self.meta.append(info)
        
        self.ep_len = len(self.meta)
        # np array containing np arrays 
        self.meta = np.array(self.meta, dtype=np.float32)                                   # info for CSV logging

    # returns the obs matrix at time step t (obs is a rolling window of candles of size window_size, with each t being an individual candle)
    def _get_obs(self, t):
        obs = self.obs_matrix[t : t + self.window_size]
        return obs

   # reward calculations for reversals are handled impliclitly
    def _calculate_reward(self, t):
        prev_t = self._check_index(t)
        
        curr_action = self.meta[t][self.action_idx]
        prev_action = self.meta[prev_t][self.action_idx]
        target = 0.003 * self.init_cash                                                                     # profit target: 0.3% of portfolio value per trade                         
        scaled_cost = (0.00005 * abs(self.meta[t][self.psize_idx])) / target                                # used for slippage and fees to be used as a penalty for overtrading and entering weak positions              
        
        # pnl
        unrealized_pnl = self.meta[t][self.unrealized_pnl_idx]
        realized_pnl = self.meta[t][self.realized_pnl_idx]
        relative_unrealized = unrealized_pnl / target
        relative_realized = realized_pnl / target

        reward = 0
        # update next timestep in obs with the active positions future pnl data at t+1
        if t != self.ep_len - 1:
            self.obs_matrix[t + self.window_size, self.o_pnltarget_idx] = self.meta[t+1][self.unrealized_pnl_idx] / target
        
        # no change in position status (whether holding same position or no position)
        if curr_action == prev_action:
            # holding no position
            if curr_action == 0.0:
                price = self.meta[t][self.marketprice_idx]
                prev_price = self.meta[prev_t][self.marketprice_idx]
                change = (price - prev_price) / prev_price
                
                # reward holding during poor price movement
                if abs(change) < 0.0005:
                    reward = 0.0001
                
                # penalize for holding out of trading too much
                else:
                    reward -= abs(change) * 5
                
            # holding same position (no change in delta_a)
            else:
                # if held position is growing in value
                if relative_unrealized > 0 and relative_unrealized < 0.8:
                    reward = relative_unrealized * 2

                # if held position has exceeded a certain amount of profit growth (incentivise agent to close positions at a certain point)
                elif relative_unrealized >= 0.80:
                    reward = -0.0001  
                
                # if held position has lost value
                else:
                    # high loss in value
                    if relative_unrealized <= -1.5:
                        reward = relative_unrealized * 1.25
                    
                    # threshold/tolerable loss
                    else:
                        reward = -0.0001      
        # sold position
        else:
            flag = self.flag_arr[t]
            if flag in ["LS", "SS"]:
                # profit
                if relative_realized > 0:
                    if relative_realized > 0.7:
                        reward = relative_realized * 3
                    elif relative_realized < 0.7 and relative_realized > 0.35:
                        reward = relative_realized * 2
                    else:
                        reward = relative_realized * 1.15
                # loss
                else:
                    if relative_realized <= -1.5:
                        reward = relative_realized * 1.5
                    else:
                        reward = relative_realized * 0.9
                
                # slippage cost
                reward -= scaled_cost
            
            # penalize reversals
            elif flag in ["LR", "SR"]:
                reward = -3
            
            # entered a new position
            else:
                # transaction cost 
                reward -= scaled_cost
        
        # add new reward at current timestep
        self.meta[t][self.reward_idx] += reward

    def _trade_helper(self, t):
        action = self.meta[t][self.action_idx]

        # edge case, t == 0 first timestep
        if t == 0:
            prev_action = 0
        else:
            prev_action = self.meta[t-1][self.action_idx]

        # holding no position
        if action == 0 and prev_action == 0:
            self.meta[t][self.realized_pnl_idx] = 0.0
            self.meta[t][self.avl_cash_idx] = self.meta[t-1][self.avl_cash_idx]
            self.flag_arr[t] = "NP"

        # determine whether action is "buy short position (SB)" or "sell short position (SS)"
        elif action <= 0 and prev_action <= 0:
            if action <= prev_action:
                self._buy(action, prev_action, "SB", t, reversal=False)
            elif action > prev_action:
                self._sell(action, prev_action, "SS", t)

        # determine whether action is "sell long position (LS)" or "buy long position (LB)"
        elif action >= 0 and prev_action >= 0:
            if action < prev_action:
                self._sell(action, prev_action, "LS", t)
            elif action >= prev_action:
                self._buy(action, prev_action, "LB", t, reversal=False)

        # determine whether action is "short reversal-> sell short position + buy new long position" 
        # or "long reversal-> sell long position + buy new short position"
        else:
            if action > 0 and prev_action < 0:
                self._sell(0, prev_action, "SS", t)
                self._calculate_reward(t)                                                          # additional call is necessary for reversals since it's technically two trading actions
                self._buy(action, 0, "LB", t, reversal=True)
                self.flag_arr[t] = "SR"
            else:
                self._sell(0, prev_action, "LS", t)
                self._calculate_reward(t)
                self._buy(action, 0, "SB", t, reversal=True)
                self.flag_arr[t] = "LR"
        
        # compute reward for step()
        self._calculate_reward(t)
        
    def _buy(self, action, prev_action, flag, t, reversal):
        # check t == 0
        prev_t = self._check_index(t)                                                              

        # change in action (delta) used to determine shares for a position
        delta_a = action - prev_action                                                             
        market_price = self.meta[t][self.marketprice_idx]

        # get future price for unrealized PnL calculation
        future_price = market_price
        if t != self.ep_len - 1:
            future_price = self.meta[t+1][self.marketprice_idx]
        
        current_size = 0
        if delta_a != 0:
            if (reversal):
                # if this operation is part of second leg of reversal, then use the current timestep instead to access info for calculations  
                avl_cash = self.meta[t][self.avl_cash_idx]
                prev_shares = 0
                prev_size = 0
            else:
                # else initialize as normal
                avl_cash = self.meta[prev_t][self.avl_cash_idx]
                prev_shares = int(self.shares_arr[prev_t])
                prev_size = self.meta[prev_t][self.psize_idx]

            # calculate shares and pnl
            if flag == "LB":            # buy long position
                new_shares, position_size = self.positions.add_position(abs(delta_a), "L", market_price)
                current_shares = prev_shares + new_shares
                current_size = prev_size + position_size 

            elif flag == "SB":          # buy short position
                new_shares, position_size = self.positions.add_position(abs(delta_a), "S", market_price)
                current_shares = prev_shares - new_shares
                current_size = prev_size - position_size                    
            
            # update avl cash
            avl_cash -= position_size
        
        # if holding same position as previous timestep (no change in action)
        else:
            current_shares, position_size = self.positions.hold_position()
            avl_cash = self.meta[prev_t][self.avl_cash_idx]
            flag = "HP"

        # update info
        self.meta[t][self.avl_cash_idx] = avl_cash                       
        self.meta[t][self.psize_idx] = current_size
        if t != self.ep_len - 1:
            self.meta[t+1][self.unrealized_pnl_idx] = self.positions.get_unrealized_pnl(future_price)
        self.flag_arr[t] = flag
        self.shares_arr[t] = int(current_shares)  

    def _sell(self, action, prev_action, flag, t):
        # check t == 0 and correct it
        prev_t = self._check_index(t)                                                           
        
        # change in action (delta) used to determine shares for a position
        delta_a = action - prev_action
        market_price = self.meta[t][self.marketprice_idx]
        avl_cash = self.meta[prev_t][self.avl_cash_idx]
        prev_shares = int(self.shares_arr[prev_t])
        prev_size = self.meta[prev_t][self.psize_idx]

        # get future price for unrealized PnL calculation
        future_price = market_price
        if t != self.ep_len - 1:
            future_price = self.meta[t+1][self.marketprice_idx]

        current_size = 0
        # calculate shares and pnl
        if flag == "LS":                # sell long position                                                                         
            new_shares, realized_pnl, position_size = self.positions.sell_position(abs(delta_a), "L", market_price)
            current_shares = prev_shares - new_shares
            current_size = prev_size - position_size
            self.total_ep_pnl += realized_pnl

        elif flag == "SS":              # sell short position
            new_shares, realized_pnl, position_size = self.positions.sell_position(abs(delta_a), "S", market_price)          # helper returns positive shares
            current_shares = prev_shares + new_shares
            current_size = prev_size + position_size
            self.total_ep_pnl += realized_pnl
        
        # update avl_cash
        avl_cash += position_size

        # update info
        self.meta[t][self.realized_pnl_idx] = realized_pnl
        self.meta[t][self.unrealized_pnl_idx] = self.positions.get_unrealized_pnl(market_price)
        if t != self.ep_len - 1:
            self.meta[t+1][self.unrealized_pnl_idx] = self.positions.get_unrealized_pnl(future_price)                       # add unrealized_pnl calculation for next timestep
        self.meta[t][self.avl_cash_idx] = avl_cash
        self.meta[t][self.psize_idx] = current_size
        self.shares_arr[t] = int(current_shares)
        self.flag_arr[t] = flag
    
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
    
    # if the whole position size amount of shares is not to be sold
    def partial_sell(self, delta, market_price):
        output_shares = (delta * self.cash) / self.position_price
        output_size = output_shares * self.position_price
        
        realized_pnl = 0
        if (self.flag == "L"):
            realized_pnl = (market_price - self.position_price) * output_shares
        elif (self.flag == "S"):
            realized_pnl = -1 * ((market_price - self.position_price) * output_shares)
        
        self.delta = np.round((self.delta - delta), 1)
        self.shares -= int(output_shares)
        self.size -= int(output_size)

        return int(output_shares), realized_pnl, output_size
    
    # helper to calculate pnl (different calculation for long or short positions)
    def get_pnl(self, market_price):
        if (self.flag == "L"):
            return (market_price - self.position_price) * self.shares
        elif (self.flag == "S"):
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
    
    def hold_position(self):
        total_shares = 0
        total_size = 0
        
        for position in self.ledger:
            total_shares += position.shares
            total_size += position.size
            
        return int(total_shares), total_size
    
    # open new position and return shares, pnl, and size
    def add_position(self, delta, flag, current_price):
        delta = np.round(abs(delta), 1)     
     
        if delta != 0: 
            node = PositionNode(delta, self.cash, current_price, flag)
            self._push(node)

            return int(node.shares), node.size

    # close position and return shares, pnl, and size 
    def sell_position(self, delta, flag, market_price):
        total_shares = 0
        realized_pnl = 0
        total_size = 0
        
        delta = np.round(abs(delta), 1)
        top = self._pop() 

        if top is not None:
            remaining = np.round(np.float64(top.delta - delta), 1)
            if remaining < 0:
                total_shares += top.shares
                realized_pnl += top.get_pnl(market_price)
                total_size += top.size
                
                shares, current_pnl, size = self.sell_position(abs(remaining), flag, market_price)
                total_shares += shares
                realized_pnl += current_pnl
                total_size += size
            
            elif remaining > 0:
                shares, current_pnl, size = top.partial_sell(abs(delta), market_price)
                total_shares += shares
                realized_pnl += current_pnl
                total_size += size
                self._push(top)
            
            else:
                total_shares += top.shares
                realized_pnl += top.get_pnl(market_price)
                total_size += top.size

        else:
            return 0.0, 0.0, 0.0, 0.0

        return int(total_shares), realized_pnl, total_size
    
    def get_unrealized_pnl(self, future_price):
        unrealized_pnl = 0.0
        if len(self.ledger) == 0:
            return 0.0
        else:
            for position in self.ledger:
                if position is not None:
                    unrealized_pnl += position.get_pnl(future_price)
                else:
                    unrealized_pnl += 0.0
            
            return unrealized_pnl
        
        
    


     

    

    