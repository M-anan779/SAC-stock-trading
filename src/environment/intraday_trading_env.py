import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

# custom env class inherting from base gym class
class TradingEnv(gym.Env):
    # __init__() class constructor, required for gym
    def __init__(self, tickers, episode_list, model_path, validation, n_envs, num, total_steps):
        super().__init__()

        # initialization
        self.n_envs = n_envs                        # number of envs being used (account for logging frequency)
        self.num = num                              # run id to save files with the correct naming
        self.tickers = tickers                      # tickers being used as part of data set
        self.model_path = model_path                # model save path
        self.validation = validation                # validation run or not (different file save and data set used)
        self.episode_df_list = episode_list         # list of episodes to be used (no random shuffling for validation runs)
        self.total_steps = total_steps              # total number of training/validation steps used for finishing logging

        # environment state variables
        self.current_step = 0
        self.global_step = 0
        self.steps_since_lastlog = 0
        self.current_ep = 0
        self.ep_len = 0
        self.meta = None 
        self.window_size = 30
        self.features = 9
        self.init_cash = 50000
        self.total_ep_pnl = 0
        self.obs_buffer = []                        # buffer to limit CSV write outs
        self.log_buffer = []
        self.p_streak = 0                           # helps keep track of good action streaks for incremental step wise rewards
        self.l_streak = 0
        self.total_p_streak = 0
        self.total_l_streak = 0
        
        # action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape = (1, ), dtype = np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape = (self.window_size, self.features), dtype = np.float32)

        # obs matrix index order (these values are computed after every timestep as they are related to preceding agent actions)
        self.o_action_idx = None
        self.o_pnltarget_idx = None

        # meta vector index order (the vector contains state transition info produced by the env at a single timestep)
        self.avl_cash_idx = 0
        self.marketprice_idx = 1
        self.psize_idx = 2
        self.realized_pnl_idx = 3
        self.unrealized_pnl_idx = 4
        self.reward_idx = 5

        self.meta_len = self.reward_idx + 1                                                       
        self.flag_arr = []                                            # since flags are strings and not np.float32, they cannot be directly added to the numpy array 
        self.shares_arr = []                                          # since shares are ints and not np.float32                              

        if (self.validation != True):
            np.random.shuffle(self.episode_df_list)

    # reset() function, called once per start of new episode, required for gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # initialization of new training episode
        self.positions = PositionLedger()
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
        prev_t = self._check_index(t)

        # clean and store action
        action = np.clip(action, -1, 1)
        action = np.round(action, 3)                                                    # round to one decimal to prevent minute changes in position size
        
        action = int(np.round(abs(action) * 1000))
        action = action % 2

        prev_action = self.flag_arr[prev_t]
        # compute trade
        # long
        if action == 0:     
            if prev_action in ["NP", "LS"]:
                self._buy("LB", t)
            
            elif prev_action in ["LB", "HP"] :
                self._buy("HP", t)
        # sell
        elif action == 1:
            if prev_action in  ["LB", "HP"]:
                self._sell("LS", t)
            
            elif prev_action in ["NP", "LS"]:
                self.meta[t][self.realized_pnl_idx] = 0.0
                self.meta[t][self.avl_cash_idx] = self.meta[prev_t][self.avl_cash_idx]
                self.shares_arr[t] = 0
                self.flag_arr[t] = "NP"
             
        # store action as part of env info and obs (agent sees it's previous action for context regarding it's trading position)
        self.obs_matrix[t + self.window_size - 1, self.o_action_idx] = action

        # get reward
        self._calculate_reward(t)
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
                    "avl_cash", "market_price", "position_size", "realized_pnl", "unrealized_pnl", "reward"
                ])
            meta_log_df.insert(0, "ticker", self.current_ep_ticker)
            meta_log_df.insert(0, "date", self.current_ep_date)
            meta_log_df.insert(0, "time", self.current_ep_timeseries)
            meta_log_df.insert(0, "flag", pd.Series(self.flag_arr))
            meta_log_df.insert(0, "shares", pd.Series(self.shares_arr))
            meta_log_df = meta_log_df[["ticker", "date", "time", "avl_cash", "position_size", "shares", "market_price", "unrealized_pnl", "realized_pnl", "reward", "flag"]]
            
            # sum of episode pnl data used later for tensorboard logging                                                        
            info = {"episode_pnl": self.total_ep_pnl}

            # default file name when environment is being used for training
            obs_path = f"{self.model_path}-obs-t-{self.num}.csv"
            csv_path = f"{self.model_path}-training-{self.num}.csv"
            
            # round float values for presentation
            meta_log_df = np.round(meta_log_df, 5)
            self.obs_df = np.round(pd.DataFrame(self.obs_matrix, columns=self.obs_df.columns.tolist()), 2)
            self.log_buffer.append(meta_log_df)
            self.obs_buffer.append(self.obs_df)
            

            # change file name and logging frequency if running environment for validation of a model
            if self.validation:
                obs_path = f"{self.model_path}-obs-v-{self.num}.csv"
                csv_path = f"{self.model_path}-validation-{self.num}.csv"
            
            # multiply step count by number of environments which are stepping in parallel
            self.steps_since_lastlog = self.steps_since_lastlog * self.n_envs                                               
            # log buffer to CSV once every 10k steps
            if self.steps_since_lastlog >= 10000:                                                                            
                self._buffer_helper(csv_path, obs_path)
                self.steps_since_lastlog = 0
            # log any remainders near end of training
            elif self.total_steps - self.global_step <= 10000:
                self._buffer_helper(csv_path, obs_path)
            self.steps_since_lastlog += self.current_step                                                           # update logging counter
                  
        # return values for step()
        return observation, reward, terminate, truncated, info
    
    # helper to dump buffer into CSV and then reset the buffer
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
            self.flag_arr.append("NP")
            self.shares_arr.append(0)
            self.meta.append(info)
        
        self.ep_len = len(self.meta)
        # np array containing np arrays 
        self.meta = np.array(self.meta, dtype=np.float32)                                   # info for CSV logging

    # returns the obs matrix at time step t (obs is a rolling window of candles of size window_size, with each t being an individual candle)
    def _get_obs(self, t):
        obs = self.obs_matrix[t : t + self.window_size]
        return obs

    # main reward shaping logic
    # reward uses temporal feedback (long profitable streaks vs chop due to volatility), PnL feedback (profits outweigh)
    def _calculate_reward(self, t):
        curr_action = self.flag_arr[t]
        profit_target = 200                                                                     # profit target per trade                        
        loss_target = profit_target / 2                                                         # loss target always set to enforce a 2:1 profit to loss ratio
        scaled_cost = (0.000125 * abs(self.meta[t][self.psize_idx])) / profit_target            # slippage and transaction fees calculated relative to profit target (% of profit deduction)            
        unrealized_pnl = self.meta[t][self.unrealized_pnl_idx]                                  # raw unrealized pnl of position if position was held unchanged at current timestep
        realized_pnl = self.meta[t][self.realized_pnl_idx]                                      # raw realized pnl if position was old at current timestep

        # normalize unrealized + realized pnl as a profit or loss factor
        relative_realized = 0
        relative_unrealized = 0
        if realized_pnl > 0:
            relative_realized = (realized_pnl / profit_target)
        else:
            relative_realized = (realized_pnl / loss_target)
        if unrealized_pnl > 0:
            relative_unrealized = (unrealized_pnl / profit_target)
        else:
            relative_unrealized = (unrealized_pnl / loss_target)
        
        # market price change over N = 15 average price
        window = 15        
        lookback = min(window, t + 1)                                                           # adjust to look back near start of ep 
        recent_prices = [self.meta[t - i][self.marketprice_idx] for i in range(lookback)]
        price_range = (max(recent_prices) - min(recent_prices)) / np.mean(recent_prices)

        reward = 0

        # update next timestep in obs with the active positions future pnl data at t+1
        if t != self.ep_len - 1:
            future_pnl = self.meta[t+1][self.unrealized_pnl_idx]
            if future_pnl < 0:
                relative_pnl = future_pnl / loss_target
            else:
                relative_pnl = future_pnl / profit_target
            self.obs_matrix[t + self.window_size, self.o_pnltarget_idx] =  relative_pnl
       
        # CASE 0: holding no position
        if curr_action == "NP":
            # CASE 0A: minimize negative reward for no trading once tolerable total loss has been reached
            if self.total_ep_pnl <= loss_target:
                reward = -0.001

            # CASE 0B: trading during poor price movement
            elif price_range <= 0.003:
                if self.total_ep_pnl <= profit_target:
                    reward = 0.001
                
                # larger reward for avoiding once profit target has been reached
                else:
                    reward = 0.01
        
            # CASE 0C: no position during large price movement
            elif price_range >= 0.006:
                # penalize no trading when daily profit target has not been reached
                if self.total_ep_pnl <= profit_target:
                    reward = -0.15
                
                # allow avoiding risky trading as an option once profit target has been reached
                else:
                    reward = 0.1
                
        # CASE 1: holding same position 
        elif curr_action == "HP":
            
            # CASE 1A: position is growing in value
            if relative_unrealized >= 0 and relative_unrealized <= 1:
                self.total_p_streak += 1
                self.p_streak += 1
                self.l_streak = 0
                
                reward = (self.p_streak / window) * relative_unrealized

            # CASE 1B: position has exceeded desired growth (incentivise agent to close positions)
            elif relative_unrealized >= 1:
                self.total_p_streak += 1
                self.p_streak += 1
                self.l_streak = 0
                
                reward = 0.1 * relative_unrealized
            
            # CASE 1C: position has lost value
            elif relative_unrealized < 0:
                self.total_l_streak += 1
                self.l_streak += 1
                self.p_streak = 0

                # threshold/tolerable loss
                if relative_unrealized < 0 and relative_unrealized >= -1:
                    reward = 0.2 * relative_unrealized
                
                # high loss in value
                elif relative_unrealized < -1:
                    reward = (self.l_streak / window) * relative_unrealized 
            
            # multipliers for preferring certain regimes
            # low volatility
            if  price_range <= 0.003:
                reward *= 0.8
            
            # high volatility
            elif price_range >= 0.0075:
                reward *= 0.6
            
            # medium volatility
            else:
                reward *= 1.5

        # CASE 2: sold position
        elif curr_action in ["LS", "SS"]:
            
            # CASE 2A: profit
            if relative_realized > 0:
                self.total_p_streak += 1
                total_p_streak = self.total_p_streak / window
                total_l_streak = self.total_l_streak / window

                # high profit
                if relative_realized >= 1:
                    reward = abs(total_p_streak - total_l_streak) * relative_realized * 2
                
                # expected profit
                elif relative_realized < 1 and relative_realized >= 0.5:
                    reward = abs(total_p_streak - total_l_streak) * relative_realized * 1.5
                
                # low profit
                else:
                    reward = abs(total_p_streak - total_l_streak) * relative_realized * 1
            
            # CASE 2B: loss
            else:
                self.total_l_streak += 1
                total_l_streak = self.total_l_streak / window
                total_p_streak = self.total_p_streak / window                                   
                
                # high loss
                if relative_realized <= -1:
                    reward = abs(total_p_streak - total_l_streak) * relative_realized * 0.5
                
                # tolerable loss
                else:
                    reward = abs(total_p_streak - total_l_streak) * relative_realized * 0.25
            
            # reset variables
            self.p_streak = 0
            self.l_streak = 0
            self.total_p_streak = 0
            self.total_l_streak = 0

            # slippage cost
            reward -= scaled_cost
            
        # CASE 3: bought position
        # negative rewards implicitly require the agent to secure higher profit to get a better cumulative rewards
        elif curr_action in ["LB", "SB"]:
            
            # CASE 3A: trading during poor price movement
            if  price_range <= 0.003:
                reward = scaled_cost * -2
            
            # CASE 3B: trading during volatile price movement
            elif price_range >= 0.0075:
                reward = scaled_cost * -3
            
            # CASE 3C: trading during moderate price movement
            else:
                reward -= scaled_cost
        
        # OUTPUT: new reward at current timestep
        self.meta[t][self.reward_idx] += reward 
        
    def _buy(self, flag, t):
        # check t == 0
        prev_t = self._check_index(t)                                                              

        # change in action (delta) used to determine shares for a position                                                           
        market_price = self.meta[t][self.marketprice_idx]
        
        avl_cash = self.meta[prev_t][self.avl_cash_idx]
        prev_shares = int(self.shares_arr[prev_t])

        position_size = 0
        # calculate shares and pnl
        match(flag):
            case("LB"):           # buy long position
                position_size, new_shares = self.positions.add_position("L", market_price, avl_cash)
                current_shares = prev_shares + new_shares
                avl_cash -= position_size 

            case("SB"):          # buy short position
                position_size, new_shares= self.positions.add_position("S", market_price, avl_cash)
                current_shares = prev_shares - new_shares
                avl_cash -= position_size

            case("HP"):          # hold same position as previous time step
                if t != self.ep_len - 1:
                    future_price = self.meta[t+1][self.marketprice_idx]
                    unrealized_pnl = self.positions.hold_position(future_price)
                    position_size = self.meta[prev_t][self.psize_idx]
                    avl_cash = self.meta[prev_t][self.avl_cash_idx]
                    self.meta[t+1][self.unrealized_pnl_idx] = unrealized_pnl
                
                current_shares = prev_shares       
        
        # update info
        self.meta[t][self.avl_cash_idx] = avl_cash                        
        self.meta[t][self.psize_idx] = position_size
        self.shares_arr[t] = int(current_shares)
        self.flag_arr[t] = flag

    def _sell(self, flag, t):
        # check t == 0 and correct it
        prev_t = self._check_index(t)                                                           
        
        # change in action (delta) used to determine shares for a position
        market_price = self.meta[t][self.marketprice_idx]
        avl_cash = self.meta[prev_t][self.avl_cash_idx]

        current_shares = 0
        # calculate shares and pnl
        if flag == "LS":                # sell long position                                                                         
            realized_pnl, position_size = self.positions.sell_position(market_price)
            current_shares = 0
            self.total_ep_pnl = realized_pnl

        elif flag == "SS":              # sell short position
            realized_pnl, position_size = self.positions.sell_position(market_price)          # helper returns positive shares
            current_shares = 0
            self.total_ep_pnl = realized_pnl
        
        # update avl_cash
        avl_cash += position_size

        # update info
        self.meta[t][self.realized_pnl_idx] = realized_pnl
        self.meta[t][self.unrealized_pnl_idx] = 0
        self.meta[t][self.avl_cash_idx] = avl_cash
        self.meta[t][self.psize_idx] = 0
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
    def __init__(self, cash, position_price, flag):
        self.cash = cash
        self.position_price = position_price
        self.shares = int(self.cash / position_price)
        self.size = self.shares * position_price
        self.flag = flag
    
    # helper to calculate pnl (different calculation for long or short positions)
    def get_pnl(self, market_price):
        if (self.flag == "L"):
            return (market_price - self.position_price) * self.shares
        elif (self.flag == "S"):
            return -1 * (market_price - self.position_price) * self.shares

# kept general structure from previous iteration for implementing simultaneous trades in multiple tickers for the future
# helper class to abstract position tracking and managemement, used to return appropriate shares and pnl after trading actions
class PositionLedger:
    def __init__(self):
        self.ledger = []
    
    def hold_position(self, market_price):  
        unrealized_pnl = 0 
        
        if len(self.ledger) != 0:
            position = self.ledger.pop()
            unrealized_pnl = position.get_pnl(market_price)
            self.ledger.append(position)
        
        return unrealized_pnl
    
    # open new position and return shares, pnl, and size
    def add_position(self, flag, current_price, cash):    
        position = PositionNode(cash, current_price, flag)
        self.ledger.append(position)
        return position.size, int(position.shares)

    # close position and return shares, pnl, and size 
    def sell_position(self, market_price):
        realized_pnl = 0 
        size = 0
        
        if len(self.ledger) != 0:
            position = self.ledger.pop()
            realized_pnl = position.get_pnl(market_price)
            size = position.size
        
        return realized_pnl, size
    
        
        
    


     

    

    