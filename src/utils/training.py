import os
import gymnasium as gym
import torch
from datetime import datetime
from stable_baselines3 import SAC
from environment.intraday_trading_env import TradingEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from torch.utils.tensorboard import SummaryWriter
from model_arch.tcn import TCN

print(torch.cuda.get_device_name(0))        # check whether using GPU

# register gym base environment
gym.envs.registration.register(
    id="TradingEnv-v0",
    entry_point="rl_env:TradingEnv", 
)

class PnLCallBack(BaseCallback):
    def __init__(self, writer, ma_window, verbose: int = 0):
        super().__init__(verbose)
        self.writer = writer
        self.ma_window = ma_window
        self.episode_pnls = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # get info dictionary for total pnl for episode and done flag for episode end
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            if done and "episode_pnl" in info:
                ep_pnl = info["episode_pnl"]
                self.episode_count += 1
                self.episode_pnls.append(ep_pnl)
                
                # calculate and log moving average of total pnl per episode on tensorboard
                if len(self.episode_pnls) >= self.ma_window:
                    avg = sum(self.episode_pnls[-self.ma_window:]) / self.ma_window
                    self.writer.add_scalar(f"ep_pnl_ma", avg, self.episode_count)

        return True

def train(splits, train_dir, model_path, load_save):
    log_id = datetime.now().strftime("%m%d-%H%M")
    log_dir = os.path.join("logs", log_id)
    os.makedirs(log_dir, exist_ok=True)

    model = None
    if load_save == False:
        model_path = os.path.join(log_dir, f"model_0")
    
    # train model iteratively through each split (curriculum)
    for split_count, split in enumerate(splits):
        tickers = split["tickers"]
        timesteps = split["timesteps"]
        print(f"Starting split_{split_count}: {tickers}, {timesteps} steps\n")
        
        # initialize training environtment, tensorboard and env callback logging
        train_env = make_vec_env(lambda: Monitor(TradingEnv(train_dir, tickers, model_path, validation=False, num="_", steps=timesteps)), n_envs=12)
        pnl_writer = SummaryWriter(f"{model_path}-pnl_ma")
        pnl_callback = PnLCallBack(writer=pnl_writer, ma_window=5)
        
        # load model from previous split in same training run (train the same model iteratively through various splits)
        if split_count >= 1:
            model_path = os.path.join(log_dir, f"model_{split_count}")
            model = SAC.load(os.path.join(log_dir, f"model_{split_count-1}"), env=train_env)
            model.learning_starts = 0
        
        # first iteration
        else:
            # load model from some previous save
            if load_save:       
                model = SAC.load(model_path, env=train_env)
            
            # load base model
            else:           
                model = SAC(
                    policy="MlpPolicy",
                    env=train_env,
                    verbose=1,
                    tensorboard_log=log_dir,
                    policy_kwargs = dict(
                        features_extractor_class=TCN,
                        features_extractor_kwargs=dict(features_dim=32),
                        net_arch=dict(pi=[], qf=[32, 32]),
                        activation_fn=torch.nn.Tanh,
                        share_features_extractor=False              
                    ),
                    learning_rate=3e-4,
                    buffer_size=1_000_000,
                    batch_size=512,
                    learning_starts=25_000,
                    train_freq=1,
                    gradient_steps=4,
                    gamma=0.99,
                    tau=0.005,
                    use_sde=True,
                    sde_sample_freq=20,
                    target_entropy=-0.5,
                    ent_coef="auto",
                )
        
        # begin model training and save
        model.learn(total_timesteps=timesteps, callback=pnl_callback, tb_log_name=f"model_{split_count}-learning")
        model.save(model_path)
        



