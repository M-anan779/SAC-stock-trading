import os
import gymnasium as gym
import torch
import torch.nn as nn
from datetime import datetime
from stable_baselines3 import SAC
from environment.intraday_trading_env import TradingEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter 

print(torch.cuda.get_device_name(0))        # check whether using GPU

# Register the custom environment
gym.envs.registration.register(
    id='TradingEnv-v0',
    entry_point='rl_env:TradingEnv', 
)

# create and check environment shape
env = TradingEnv()
obs, info = env.reset()
print(f"Observation shape: {obs.shape}, dtype: {obs.dtype}")
print(f"Action space: {env.action_space}, Obs space: {env.observation_space}")

# wrap in VecEnv for SB3
vec_env = make_vec_env(lambda: Monitor(TradingEnv()), n_envs=1)

# SAC model setup
log_dir = os.path.join("logs", datetime.now().strftime("%m%d-%H%M"))
os.makedirs(log_dir, exist_ok=True)

model = SAC(
    policy='MlpPolicy',
    env=vec_env,
    verbose=1,
    tensorboard_log=log_dir,
    policy_kwargs=dict(
        activation_fn=nn.Tanh,     
        net_arch=[256, 256, 256]        
    ),
    learning_rate=3e-4,
    buffer_size=1_000_000,
    batch_size=256,
    train_freq=(1, 'step'), 
    gradient_steps=1,  
    gamma=0.99,
    tau=0.005,
    ent_coef=0.05,
)

# evaluation callback
eval_env = TradingEnv()
eval_env = Monitor(eval_env)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=log_dir,
    log_path=log_dir,
    eval_freq=2000,
    deterministic=True,
    render=False,
)

class PnLCallBack(BaseCallback):
    def __init__(self, writer, verbose: int = 0):
        super().__init__(verbose)
        self.writer = writer

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode_pnl" in info:
                ep_pnl = info["episode_pnl"]
                self.writer.add_scalar("ep_pnl", ep_pnl, self.num_timesteps)
        return True

pnl_writer = SummaryWriter(os.path.join(log_dir, "pnl"))
pnl_callback = PnLCallBack(writer=pnl_writer)

# train model
print("Starting training...")
model.learn(total_timesteps=100_000, callback=[eval_callback, pnl_callback], tb_log_name="SAC_Trading")
print("Training complete.")

# save model
model.save(os.path.join(log_dir, "sac_trading_env"))
print(f"Model saved to {log_dir}")


