import os
import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from datetime import datetime
from environment.intraday_trading_env import TradingEnv 

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
        net_arch=[256, 256]        
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
    eval_freq=2500,
    deterministic=True,
    render=False,
)

# train model
print("Starting training...")
model.learn(total_timesteps=75_000, callback=eval_callback, tb_log_name="SAC_Trading")
print("Training complete.")

# save model
model.save(os.path.join(log_dir, "sac_trading_env"))
print(f"Model saved to {log_dir}")


