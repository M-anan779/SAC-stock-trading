import random
import time
from stable_baselines3 import SAC
from environment.intraday_trading_env import TradingEnv
from utils.training_analysis import Analyzer
from torch.utils.tensorboard import SummaryWriter

def validate(model_path, val_dir, tickers, eval_steps):
    num = random.randint(0, 999)                                                                            # random number helps distinguish validation runs for the same model save
    
    # initialize base environment as validation env and load saved model
    eval_env = TradingEnv(val_dir, tickers, model_path, validation=True, num=num, steps=eval_steps)
    model = SAC.load(model_path, env=eval_env)

    obs, info = eval_env.reset()

    arr = []
    n = 5
    timesteps = 0
    episode_count = 0
    writer = SummaryWriter(f"{model_path}-eval_pnl-{num}")                          # tensorboard logging
    start_time = time.time()                                                        # for timing
    current_time = start_time
    
    while timesteps < eval_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        timesteps += 1

        # if at the episode end then compute moving average of total pnl from that episode
        if terminated or truncated:
            episode_count += 1
            pnl = info.get("episode_pnl", 0.0)
            arr.append(pnl)

            if len(arr) >= n:
                avg = sum(arr[-n:]) / n
                writer.add_scalar(f"eval_pnl_ma-{num}", avg, episode_count)

            obs, info = eval_env.reset()

        # print run time and elapsed time
        if timesteps % 10000 == 0:
            prev_time = current_time
            current_time = time.time()
            print(f"\nFinished {timesteps} steps of evaluation, total time elapsed: {round(current_time - start_time, 3)} seconds, time taken: {round(current_time - prev_time, 3)} seconds")  
    
    # compute stats from environment/trading auxiliary info csv file created after validation run
    print(f"\nValidation completed({eval_steps}), outputting stats...")
    csv_path = f"{model_path}-validation-{num}.csv"
    analyzer = Analyzer(csv_path)
    analyzer.get_summary()



