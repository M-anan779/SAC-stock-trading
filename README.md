# Deep Reinforcement Learning for Intraday Stock Trading

This project implements a complete end-to-end **deep reinforcement learning (DRL) pipeline** for training stock trading agents using the **Soft Actor-Critic (SAC)** algorithm in a **custom gym environment**.  
The base SAC implementation is handled by **Stable-Baselines3**, but it is modified to use a **Temporal Convolutional Network (TCN)** as the feature extractor rather than an MLP. Since it’s better suited to time-series data.

The pipeline performs several actions to facilitate this goal, including:
* Data preprocessing
* Training 
* Validation
* Logging
* Computing performance stats

---

## Features

* **Data Pipeline** (`data_enrichment.py`, `data_ingestion.py`)
  * **Raw data ingestion** from [Polygon.io](https://polygon.io) (past 10 years, multiple stock tickers)
  * **Preprocessing** raw OHLCV data to compute technical indicators
  * **Feature engineering** statistical features from these indicators to produce the final observations
  * **Normalization** of observations using tanh squashing
  * Splitting data into **training + validation sets**
    
* **Custom Gym Environment** (`intraday_trading_env.py`)
  * **Gym API**–compatible training environment for Stable-Baselines3
  * Simulates **realistic broker/trading logic** from interpreting SAC’s continuous action space
  * Outputs a rolling window of 30 candles as observations (flow of market data), with each candle represented by 9 features
  * Tracks positions and PnL of active positions
  * Implements financial accounting (available cash, shares, PnL, etc.)
  * **Custom reward logic** (nuanced reward calculations for multiple categories of trading states)
  * **CSV logging** of agent activity and environment state (position type/size/value, reward, etc.)
  * **TensorBoard logging** of SB3 metrics (actor/critic losses, reward, etc.) plus a moving average of total PnL at episode end

* **TCN Feature Extractor** (`tcn.py`)
  * 7 layer 1D convolutional net with k = 5, out_channels = 128 and LayerNorm per layer preceding the activation (tanh)
  * Uses a 1 x 1 convolutional layer as a projection layer to project input features (in_channels = 9)
  * Separate feature extractor for actor and critic heads
  * Final linear projection outputs latent vector to actor and critic MLP heads (these are two layers, 128, tanh activation)

* **Training** (`training.py`)
  * Define a base SB3 model (algorithm, network layers, hyperparameters, training params)
  * Run user-defined training splits from `config.yaml`
  * Save model and continue training later
  
* **Validation** (`validation.py`)
  * Deterministic validation runs of saved models on unseen data
  * Select ticker(s) and number of steps 
  * Uses a held-out validation set (data range: 2023–2025)

* **Analysis** (`training_analysis.py`)
  * Outputs performance stats such as win/loss rate, PnL ratio, expected value, etc.
  * Uses the CSV logs produced by the environment during training/validation

* **Parallelization and Other Optimizations**
  * **Multiprocessing** for data ingestion and enrichment
  * **Vectorized pandas operations** where possible
  * **VecEnv** during training to maximize GPU utilization and speed
  * Buffered CSV logging and NumPy arrays in the environment
  
* **CLI** (`run.py`)
  * Run individual functions easily
 
**Additional Notes:** 
  * `config.yaml` is edited manually to define training runs and validation tickers
  * A Polygon API key is required to fetch data
  * Files under `old_flatfile/` are deprecated (from early CSV-based development)

---

## Installation

```bash
git clone https://github.com/M-anan779/SAC-stock-trading.git
cd SAC-stock-trading
pip install -r requirements.txt
````

**Required Python packages:**

* `torch`, `stable-baselines3`, `gymnasium`, `tensorboard`
* `numpy`, `pandas`, `pandas-ta`
* `requests`

---

## Example Usage

Run the CLI:

```bash
python src/run.py
```

Main options:

```
0 - train
1 - validate
2 - analyse
3 - fetch data
4 - generate features
5 - quit
Select action: 2
```

Depending on the selection, you’ll either get further prompts or the program will run using values from `config.yaml`.

**Example: running analysis**

```
training runs: 
0 - 0827-2035
1 - 0827-2116
2 - 0827-2157
3 - 0827-2238
4 - 0827-2319
5 - 0828-2059

Enter index number to select training run: 5

0 - model_0
Enter index number to select model save: 0

0 - model_0-training-_.csv
1 - model_0-validation-594.csv
2 - model_0-validation-658.csv
3 - model_0-validation-982.csv
Enter index number to select file for analysis: 3

Running analysis using 'logs\0828-2059\model_0-validation-982.csv'...

Performance Stats:
buy_count: 72280.0
sell_count: 57348.0
hold_count: 3518.0
buy_rate: 0.543
sell_rate: 0.431
hold_rate: 0.026
profit_total: 1781935.41
loss_total: -1536514.474
pnl_total: 245420.936
profit_avg: 61.92
loss_avg: -54.44
profit_max: 1547.796
loss_max: -2512.244
win_count: 28778.0
lose_count: 28224.0
win_rate: 0.505
lose_rate: 0.495
winlose_ratio: 1.02
avg_pl_ratio: 1.137
pnl_per_year: 105644.774
expected_value: 4.305

0 - train
1 - validate
2 - analyse
3 - fetch data
4 - generate features
5 - quit
Select action: 5

Quitting...
```
These are results from an actual training run, though at that point the network architecture was an MLP.

## Outputs

* Under each `/logs/<run_id>/` directory (created per training run):

  * Training logs and TensorBoard metrics
  * Saved model (`model_<split_num>.zip`)
  * Agent trading CSV logs for training/validation (`model_<split_num>-training-_.csv`, `model_<split_num>-validation-<rand>.csv`)
* In terminal:

  * Performance stats computed from the CSV files

---

## Architecture and Design

### Data Set
The raw data set is requested from Polygon.io by the pipeline, requests are limited to just the stock data endpoint. The following parameters define what data is fetched:
  * Time span: Data from 2015 to 2025 was used for this project
  * Tickers: Any stock listed on the US market. I selected roughly 20 stock picks from the NASDAQ100 based on volume and volatility metrics (example beta < 1.8)
  * Time frame: Such as 5-minute, 1-day etc. For this project I initially used 5-minute data but am currently experimenting with 1-minute data. The reason being as this data set contains more state transitions.

After preprocessing of the raa data as well, this results in about 300k-550k data points per individual stock. Giving well over **8.4 million state transitions** that could be used for training.

### Goal
The goal for the agent is to be able to learn to trade multiple tickers through some level of **cross ticker generalization**. Since real world data is limited, training on one ticker will not produce enough timesteps to learn such a complex task through DRL and not using enough varied data will most definately lead to overfitting.

As a result, my goal is to design the environment (`obs, reward`) and model architecture such that it is able to generalize learning across tickers and therefore benefit productively from being **trained across different tickers and market regimes**.

### Observation Space

The **observation** space is a `[30, 9]` matrix representing a rolling window of 30 timesteps, each with a 9-feature vector. The 9 features encode an OHLCV “candle.”
“Rolling window” means at time `t = 0` the indices are `[0, 30]` (inclusive), while at `t = 1` they are `[1, 31]`. Each window contains `N − 1` prior candles and one “current” candle. Using 1-minute data, a 30-step window covers 30 minutes, which is a practical context length for intraday trading at this smaller time frame.

### Action Space

The action is a single float in `[-1, 1]`. The environment interprets this value as discretized buckets as so:
```
        action = np.clip(action, -1, 1)
        action = np.round(action, 3)                                                   
        
        action = int(np.round(abs(action) * 1000))
        action = action % 2
```
In this case there are only 2 buckets:
  * mod 0: "NP", "LS" (hold no position, sell long position)
  * mod 1: "LB", "HP" (buy long position, hold same position)

These two buckets are repeated to cover the entire action space, that is to say the continuous action space at any point is always interpreted as either mod 0 or mod 1. The reason I chose this design was because my previous agents would always get stuck to one "type" of action (such as only going long) but not trying to hold or sell. This was because huge areas of my action space was just one type of action. Example: action > 0 interpreted as long. So for the agent to discover shorting it would need to output action < 0. The issue is that SAC samples from a gaussian distribution given by paramters it learns, the mean and the std. As training progresses SAC shrinks the std which results in sampled actions that are closer to the mean. This means if the action mean is equal to 0.5 then it is very unlikely for the agent to consistently output -0.5 and continue exploring. This issue becomes worse when the only way to hold out of trading is outputting 0. This explained why all of my previous agents could not discover holding consistently. 

By making the buy and sell options always next to each other and repeatedly at various values in the action space, this makes the agent much more likely to discover selling or holding (both same position and no position) even with smaller STD. With this change the agents I trained exhibited much healthier aciton distributions with "hold" type actions outweighing the buy/sell actions. i.e. the agent was being logical about it's trading activity. 

The agents action have no variable position size element. This is because in day trading especially involving one stock, it does not make sense to take positions without using the entire capital that is available to us. Since profits scale linearly with position size. This also made analyzing PnL trends much simpler too, and made small bugs visible that had previously gone unnoticed.

### Features

The 9 features are computed by transforming technical-indicator outputs (derived from OHLCV) into custom **representations**.

### Normalization

Most features are **z-score normalized** (per feature); a few use feature-specific scaling (e.g., dividing a counter by window size). Since z-scores can be unbounded, **tanh** is used to squash values to `[-1, 1]`, which also aligns with the network’s **Tanh** activations.

### Network Architecture

The network uses a **TCN** as a feature extractor. The extractor is **not shared**: the actor and critic each have their own TCN so their gradients don’t interfere. One extractor learns representations suited to the policy, the other for the critic.

The first layer of the extractor is a 1x1 Conv layer that mixes values across features but only within one time step (k = 1). This projection layer is meant to stabilize the feature encoding:
```
          nn.Conv1d(in_channels=in_c, out_channels=features_dim, kernel_size=1),
```

The extractor has seven `Conv1d` layers. Each uses 128 channels and **kernel size = 5**, giving a receptive field that covers the entire window with R = 31. I chose having deeper layers over using dilations.
After each conv:
* `LayerNorm` with `[features_dim, timesteps]`
* `Tanh` activation (preserves negative inputs that `ReLU` would zero)
```
          causalConv1d(in_channels=features_dim, out_channels=features_dim, kernel_size=5), 
          nn.LayerNorm(normalized_shape=[features_dim, timesteps]),
          nn.Tanh(),
```

`causalConv1d` is a thin wrapper around `Conv1d` with left padding.

We take the **last timestep** from the extractor output as a flat vector for the policy/critic heads.
Both the **policy head** and the **critic heads** use two linear layers with `Tanh` activations, size: 128 → 128.
```
policy_kwargs = dict(
         features_extractor_class=TCN,
         features_extractor_kwargs=dict(features_dim=128),
         net_arch=dict(pi=[128, 128], qf=[128, 128]),
         activation_fn=torch.nn.Tanh,
         share_features_extractor=False,
         log_std_init=-2.3,      # σ ≈ 0.1         
     )
```

### Model Parameters

Current model parameters:
```
          learning_rate=1e-4,
          buffer_size=650_000,
          batch_size=256,
          learning_starts=35_000,
          train_freq=1,
          gradient_steps=1,
          gamma=0.98,
          tau=0.03,
          use_sde=True,
          sde_sample_freq=20,
          ent_coef="auto",
          target_entropy="-0.5"
```
Typical training runs with this model are done using 3 ticker pairings (AMD, NVIDIA, INTC: chip manufacturers) over 1 to 1.5 million timesteps (roughly 8 years of data). Validation is done on the remaining 2 years of unseen data for each ticker used during training. 

## Future Developments
* Keep experimenting with net_arch and oba

