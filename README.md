# Deep Reinforcement Learning for Intraday Stock Trading

This project implements a complete end-to-end **deep reinforcement learning (DRL) pipeline** for training stock trading agents using the **Soft Actor-Critic (SAC)** algorithm in a **custom gym environment**. 
The base SAC implementation is handled by **Stable-Baselines3** but has been modified to use a **Temporal Convolutional Network (TCN)** as the feature extractor rather than an MLP. Since it is better suited for learning time series data.

The pipeline performs several actions to faciliate this goal, including:
* Data preprocessing
* Training 
* Validation
* Logging figures
* Computing performance stats

---

## Features

* **Data Pipeline** (`data_enrichment.py`, `data_ingestion.py`)
  * **Raw data ingestion** from [Polygon.io](https://polygon.io) stocks endpoint API (past 10 years, multiple stock tickers)
  * **Preprocessing** raw data (price + volume data) to compute technical indicator data
  * **Feature engineering** statistical features from these computed technical indicators to produce the final observations data
  * **Normalization** of obs data using tanh squashing
  * Splitting obs data into **training + validation sets**
    
* **Custom Gym environment** (`intraday_trading_env.py`)
  * **Gym API** compliant training environment compatibale with Stable-Baselines3
  * Simulates **realistic broker/trading logic** from interpreting SAC's continuous action space
  * Outputs a rolling window of 12 candles as observations (represent the flow of market data), with each candle being represented by 10 features
  * Allows for **trading both short and long positions**
  * Keeps track of different positions and PnL of active positions
  * Implements the necessary financial accounting logic involved in stock trading (portfolio cash, available cash, short positions, shares, Pnl, etc.) 
  * **Custom reward logic** (computes varying rewards for different types of trading actions)
  * **CSV logging** of agent trading activity and environment states (position type, position size, position value, reward, etc.)
  * **Tensorboard logging** of SB3 training logs (actor, critic losses, reward, etc.) and a moving average of total PnL at every episode end

* **Training** (`training.py`)
  * Define a base model in SB3 to train (algorithm name, layers, hyperparameters, training parameters, etc.)
  * Run different user defined training splits from `config.yaml`
  * Save model and allow for continuing training later
   
* **Validation** (`validation.py`)
  * Deterministic validation runs of saved models on unseen data
  * Select ticker data to validate on and number of steps 
  * Uses a separate validation set of data for the specified ticker(s) (data range: 2023 - 2025)

* **Analysis** (`training_analysis.py`)
  * Outputs performance stats such as win/loss rate, PnL ratio, expected value, etc.
  * Uses the CSV logs produced by the environment during training/validation runs
  
* **Parallelization and other Optimizations**
  * **Multi-processing** for data ingestion and enrichment
  * **Vectorized pandas operations** for efficient DataFrame usage where possible
  * **Vec-env** used during training to maximize GPU utilization and speed up training
  * Buffered CSV logging and numpy arrays used in the environment
  
* **CLI** (`run.py`)
  * Makes it easier to run these functions individually
 
**Additional Notes:** 
  * `config.yaml` needs to be edited manually for defining some information like training runs and validation tickers
  * polygon API key is required to run data ingestion and fetch data
  * files under `old_flatfile/` are deprecated and not used currently (was part of early development when csv flatfile downloads contained data).

---

## Installation

```
git clone https://github.com/M-anan779/SAC-stock-trading.git
cd SAC-stock-trading
pip install -r requirements.txt
```

Required Python packages:

* `torch`, `stable-baselines3`, `gymnasium`, `tensorboard`
* `numpy`, `pandas`, `pandas-ta`
* `requests`

---

## Example Usage

Run the CLI:

```
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

Depending on the selected option, you will either get further prompts to select further options or the program will run using the information it finds in the `config.yaml` file.

Example of running analysis:

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

## Outputs
* Under each `/logs/<run_id>/` directory (created for each unique training run at the project root):
  * Training logs and TensorBoard metrics
  * Saved model (as `model_<split_num>.zip`)
  * Agent trading CSV logs for training and validation (as `model_<split_num>-training-_.csv`, `*model_<split_num>-validation-<rand_num>*.csv` respectively)
* In terminal:
  * Performance stats computed from the CSV files
---

## Architecture and Design

### Observation Space
The observstion space is a `[12, 10]` matrix to represent a rolling window of size 12 timesteps with each timestep having a vector of size 10 features. The 10 features aim to encode a raw OHLCV candle (raw market data of the candle). By rolling window it means that timestep `t = 0` represents indices `[0, 11]` (inlcusive), while `t = 1` represents indices `[1, 12]`. Each timestep contains `N - 1` previous "candles" with the last one being the "latest" current candle. The market data being used for training is of the 5 min frequency and so a window size of 12 represents 1 hour of trading time, which is appropriate for intraday trading and finding relevant patterns. 

### Action Space
The action space is one floating point value between `[-1, 1]` which is translated into trading actions by the environment. On it's own the envrionment can interpret this value to represent position sizing as a percentage. Meanwhile the change in the current action compared to the previous can represent direction. Which in turn can be interpreted as buying and selling. The environment can map the action to buying or selling: **long, short and reversal positons with different position sizes**. 
However, since trading inherently is a domain with a discrete action space, the continuous action space can be quite noisy and output subpar behaviour; such as not being able to output 0 consistently to sell completely and stay out of market. As a result the action space has a "dead zone", a range of values that all map to 0 so as to make that output artifically more frequent (agent doesn't have to be as precise). Additionally, the action is rounded to one decimal so that more of the action space maps to the same 10% increments. This is necessary since less than 10-20% granular control for position sizing is meaningless in trading and just noise.

### Features
The 10 features in the observation space are computed from transforming raw technical indicator data into other custom representatoins. The techical indicator data itself is a tranformation of the raw OHLCV data. 

### Normalizaton
Most features are **z score normalized** (per feature) while a few use a different technique specific to it (such as dividing by window size for a counter). Since some of these features will be unbounded due to z scores, tanh is used to squash the values between -1 and 1. This in paricular is also suitable for the tanh activation function that is used by each layer in the neural network. 

### Network Architecture
The network architecture is using a `TCN` as a **feature extractor**. The feature extractor is not shared and so the actor and the critic have their own `TCN`. This was desirable as now the policy and critic gradients do not interfere like they would if both gradients flowed through the same extractor. Theoretically it should also make learning better since one `TCN` specifically updates to encode features over the input for the actor, and the other specifically learns better encoding to suit the critic. 

The feature extractor is three `Conv1d` layers. Each layer has input and output channels set to 32 except for the first layer whos input channels matches the number of features in the input obs (10). The kernal size is 5 which leads to a receptive field that just about covers the whole obs window. Since the timesteps are small (12) there is no need to use dilated convolutions. Each layer has a `LayerNorm` right after with the same number of channels. Lastly, `Tanh` is used for the activation layer. `Tamh` is used since the 10 features of the input obs has negative values which `ReLU` would interpret as 0, killing signal. 
```
causalConv1d(in_channels=in_c, out_channels=features_dim, kernel_size=5), 
            nn.LayerNorm(normalized_shape=[features_dim, timesteps]), nn.Tanh(),
```
The `causalConv1d` here is just a wrapper class for the standard `Conv1d` but with padding on the left. 

The last timestep is taken from the output generated by the extractor to get a flat vector for the subsequent linear nets for the task specific heads for the policy and critic networks. The policy network uses a linear activation (no hidden layer) since it's job is quite simple and it's accuracy relies a lot more on the critics. The critic networks have a more complex and crucial job and so the critic networks have hidden layers on top of the extractor so that it can capture any nonlinearities present in the encoding. The hidden layer is the default linear net but with tanh as the activation function (to match the rest of the network). There are two layers here, one of size 64 and the other size 32. 

```
policy_kwargs = dict(
    features_extractor_class=TCN,
    features_extractor_kwargs=dict(features_dim=32),
    net_arch=dict(pi=[], qf=[64, 32]),
    activation_fn=torch.nn.Tanh,
    share_features_extractor=False              
)
```

### Model Parameters
The model parameters have not been tweaked much. Following is the current set:
```
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
```

## Future developments
* Fine tune TCN architecture
* Experiment with feature generation to further improve cross ticker generalization
* Change CLI (`run.py`) to be able to directly edit `config.yaml` for "train" and "fetch data" actions
