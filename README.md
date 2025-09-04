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

## Expanded Details
I thought I would talk a bit more about the lower level details of the project.

**Data Pipeline**
The data pipeline has two parts:
* data_ingestion.py: fetches data from Polygon.io
* data_enrichment.py: transforms raw data into better features

The fetch part of the program can request data from any time range, any frequency of data (1 minute, 5 minute, etc.) and any selection of tickers, assuming you have a subscription to Polygon. I specifically chose to request data in the 5 minute frequency as this timeframe presents a sweet spot for intraday trading. Though I have also considered experimenting with 1 minute frequency to train a scalp trader. 
For the time span I chose to get every piece of data that was available to me which was the past 10 years. Since my goal isn't just learning one ticker at a time, but rather learning some level of cross ticker generalization. This problem is quite complex and so it would naturally benefit from having as many timesteps of data as possible.
For the tickers I selected roughly 20 or so which I filtered using the yahoo finance stock screener. The stock screener allowed me to fine tune my data selection to only include tickers that were likely to have stable patterns. These were high volume, moderate beta, large cap stocks listed on the nasdaq100. 

The rest endpoint responds with JSON to requests and this object contains the raw candle data (open, high, low, close prices + volume) which is parsed and saved to a CSV file for each ticker. The pipeline uses multiprocessing to speed up the fetch by requesting, parsing and then saving to CSV for multiple tickers (8 in this case) at the same time. This speed up is necessary since the end result is millions of data points. 

After this point the data needs to be preprocessed. data_enrichment.py does some basic pandas operations to accomplish this, but the main work is done by **feature_generation.py**. The class here computes custom features made from transforming raw technical indicator data (which itself is computed using pandas_ta over the raw OHLCV data). Normalization happens here as well, most features are z score normalized while a few use a different technique specific to that feature. Since some of these features will be unbounded due to z scores, and so tanh is used to squash the value between -1 and 1. This is suitable for the tanh activation function that is used by each layer in the neural network. 

**Custom Gym**
The gym is defined with an observation space that essentially has a 2D matrix shape. This is because the obs are a rolling window of size n over size m features, N x M. Currently this is 12 x 10, meaning 12 timesteps with each timestep being a vector of 10 features. The "rolling" window just means that the next state/obs is made of the the 1st till 12th index, meanwhile the previous was 0th till 11th. Each rolling window contains N - 1 previous candles and only 1 new "present" candle. This behaviour mimics how humans view the time series when trading. 

12 timesteps with 5 minute candles equates to 1 hour of trading time. This is essentially the exact amount I want the agent to be able to "see" or consider when it's making a trading decision. 1 hour is the general rule when doing intraday trading on the 5 minute timeframe, and so this number felt appropriate given the timeframe. Each episode is a single day involving a single ticker, and so upon initialization the gym precomputes the ticker-day episodes that are needed for a run (every defined training run can use any combination of tickers)

The action space meanwhile is just a singular float between -1 and 1. The action space is not complex at all for trading, even the continous action space proves to be bothersome for trading which in reality just has 3 discrete actions. The environment maps this float to taking certain trades, this is done by interpreting the float as portfolio allocation percentage. For example 0.2 would mean "buy long position with 20% of cash", similarly the negative can represent short positions and moving more towards one end could mean buying and regressing in the opposite direction could mean selling. For example 0.2 -> 0.3 means buy another 10% worth of shares at timestep 2. This way the environment is able to process long, short and reversal trades.

However the issue is that the agent finds it difficult to stay out of the market (consistently output 0) or sell off the entire position (output 0 not 0.2) due to the continuous action space. A mildly effective fix for this was to map a higher portion of the action space to 0 (currently actions between -0.4 and 0.4 map to 0) and rounding the float to one decimal place (so that the actions jump in 10% increments). These two additions greatly reduce the interpretable action space which cut down a whole lot of noise from sporarid training. The agent still produces it's normal float but the idea is to make learning easier by not forcing it to learn very precise behaviour in an already noisy and complex domain. 

The gyms reward logic is also complex. In the early iterations the reward was directly the net asset value of the trades the agent makes. Now, the reward is based on different kinds of states that the agent is likely to be in. So for example there is a different reward calculation for holding no trades, holding losing or winning trades, selling losing or winning trades, and even penalizations for overtrading (via slippage and transaction cost estimates). This fine tuning of dense reward has noticably sculpted intelligent behaviour. It is not perfect, but the agent's that are trained do end up showing consistently larger average profit compared to average loss. The base value used to calculate the reward is the current unrealized PnL (or realized PnL if rewarding a selling position state) relative to a profit target. For example the profit target can be 100 dollars per trade, if the current PnL is 80 dollars then the relative value is 0.8. It's a simple representation but can be used in a nuanced way. 

Lastly, the gym handles extensive logging. It saves the normalized obs that the agent sees (for debugging), it creates an info vector for every timestep to keep track of what the agent/environment is doing. This too is saved as a CSV and can be used to analyze and produce performance stats (such as the average profit and losses). It contains only useful auxiliary information to expose what's occurring under the head. It also outputs the total PnL at every episode to a callback wrapper which logs this to tensorboard alongside the default SB3 logging. The gym also outputs differently labelled CSVs for training and validation runs for the same model so that data is not overwritten.

## Future developments
* Fine tune TCN architecture
* Experiment with feature generation to further improve cross ticker generalization
* Change CLI (`run.py`) to be able to directly edit `config.yaml` for "train" and "fetch data" actions
