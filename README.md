# Deep Reinforcement Learning for Intraday Stock Trading

This project implements a complete end-to-end **deep reinforcement learning (DRL) pipeline** for training stock trading agents using the **Soft Actor-Critic (SAC)** algorithm in a **custom environment**. The SAC implementation is handled in **Stable-Baselines3**.

The pipeline performs several actions to faciliate this goal, including:
* Data preprocessing
* Training and validation (custom gym)
* Tensorboard and CSV logging
* Computing performance metrics

## 

---

## Features

* **Data Pipeline** (`data_enrichment.py`, `data_ingestion.py`)
  * **Raw data ingestion** from [Polygon.io](https://polygon.io) stocks endpoint API (past 10 years, multiple stock tickers)
  * **Preprocessing** raw data (price + volume data) to compute technical indicator data
  * **Feature engineering** statistical features from these computed technical indicators
  * **Normalization** of data using tanh squashing
  * Splitting data into **training + validation sets**
    
* **Custom Gym environment** (`intraday_trading_env.py`)
  * `TradingEnv` (Gymnasium API) compliant training environment
  * Simulates **realistic broker/trading logic** from interpreting SAC's continuous action space
  * Outputs a rolling window of 12 candles as observations (represent the flow of market data)
  * Allows for **trading both short and long positions**
  * Keeps track of different positions and pnl of active positions
  * Implements necessary financial accounting involved in trading (portfolio cash, available cash, short positions, shares, Pnl, etc.) 
  * **Custom reward logic** (computes varying rewards for different types of trading actions)
  * **CSV logging** of agent trading activity and environment states (position type, position size, position value, reward, etc.)
  * **Tensorboard logging** of SB3 training logs (actor, critic losses, reward, etc.) and a moving average of total PnL at every episode end

* **Training** (`training.py`)
  * Define a base model in SB3 to train (algorithm name, layers, hyperparameters, training parameters)
  * Run different user defined training splits from `config.yaml`
  * Save model and allow for continuing training later
   
* **Validation** (`validation.py`)
  * Deterministic validation runs of saved models
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

## Future developments
* Modify the neural network architecture by swapping the MLP for a TCN feature extractor
* Experiment with feature generation to further improve cross ticker generalization
* Change CLI (`run.py`) to be able to directly edit `config.yaml` for "train" and "fetch data" actions
* Add name of tickers used for validation when printing performance stats in `Analyzer`
