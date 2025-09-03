import os
import yaml
from pathlib import Path
from utils.validation import validate
from utils.training import train
from utils.training_analysis import Analyzer
from pipeline.restapi.data_ingestion import run as fetch
from pipeline.restapi.data_enrichment import run as enrich

# helper for checking valid integer inputs from user and prompting retries if the input exceeds the specified [0, length - 1] positive range
def _get_input(input_prompt, length):
    user_input = 0
    valid = False
    while not valid:
        user_input = int(input(input_prompt))
        if user_input < 0 or user_input > length - 1:
            print("Invalid input")
            continue
        else:
            valid = True
    
    return user_input

# helper used to manage initial model select parts of the UI
def _model_select_helper():
    # select log id (time at which some training was done)
    log_dir = Path("logs/")
    log_ids = []

    print("\ntraining runs: ")
    for i, log_id in enumerate(os.listdir(log_dir)):
        log_ids.append(log_id)
        print(f"{i} - {log_id}")

    prompt = "Enter index number to select training run: "
    length = len(log_ids)
    user_input = int(_get_input(prompt, length))
    log_id = log_ids[user_input]
    model_dir = Path(f"logs/{log_id}")
    
    # select saved model
    model_saves = []
    j = 0
    print("\nModel saves: ")
    for save_name in os.listdir(model_dir):
        if save_name.endswith(".zip"):
            save_name = os.path.splitext(save_name)[0]
            model_saves.append(save_name)
            print(f"{j} - {save_name}")
            j += 1
    prompt = "Enter index number to select model save: "
    length = len(model_saves)
    user_input = int(_get_input(prompt, length))
    save_name = model_saves[user_input]

    return model_dir, save_name

# load a training or validation log file for a saved model and then compute performance stats from it
def run_analysis():

    # helper to get user input to load a model
    model_dir, save_name = _model_select_helper()
    
    # get user input to load csv
    csv_files = []
    j = 0

    print("\nCSV logs: ")
    for file in os.listdir(model_dir):
        if file.endswith(".csv"):
            if save_name in file and ("validation" in file or "training" in file):
                csv_files.append(file)
                print(f"{j} - {file}")
                j+=1
    prompt = "Enter index number to select file for analysis: "
    length = len(csv_files)
    user_input = int(_get_input(prompt, length))
    csv_name = csv_files[user_input]
    
    # compute performance stats
    csv_path = Path(f"{model_dir}/{csv_name}")
    analyzer = Analyzer(csv_path)
    print(f"\nRunning analysis using '{csv_path}'...")
    analyzer.get_summary()

# validates saved models by loading validation set data by selecting tickers and providing the total number of timesteps to validate the model over 
def run_validation(config):
    
    # helper to get user input to load a model
    model_dir, save_name = _model_select_helper()
    model_path = Path(f"{model_dir}/{save_name}")
    valid_tickers = config["tickers"]
    
    # store user input to create the list of tickers to use for validation data
    tickers = []
    steps = int(input("\nEnter number of evaluation steps: "))
    i = 0
    print("\nTickers: ")
    for ticker in valid_tickers:
        print(f"{i} - {ticker}")
        i+=1
    print(f"{i} - cancel")
    print(f"{i+1} - restart")
    print(f"{i+2} - confirm")
    prompt = "Select validation ticker(s) (or cancel/restart/confirm): "
    user_input = -1
    while True:
        user_input = int(_get_input(prompt, len(valid_tickers) + 3))
        
        # cancel
        if user_input == i:
            print("\nCancelling...")
            break

        # restart
        elif user_input == i+1:
            tickers = []
            print("\nRestarting process...")
            continue

        # confirm/continue 
        elif user_input == i+2:
            if len(tickers) > 0:
                print(f"\nProceeding with selection: {tickers}")
                validate(model_path, config["validation_dir"], tickers, steps)
                break

            # edge case no tickers selected
            else:
                tickers = []
                print(f"\nNo tickers were selected...")
                continue
        
        # add selected ticker to list
        else:
            ticker = config["tickers"][user_input]
            if ticker not in tickers:
                print(f"\nAdding ticker: {ticker}")
                tickers.append(ticker)
                print(f"Current selection: {tickers}")
            
            #edge case duplication selection
            else:
                print(f"\nTicker '{ticker}' has already been selected. Try again...")
                print(f"Current selection: {tickers}")
                continue

# trains new/saved models by running training splits defined as passing {ticker, timesteps} to the environment
def run_training(config):
    
    # print options
    print("0 - new model")
    print("1 - saved model")
    prompt = "Select training option: "
    user_input = int(_get_input(prompt, 2))

    match user_input:
        # train a new model
        case 0:
            model_path = None
            print("\nNew model will be trained...")
        
        # train a saved model further
        case 1:
            model_dir, save_name = _model_select_helper()
            model_path = Path(f"{model_dir}/{save_name}")
            print(f"\n{save_name} from {model_dir} will be trained...")
    
    i = 0
    runs = []
    print("\nTraining splits: ")
    for run, splits in config["training_runs"].items():
        runs.append(run)
        print(f"{i} - {run}:")
        for index, split in enumerate(splits):
            print(f"                    split_{index}: {split}")
        i += 1
    print(f"{i} - cancel\n")
    prompt = "Select training run: "
    user_input = int(_get_input(prompt, i+1))
    if user_input == i:
        print("\nCancelling...")
        return
    else:
        run = runs[user_input]
        print(f"\nstarting run labelled: {run}")
        train(splits, config["training_dir"], model_path)

def main():
    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f) 
    
    while (True):

        # print main menu
        print("\n0 - train")
        print("1 - validate")
        print("2 - analyse")
        print("3 - fetch data")
        print("4 - generate features")
        print("5 - quit")

        prompt = "Select action: "
        user_input = int(_get_input(prompt, 6))

        match user_input:
            # train new models
            case 0:
                print("\nTraining:")
                run_training(config)

            # validate saved models
            case 1:
                run_validation(config)
            
            # analyze training/validation csv files to compute stats
            case 2:
                run_analysis()
            
            # fetch data from polygon api
            case 3:
                print("\nFetching data...")
                fetch(config)
            
            # data enrichment to compute features
            case 4:
                print("\nComputing features...")
                enrich(config)
            
            # exit program
            case 5:
                print("\nQuitting...")
                return

if __name__ == "__main__":
    main()
