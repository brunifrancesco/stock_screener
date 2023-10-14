import datetime
import itertools
import json
import os
import random
import shutil
import time
import traceback
from itertools import chain
from multiprocessing import Pool

import numpy as np
import pandas as pd
from polygon import RESTClient

from strategy import test1, test2, test3, Params, test1_bis
from datetime import datetime, timedelta

client = RESTClient(api_key=os.environ.get("POLYGON_KEY", None))

ema_short = list(range(2, 20))
ema_long = list(range(20, 50))
ma_long = list(range(100, 300, 20))

rsi_buy_threshold = list(map(lambda item: item * 1.05, (20, 25, 30, 35, 40, 45, 50)))
rsi_sell_threshold = list(map(lambda item: item * 1.05, (50, 55, 60, 65, 70, 75, 80)))
rsi_window = (10, 11, 12, 13, 14, 15, 16, 17, 18)
take_profit_percentage, stop_loss_percentage = (
    25,
    100,
    200,
), (8,)
k_threshold_range = np.arange(0.1, 10.1, 0.1)
d_threshold_range = np.arange(0.01, 1.01, 0.01)


# fetch raw data
def fetch():
    start_date = datetime(2022, 1, 1)
    end_date = datetime.now()
    interval = timedelta(days=30)

    date_ranges = []

    while start_date < end_date:
        next_date = start_date + interval
        if next_date > end_date:
            next_date = end_date
        date_ranges.append((start_date, next_date))
        start_date = next_date

    for stock_name in (
        "AAPL",
        "JNJ",
        "TSM",
        "TSLA",
        "PYPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "JPM",
        "WMT",
        "MA",
        "ORCL",
        "BABA",
        "ADBE",
        "ROG",
        "CSCO",
        "ACN",
        "NKE",
        "MCD",
        "FEZ",
        "KO",
        "FB",
        "IBM",
        "SONY",
    ):
        dfs = []
        if os.path.exists(f"raw/{stock_name}_D.csv"):
            continue
        for start, end in date_ranges:
            print(f"Processing {stock_name} from {start} to {end}")
            df = pd.DataFrame(
                client.list_aggs(
                    ticker=stock_name,
                    multiplier=1,
                    timespan="day",
                    from_=start.strftime("%Y-%m-%d"),
                    to=end.strftime("%Y-%m-%d"),
                    limit=50000,
                )
            )
            dfs.append(df)
            time.sleep(random.randrange(10, 50))
        pd.concat(dfs, axis=0).to_csv(f"raw/{stock_name}_D.csv", index=False)


def _run_strategy(
    candlestick_data_resampled, stock_name, interval, combinations, strategy
):
    for index, combination in enumerate(combinations):
        print_status(combinations, index, strategy)
        param = Params.from_combinations(
            stock_name.split("/")[-1], interval, strategy.__name__, combination
        )
        yield strategy(candlestick_data_resampled, param)


def run_strategy_1(candlestick_data_resampled, stock_name, interval):
    combinations = list(
        itertools.product(
            ema_short,
            ema_long,
            rsi_window,
            rsi_buy_threshold,
            rsi_sell_threshold,
            take_profit_percentage,
            stop_loss_percentage,
            (1,),
            (1,),
            (1,),
            (1,),
        )
    )
    return _run_strategy(
        candlestick_data_resampled, stock_name, interval, combinations, test1
    )


def run_strategy_2(candlestick_data_resampled, stock_name, interval):
    combinations = list(
        itertools.product(
            ema_short,
            ema_long,
            (1,),
            (1,),
            (1,),
            take_profit_percentage,
            stop_loss_percentage,
            (1,),
            (1,),
            (1,),
            (1,),
        )
    )
    return _run_strategy(
        candlestick_data_resampled, stock_name, interval, combinations, test2
    )


def run_strategy_3(candlestick_data_resampled, stock_name, interval):
    combinations = list(
        itertools.product(
            (1,),
            (1,),
            rsi_window,
            rsi_buy_threshold,
            rsi_sell_threshold,
            take_profit_percentage,
            stop_loss_percentage,
            (1,),
            (1,),
            (1,),
            (1,),
        )
    )
    return _run_strategy(
        candlestick_data_resampled, stock_name, interval, combinations, test3
    )


def run_strategy_1bis(candlestick_data_resampled, stock_name, interval):
    combinations = list(
        itertools.product(
            (1,),
            ma_long,
            rsi_window,
            rsi_buy_threshold,
            rsi_sell_threshold,
            take_profit_percentage,
            stop_loss_percentage,
            (1,),
            (1,),
            (1,),
            (1,),
        )
    )
    return _run_strategy(
        candlestick_data_resampled, stock_name, interval, combinations, test1_bis
    )


def print_status(combinations, index, fn):
    if index % 2000 == 0:
        print(f"Completed {index} out of {len(combinations)} for {fn.__name__}")


# find the models for a stock
def train_single_model(file: str):
    stock_name = file.replace(".csv", "")
    interval = "24H"
    try:
        candlestick_data = pd.read_csv(os.path.join("raw", file))
        candlestick_data["time"] = pd.to_datetime(
            candlestick_data["timestamp"] / 1000, unit="s"
        )
        candlestick_data = candlestick_data.set_index(candlestick_data["time"])
        candlestick_data = candlestick_data.sort_index()

        if os.path.exists(f"best_models_for_stocks/{stock_name}_{interval}.csv"):
            return
        print(f"Processing {stock_name} for {interval}")

        result_strategy_1 = list(run_strategy_1(candlestick_data, stock_name, interval))
        result_strategy_2 = list(run_strategy_2(candlestick_data, stock_name, interval))
        result_strategy_3 = list(run_strategy_3(candlestick_data, stock_name, interval))
        result_strategy_1bis = list(
            run_strategy_1bis(candlestick_data, stock_name, interval)
        )

        cleaned_result = list(
            filter(
                lambda item: item is not None,
                chain(
                    result_strategy_1,
                    result_strategy_2,
                    result_strategy_3,
                    result_strategy_1bis,
                ),
            )
        )
        # Convert data class instances to dictionaries
        stock_as_dict = (vars(stock) for stock in cleaned_result)

        # Create a pandas DataFrame
        result_df = pd.DataFrame(stock_as_dict)
        result_df["n of trades"] = result_df["trades"].map(len)
        result_df["j_trades"] = result_df["trades"].map(json.dumps)

        print(f"Saving to best_models_for_stocks/{stock_name}_{interval}")
        result_df.to_csv(
            f"best_models_for_stocks/{stock_name}_{interval}.csv", index=False
        )
    except Exception as e:
        print(f"Fail to processing {stock_name}")
        traceback.print_exc()


def train(stocks):
    Pool(8).map(train_single_model, stocks)


# select the best models
def best_models():
    dfs = []
    shutil.rmtree("best_models", ignore_errors=True)
    os.makedirs("best_models")

    for file in os.listdir("best_models_for_stocks"):
        print(f"Processing {file}")
        df = pd.read_csv(f"best_models_for_stocks/{file}")
        dfs.append(df)
    all_best_models = pd.concat(dfs, axis=0)

    all_best_models = all_best_models[
        [
            "stock_name",
            "strategy",
            "pvalue",
            "n of trades",
            "sharpe_ratio",
            "max_drawdown",
            "winning_ratio",
            "ema_short",
            "ema_long",
            "rsi_window",
            "rsi_buy_threshold",
            "rsi_sell_threshold",
            "take_profit_percentage",
            "stop_loss_percentage",
            "trades",
            "ema_diff",
            "rsi_diff",
            "is_ema_diff_valid",
            "is_rsi_diff_valid",
        ]
    ]

    all_best_models = all_best_models[all_best_models["is_ema_diff_valid"] == True]
    all_best_models = all_best_models[all_best_models["is_rsi_diff_valid"] == True]
    all_best_models = all_best_models[all_best_models["winning_ratio"] > 0.6]
    all_best_models["n of trades"] = all_best_models["n of trades"] * 2

    all_best_models.groupby(["stock_name"], as_index=False).apply(
        lambda x: x.sort_values(
            by=[
                "pvalue",
                "n of trades",
                "winning_ratio",
                "max_drawdown",
                "ema_diff",
                "rsi_diff",
            ],
            ascending=[False, True, False, False, False, False],
        )
    ).reset_index(drop=True).groupby(["stock_name"], as_index=False).head(10).to_csv(
        f"best_models/best_D.csv", index=False
    )

    all_best_models[all_best_models["sharpe_ratio"] > 0.7].groupby(
        ["stock_name"], as_index=False
    ).apply(
        lambda x: x.sort_values(
            by=[
                "pvalue",
                "n of trades",
                "winning_ratio",
                "sharpe_ratio",
                "max_drawdown",
                "ema_diff",
                "rsi_diff",
            ],
            ascending=[False, True, False, False, True, False, False],
        )
    ).reset_index(
        drop=True
    ).groupby(
        ["stock_name"], as_index=False
    ).head(
        10
    ).to_csv(
        f"best_models/best_with_sharpe07.csv", index=False
    )

    all_best_models[all_best_models["sharpe_ratio"] > 0.7].groupby(
        ["stock_name", "strategy"], as_index=False
    ).apply(
        lambda x: x.sort_values(
            by=[
                "pvalue",
                "n of trades",
                "winning_ratio",
                "sharpe_ratio",
                "max_drawdown",
                "ema_diff",
                "rsi_diff",
            ],
            ascending=[False, True, False, False, True, False, False],
        )
    ).reset_index(
        drop=True
    ).groupby(
        ["stock_name", "strategy"], as_index=False
    ).head(
        1
    ).to_csv(
        f"best_models/best_sharpe07.csv", index=False
    )


if __name__ == "__main__":
    fetch()
    # train(
    #       filter(lambda item: '.csv' in item, os.listdir('raw'))
    # )
    # best_models()
    pass
