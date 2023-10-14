import dataclasses
from typing import Dict, Iterable

import numpy as np
import pandas as pd
from pandas import DataFrame

NA = "NA"


@dataclasses.dataclass
class Params:
    stock_name: str = None
    interval: str = None
    strategy: str = None
    ema_short: int = None
    ema_long: int = None
    rsi_window: int = None
    rsi_buy_threshold: float = None
    rsi_sell_threshold: float = None
    take_profit_percentage: int = None
    stop_loss_percentage: int = None
    k_period: int = None
    d_period: int = None
    k_threshold: float = None
    d_threshold: float = None

    @classmethod
    def from_combinations(cls, stock_name, interval, strategy, combinations):
        return Params(stock_name, interval, strategy, *combinations)


@dataclasses.dataclass
class Response(Params):
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    winning_ratio: float = 0.0
    pvalue: float = 0.0
    trades: Iterable[Dict] = ()
    movements: Iterable[float] = ()
    is_ema_diff_valid: bool = False
    is_rsi_diff_valid: bool = False
    ema_diff: int = 0
    rsi_diff: float = 0.0


def compute_max_d_and_sharpe(trades):
    # Convert the list of trades into a DataFrame
    df = pd.DataFrame(trades)

    # Calculate daily returns
    df["Date"] = pd.to_datetime(df["Date"])
    df["Sell_Date"] = pd.to_datetime(df["Sell_Date"])
    df["Days_Held"] = (df["Sell_Date"] - df["Date"]).dt.days
    df["Daily_Return"] = (df["Exit_Price"] - df["Price"]) / df["Price"]

    # Calculate max drawdown
    df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod()
    df["Max_Return"] = df["Cumulative_Return"].cummax()
    df["Drawdown"] = df["Cumulative_Return"] / df["Max_Return"] - 1
    max_drawdown = df["Drawdown"].min()

    # Calculate Sharpe ratio
    risk_free_rate = 0.03  # Adjust the risk-free rate as needed
    mean_return = df["Daily_Return"].mean()
    std_deviation = df["Daily_Return"].std()
    sharpe_ratio = (mean_return - risk_free_rate) / std_deviation

    winning_trades = sum(1 for trade in trades if trade.get("Profit", 0) > 0)

    # Calculate the winning trades ratio
    total_trades = len(trades)
    winning_trades_ratio = winning_trades / total_trades

    return max_drawdown, sharpe_ratio, winning_trades_ratio


# ema + rsi
def test1(df: DataFrame, params: Params, trade=True, position="OUT"):
    take_profit_level, stop_loss_level = None, None
    df["ema_short"] = df["close"].ewm(span=params.ema_short, adjust=False).mean()
    df["ema_long"] = df["close"].ewm(span=params.ema_long, adjust=False).mean()

    rsi = compute_rsi(df, params)
    df["rsi"] = rsi

    portfolio_value = 100.0
    trades = []
    movements = []

    if trade:
        df = df[:-1]

    # Iterate through the DataFrame rows
    for i in range(1, len(df)):
        if position == "IN_LONG":
            price = df["close"].astype(float)[i]
            if take_profit_level and price > take_profit_level:
                position = "OUT"
                take_profit_level = None
                movements.append(portfolio_value)
                trade = trades[-1]  # Get the most recent trade
                trade["Exit_Price"] = price
                trade["Profit"] = (price - trade["Price"]) * trade["Shares"]
                trade["Sell_Date"] = df["time"][i].strftime("%Y-%m-%d")
                portfolio_value += trade["Profit"]
                trade["ExitReason"] = "TP"
                continue
            if stop_loss_level and price < stop_loss_level:
                position = "OUT"
                stop_loss_level = None
                movements.append(portfolio_value)
                trade = trades[-1]  # Get the most recent trade
                trade["Exit_Price"] = price
                trade["Profit"] = (price - trade["Price"]) * trade["Shares"]
                trade["Sell_Date"] = df["time"][i].strftime("%Y-%m-%d")
                portfolio_value += trade["Profit"]
                trade["ExitReason"] = "SL"
                continue
        # Check for a buy signal
        if (
            df["ema_short"][i] > df["ema_long"][i]
            and df["rsi"][i] < params.rsi_buy_threshold
        ):  # and df['close'].astype(float)[i] > df['VWAP'][i]:
            if position == "OUT":
                # Generate a buy signal
                position = "IN_LONG"
                # Execute the buy trade here or set up a notification/alert
                movements.append(-portfolio_value)
                entry_price = df["close"].astype(float)[i]
                shares = portfolio_value / entry_price
                trades.append(
                    {
                        "Date": df["time"][i].strftime("%Y-%m-%d"),
                        "Type": "Buy",
                        "Price": entry_price,
                        "Shares": shares,
                    }
                )
                take_profit_level = entry_price * (
                    1 + params.take_profit_percentage / 100
                )
                stop_loss_level = entry_price * (1 - params.stop_loss_percentage / 100)
                # requests.post('https://api.pushover.net/1/messages.json', data={'token': 'ahjv52wf1xhw5wmss4p2anjkkyzg7a', 'user':"g6seuq5d5d33sv2v8izzevcs6vgoea", 'message': "BUY"})

        # Check for a sell signal
        elif (
            df["ema_short"][i] < df["ema_long"][i]
            and df["rsi"][i] > params.rsi_sell_threshold
        ):  # df['close'].astype(float)[i] > df['VWAP'][i]:and df['close'].astype(float)[i] < df['VWAP'][i]:
            if position == "IN_LONG":
                # Generate a sell signal
                position = "OUT"
                movements.append(portfolio_value)
                # Execute the sell trade here or set up a notification/alert
                exit_price = df["close"].astype(float)[i]
                trade = trades[-1]  # Get the most recent trade
                trade["Exit_Price"] = exit_price
                trade["Profit"] = (exit_price - trade["Price"]) * trade["Shares"]
                trade["Sell_Date"] = df["time"][i].strftime("%Y-%m-%d")
                portfolio_value += trade["Profit"]
                # requests.post('https://api.pushover.net/1/messages.json', data={'token': 'ahjv52wf1xhw5wmss4p2anjkkyzg7a', 'user':"g6seuq5d5d33sv2v8izzevcs6vgoea", 'message': "SELL"})
    if len(trades) > 0 and trades[0].get("Sell_Date", None):
        max_drawdown, sharpe_ratio, winning_ratio = compute_max_d_and_sharpe(trades)
        return Response(
            ema_short=params.ema_short,
            ema_long=params.ema_long,
            pvalue=portfolio_value,
            trades=trades,
            rsi_window=params.rsi_window,
            rsi_buy_threshold=params.rsi_buy_threshold,
            rsi_sell_threshold=params.rsi_sell_threshold,
            take_profit_percentage=params.take_profit_percentage,
            stop_loss_percentage=params.stop_loss_percentage,
            movements=movements,
            strategy=params.strategy,
            interval=params.interval,
            stock_name=params.stock_name,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            winning_ratio=winning_ratio,
            is_ema_diff_valid=params.ema_long > params.ema_short,
            is_rsi_diff_valid=params.rsi_buy_threshold < params.rsi_sell_threshold,
            ema_diff=params.ema_long - params.ema_short,
            rsi_diff=params.rsi_sell_threshold - params.rsi_buy_threshold,
        )


# 200MA + rsi
def test1_bis(df: DataFrame, params: Params, trade=True, position="OUT"):
    take_profit_level, stop_loss_level = None, None
    df["ma"] = df["close"].rolling(params.ema_long).mean()

    rsi = compute_rsi(df, params)
    df["rsi"] = rsi

    portfolio_value = 100.0
    trades = []
    movements = []

    if trade:
        df = df[:-1]

    # Iterate through the DataFrame rows
    for i in range(1, len(df)):
        if position == "IN_LONG":
            price = df["close"].astype(float)[i]
            if take_profit_level and price > take_profit_level:
                position = "OUT"
                take_profit_level = None
                movements.append(portfolio_value)
                trade = trades[-1]  # Get the most recent trade
                trade["Exit_Price"] = price
                trade["Profit"] = (price - trade["Price"]) * trade["Shares"]
                trade["Sell_Date"] = df["time"][i].strftime("%Y-%m-%d")
                portfolio_value += trade["Profit"]
                trade["ExitReason"] = "TP"
                continue
            if stop_loss_level and price < stop_loss_level:
                position = "OUT"
                stop_loss_level = None
                movements.append(portfolio_value)
                trade = trades[-1]  # Get the most recent trade
                trade["Exit_Price"] = price
                trade["Profit"] = (price - trade["Price"]) * trade["Shares"]
                trade["Sell_Date"] = df["time"][i].strftime("%Y-%m-%d")
                portfolio_value += trade["Profit"]
                trade["ExitReason"] = "SL"
                continue
        # Check for a buy signal
        if (
            df["ma"][i] > df["close"][i] and df["rsi"][i] < params.rsi_buy_threshold
        ):  # and df['close'].astype(float)[i] > df['VWAP'][i]:
            if position == "OUT":
                # Generate a buy signal
                position = "IN_LONG"
                # Execute the buy trade here or set up a notification/alert
                movements.append(-portfolio_value)
                entry_price = df["close"].astype(float)[i]
                shares = portfolio_value / entry_price
                trades.append(
                    {
                        "Date": df["time"][i].strftime("%Y-%m-%d"),
                        "Type": "Buy",
                        "Price": entry_price,
                        "Shares": shares,
                    }
                )
                take_profit_level = entry_price * (
                    1 + params.take_profit_percentage / 100
                )
                stop_loss_level = entry_price * (1 - params.stop_loss_percentage / 100)
                # requests.post('https://api.pushover.net/1/messages.json', data={'token': 'ahjv52wf1xhw5wmss4p2anjkkyzg7a', 'user':"g6seuq5d5d33sv2v8izzevcs6vgoea", 'message': "BUY"})

        # Check for a sell signal
        if (
            df["ma"][i] > df["close"][i] and df["rsi"][i] > params.rsi_sell_threshold
        ):  # and df['close'].astype(float)[i] > df['VWAP'][i]:
            if position == "IN_LONG":
                # Generate a sell signal
                position = "OUT"
                movements.append(portfolio_value)
                # Execute the sell trade here or set up a notification/alert
                exit_price = df["close"].astype(float)[i]
                trade = trades[-1]  # Get the most recent trade
                trade["Exit_Price"] = exit_price
                trade["Profit"] = (exit_price - trade["Price"]) * trade["Shares"]
                trade["Sell_Date"] = df["time"][i].strftime("%Y-%m-%d")
                portfolio_value += trade["Profit"]
                # requests.post('https://api.pushover.net/1/messages.json', data={'token': 'ahjv52wf1xhw5wmss4p2anjkkyzg7a', 'user':"g6seuq5d5d33sv2v8izzevcs6vgoea", 'message': "SELL"})
    if len(trades) > 0 and trades[0].get("Sell_Date", None):
        max_drawdown, sharpe_ratio, winning_ratio = compute_max_d_and_sharpe(trades)
        return Response(
            ema_short=params.ema_short,
            ema_long=params.ema_long,
            pvalue=portfolio_value,
            trades=trades,
            rsi_window=params.rsi_window,
            rsi_buy_threshold=params.rsi_buy_threshold,
            rsi_sell_threshold=params.rsi_sell_threshold,
            take_profit_percentage=params.take_profit_percentage,
            stop_loss_percentage=params.stop_loss_percentage,
            movements=movements,
            strategy=params.strategy,
            interval=params.interval,
            stock_name=params.stock_name,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            winning_ratio=winning_ratio,
            is_ema_diff_valid=True,
            is_rsi_diff_valid=params.rsi_buy_threshold < params.rsi_sell_threshold,
            ema_diff=params.ema_long - params.ema_short,
            rsi_diff=params.rsi_sell_threshold - params.rsi_buy_threshold,
        )


# only emas
def test2(df: DataFrame, params: Params, trade=True, position="OUT"):
    take_profit_level, stop_loss_level = None, None
    df["ema_short"] = df["close"].ewm(span=params.ema_short, adjust=False).mean()
    df["ema_long"] = df["close"].ewm(span=params.ema_long, adjust=False).mean()

    portfolio_value = 100.0
    trades = []
    movements = []
    if trade:
        df = df[:-1]

    # Iterate through the DataFrame rows
    for i in range(1, len(df)):
        if position == "IN_LONG":
            price = df["close"].astype(float)[i]
            if take_profit_level and price > take_profit_level:
                position = "OUT"
                take_profit_level = None
                movements.append(portfolio_value)
                trade = trades[-1]  # Get the most recent trade
                trade["Exit_Price"] = price
                trade["Profit"] = (price - trade["Price"]) * trade["Shares"]
                trade["Sell_Date"] = df["time"][i].strftime("%Y-%m-%d")
                portfolio_value += trade["Profit"]
                trade["ExitReason"] = "TP"
                continue
            if stop_loss_level and price < stop_loss_level:
                position = "OUT"
                stop_loss_level = None
                movements.append(portfolio_value)
                trade = trades[-1]  # Get the most recent trade
                trade["Exit_Price"] = price
                trade["Profit"] = (price - trade["Price"]) * trade["Shares"]
                trade["Sell_Date"] = df["time"][i].strftime("%Y-%m-%d")
                portfolio_value += trade["Profit"]
                trade["ExitReason"] = "SL"
                continue
        # Check for a buy signal
        if df["ema_short"][i] > df["ema_long"][i]:
            if position == "OUT":
                # Generate a buy signal
                position = "IN_LONG"
                movements.append(-portfolio_value)
                # Execute the buy trade here or set up a notification/alert
                entry_price = df["close"].astype(float)[i]
                shares = portfolio_value / entry_price
                trades.append(
                    {
                        "Date": df["time"][i].strftime("%Y-%m-%d"),
                        "Type": "Buy",
                        "Price": entry_price,
                        "Shares": shares,
                    }
                )
                take_profit_level = entry_price * (
                    1 + params.take_profit_percentage / 100
                )
                stop_loss_level = entry_price * (1 - params.stop_loss_percentage / 100)

        # Check for a sell signal
        elif df["ema_short"][i] < df["ema_long"][i]:
            if position == "IN_LONG":
                # Generate a sell signal
                position = "OUT"
                movements.append(portfolio_value)
                # Execute the sell trade here or set up a notification/alert
                exit_price = df["close"].astype(float)[i]
                trade = trades[-1]  # Get the most recent trade
                trade["Exit_Price"] = exit_price
                trade["Profit"] = (exit_price - trade["Price"]) * trade["Shares"]
                trade["Sell_Date"] = df["time"][i].strftime("%Y-%m-%d")
                portfolio_value += trade["Profit"]
    if len(trades) > 0 and trades[0].get("Sell_Date", None):
        max_drawdown, sharpe_ratio, winning_ratio = compute_max_d_and_sharpe(trades)
        return Response(
            ema_short=params.ema_short,
            ema_long=params.ema_long,
            pvalue=portfolio_value,
            trades=trades,
            rsi_window=params.rsi_window,
            rsi_buy_threshold=params.rsi_buy_threshold,
            rsi_sell_threshold=params.rsi_sell_threshold,
            take_profit_percentage=params.take_profit_percentage,
            stop_loss_percentage=params.stop_loss_percentage,
            movements=movements,
            strategy=params.strategy,
            interval=params.interval,
            stock_name=params.stock_name,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            winning_ratio=winning_ratio,
            is_ema_diff_valid=params.ema_long > params.ema_short,
            is_rsi_diff_valid=True,
            ema_diff=params.ema_long - params.ema_short,
            rsi_diff=params.rsi_sell_threshold - params.rsi_buy_threshold,
        )


# pure rsi
def test3(df: DataFrame, params: Params, trade=True, position="OUT"):
    take_profit_level, stop_loss_level = None, None

    rsi = compute_rsi(df, params)

    df["rsi"] = rsi

    portfolio_value = 100.0
    trades = []
    movements = []

    if trade:
        df = df[:-1]

    # Iterate through the DataFrame rows
    for i in range(1, len(df)):
        if position == "IN_LONG":
            price = df["close"].astype(float)[i]
            if take_profit_level and price > take_profit_level:
                position = "OUT"
                take_profit_level = None
                movements.append(portfolio_value)
                trade = trades[-1]  # Get the most recent trade
                trade["Exit_Price"] = price
                trade["Profit"] = (price - trade["Price"]) * trade["Shares"]
                trade["Sell_Date"] = df["time"][i].strftime("%Y-%m-%d")
                portfolio_value += trade["Profit"]
                trade["ExitReason"] = "TP"
                continue
            if stop_loss_level and price < stop_loss_level:
                position = "OUT"
                stop_loss_level = None
                movements.append(portfolio_value)
                trade = trades[-1]  # Get the most recent trade
                trade["Exit_Price"] = price
                trade["Profit"] = (price - trade["Price"]) * trade["Shares"]
                trade["Sell_Date"] = df["time"][i].strftime("%Y-%m-%d")
                portfolio_value += trade["Profit"]
                trade["ExitReason"] = "SL"
                continue
        # Check for a buy signal
        if (
            df["rsi"][i] < params.rsi_sell_threshold
        ):  # and df['close'].astype(float)[i] > df['VWAP'][i]:
            if position == "OUT":
                # Generate a buy signal
                position = "IN_LONG"
                movements.append(-portfolio_value)
                # Execute the buy trade here or set up a notification/alert
                entry_price = df["close"].astype(float)[i]
                shares = portfolio_value / entry_price
                trades.append(
                    {
                        "Date": df["time"][i].strftime("%Y-%m-%d"),
                        "Type": "Buy",
                        "Price": entry_price,
                        "Shares": shares,
                    }
                )
                take_profit_level = entry_price * (
                    1 + params.take_profit_percentage / 100
                )
                stop_loss_level = entry_price * (1 - params.stop_loss_percentage / 100)

        # Check for a sell signal
        elif (
            df["rsi"][i] > params.rsi_sell_threshold
        ):  # df['close'].astype(float)[i] > df['VWAP'][i]:and df['close'].astype(float)[i] < df['VWAP'][i]:
            if position == "IN_LONG":
                # Generate a sell signal
                position = "OUT"
                movements.append(portfolio_value)
                # Execute the sell trade here or set up a notification/alert
                exit_price = df["close"].astype(float)[i]
                trade = trades[-1]  # Get the most recent trade
                trade["Exit_Price"] = exit_price
                trade["Profit"] = (exit_price - trade["Price"]) * trade["Shares"]
                trade["Sell_Date"] = df["time"][i].strftime("%Y-%m-%d")
                portfolio_value += trade["Profit"]
    if len(trades) > 0 and trades[0].get("Sell_Date", None):
        max_drawdown, sharpe_ratio, winning_ratio = compute_max_d_and_sharpe(trades)
        return Response(
            ema_short=params.ema_short,
            ema_long=params.ema_long,
            pvalue=portfolio_value,
            trades=trades,
            rsi_window=params.rsi_window,
            rsi_buy_threshold=params.rsi_buy_threshold,
            rsi_sell_threshold=params.rsi_sell_threshold,
            take_profit_percentage=params.take_profit_percentage,
            stop_loss_percentage=params.stop_loss_percentage,
            movements=movements,
            strategy=params.strategy,
            interval=params.interval,
            stock_name=params.stock_name,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            winning_ratio=winning_ratio,
            is_ema_diff_valid=True,
            is_rsi_diff_valid=params.rsi_buy_threshold < params.rsi_sell_threshold,
            ema_diff=params.ema_long - params.ema_short,
            rsi_diff=params.rsi_sell_threshold - params.rsi_buy_threshold,
        )


def compute_rsi(df, params):
    window = params.rsi_window  # Specify the window period for RSI calculation
    delta = df["close"].astype(float).diff()  # Compute the price differences
    gain = delta.where(delta > 0, 0)  # Set negative differences to zero (gains)
    loss = -delta.where(delta < 0, 0)  # Set positive differences to zero (losses)
    avg_gain = gain.rolling(window).mean()  # Compute the average gain over the window
    avg_loss = loss.rolling(window).mean()  # Compute the average loss over the window
    rs = avg_gain / avg_loss  # Compute the relative strength (RS)
    rsi = 100 - (100 / (1 + rs))  # Compute the RSI"""
    return rsi
