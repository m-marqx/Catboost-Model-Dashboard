"""
This module provides a function to calculate the daily trade results for
a given exchange, initial capital, and symbol.

The function `get_trade_results` takes the following parameters:
    - exchange (ccxt.Exchange): The exchange object.
    - initial_capital (float): The initial capital for trading.
    - symbol (str): The trading symbol.

It returns a pandas DataFrame containing the daily trade results.

The trade results are calculated based on the provided exchange, initial
capital, and symbol. The function performs various calculations and
transformations on the trade data to derive the daily trade results.

The module requires the following dependencies:
    - numpy
    - pandas
    - ccxt
"""

from typing import Literal
import numpy as np
import pandas as pd
import ccxt

def get_trade_results(
    exchange: ccxt.Exchange,
    initial_capital: float,
    symbol: str,
    interval: str,
    info: Literal['Daily Results', 'Trade Info'] = 'Daily Results',
) -> pd.DataFrame:
    """
    Calculate the daily trade results for a given exchange, initial
    capital, and symbol.

    Parameters:
        exchange (ccxt.Exchange): The exchange object.
        initial_capital (float): The initial capital for trading.
        symbol (str): The trading symbol.
        interval (str): The time interval for resampling the trade data.
        info (Literal['Daily Results', 'Trade Info']): The type of
        information to return.
        (default: 'Daily Results')

    Returns:
        pd.DataFrame: The daily trade results.

    Raises:
        ValueError: If the `info` parameter is not 'Daily Results' or
        'Trade Info'.

    Note:
        This function has been tested only on Binance exchange.
    """
    if info not in ['Daily Results', 'Trade Info']:
        raise ValueError(
            "The `info` parameter must be either 'Daily Results' or "
            "'Trade Info'."
        )
    trades = pd.DataFrame(exchange.fetch_my_trades(symbol))

    trades['amount'] = np.where(
        trades['side'] == 'buy',
        trades['amount'], -abs(trades['amount'])
    )

    df_trades_info = pd.DataFrame.from_records(trades['info'])

    df_trades_info['time'] = pd.to_datetime(
        df_trades_info['time'].astype('int64'), unit='ms'
    )

    df_trades_info = df_trades_info.set_index('time')

    df_trades_info = df_trades_info.drop(
        columns=['maker', 'buyer', 'id', 'orderId', 'symbol']
    )

    df_trades_info['commission'] = -abs(
        df_trades_info['commission'].astype('float64')
    )

    df_trades_info['liquid_result'] = (
        df_trades_info[['realizedPnl', 'commission']].astype(float)
        .sum(axis=1)
    )

    df_trades_info['liquid_result_aggr'] = (
        df_trades_info['liquid_result']
        .cumsum()
    )

    df_trades_info['result'] = (
        df_trades_info['realizedPnl'].astype(float)
        .cumsum()
    )

    df_trades_info['liquid_result_usd_aggr'] = (
        df_trades_info['liquid_result_aggr']
        * df_trades_info['price'].astype(float)
    )

    df_trades_info['capital'] = (
        df_trades_info['liquid_result_usd_aggr']
        / initial_capital
    )

    df_trades_info['is_close'] = np.where(
        (df_trades_info['realizedPnl'].astype('float') == 0),
        np.nan, ((df_trades_info['capital'].diff() > 0)).astype(int)
    )

    df_trades_info.iloc[0, -1] = 0.5

    if info == 'Trade Info':
        return df_trades_info

    funding_fees = pd.DataFrame(exchange.fetch_funding_history())

    funding_fee =(
        pd.DataFrame.from_records(funding_fees['info'])
        .set_index(funding_fees['timestamp'].astype('datetime64[ms]'))
        ['income'].astype('float64')
        .to_frame()
    )
    resampled_funding_fee = funding_fee.resample(interval).sum()

    daily_results = (
        resampled_funding_fee['income']
        .rename('funding_fee')
        .to_frame()
    )

    daily_results['operational_result'] = (
        df_trades_info['liquid_result']
        .resample(interval)
        .sum()
    )

    daily_results['liquid_result'] = (
        daily_results['operational_result']
        + daily_results['funding_fee']
    )

    daily_results['price'] = (
        df_trades_info['price']
        .resample(interval)
        .last()
        .ffill()
    )

    daily_results['liquid_result_aggr'] = (
        daily_results['liquid_result']
        .cumsum()
    )

    daily_results['liquid_result_usd_aggr'] = (
        daily_results['liquid_result_aggr']
        * daily_results['price'].astype(float)
    )

    daily_results['capital'] = (
        daily_results['liquid_result_usd_aggr']
        / initial_capital
    ) * 100

    daily_results['is_profit'] = np.where(
        (daily_results['liquid_result_usd_aggr'].diff() > 0),
        1, 0
    )

    return daily_results
