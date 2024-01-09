"""
Module for time-related utility functions.

This module provides functions for converting Binance interval strings
to milliseconds.

Functions:
- interval_to_milliseconds: Convert a Binance interval string to
milliseconds.
"""

def interval_to_milliseconds(interval: str):
    """Convert a Binance interval string to milliseconds

    :param interval: Binance interval string, e.g.:
    1s, 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w

    :return:
        int value of interval in milliseconds
        None if interval prefix is not a decimal integer
        None if interval suffix is not one of m, h, d, w

    """
    seconds_per_unit: dict = {
        "s": 1,
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
    }
    try:
        return int(interval[:-1]) * seconds_per_unit[interval[-1]] * 1000
    except (ValueError, KeyError):
        return None
