from ast import literal_eval
import logging
import requests
import pandas as pd


class BlockscoutAPI:
    """
    A class to interact with the Blockscout API for retrieving
    transaction data.

    Parameters
    ----------
    verbose : bool
        If True, sets the logger to INFO level.

    Attributes
    ----------
    logger : logging.Logger
        Logger instance for logging information.
    blockscout_api_url : str
        Base URL for the Blockscout API.

    Methods
    -------
    get_transactions(txid, coin_name=False)
        Retrieves transaction details for a given transaction ID.
    get_account_transactions(wallet)
        Retrieves all transactions for a given wallet address.
    """

    def __init__(self, verbose: bool):
        self.logger = logging.getLogger("Blockscout_API")
        formatter = logging.Formatter(
            "%(levelname)s %(asctime)s: %(message)s", datefmt="%H:%M:%S"
        )

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        self.logger.addHandler(handler)
        self.logger.propagate = False

        if verbose:
            self.logger.setLevel(logging.INFO)

        self.blockscout_api_url = "https://polygon.blockscout.com/api/v2"

    def get_transactions(self, txid: str, coin_name: bool = False):
        """
        Retrieves transaction details for a given transaction ID.

        Parameters
        ----------
        txid : str
            The transaction ID to retrieve details for.
        coin_name : bool, optional
            If True, includes the coin names in the returned dictionary
            (default : False).

        Returns
        -------
        dict
            A dictionary containing the transaction details, including
            coin totals and USD price.
        """
        url = f"{self.blockscout_api_url}/transactions/{txid}/token-transfers"

        response = requests.get(url, params={"type": "ERC-20"}, timeout=10)
        data = response.json()

        first_coin_value = literal_eval(data["items"][0]["total"]["value"])
        first_coin_decimals = literal_eval(
            data["items"][0]["total"]["decimals"]
        )

        first_coin_total = first_coin_value / 10**first_coin_decimals

        second_coin_value = literal_eval(data["items"][-1]["total"]["value"])
        second_coin_decimals = literal_eval(
            data["items"][-1]["total"]["decimals"]
        )

        second_coin_total = second_coin_value / 10**second_coin_decimals

        first_coin_name = data["items"][0]["token"]["symbol"]
        second_coin_name = data["items"][-1]["token"]["symbol"]

        if first_coin_name.startswith("USD"):
            usd_price = first_coin_total / second_coin_total

        elif second_coin_name.startswith("USD"):
            usd_price = second_coin_total / first_coin_total

        else:
            usd_price = None

        if not coin_name:
            first_coin_name = "from"
            second_coin_name = "to"

        return {
            first_coin_name: first_coin_total,
            second_coin_name: second_coin_total,
            "USD Price": usd_price,
        }

