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

        transaction = {
            "from": first_coin_total,
            "to": second_coin_total,
            "USD Price": usd_price,
        }

        if coin_name:
            transaction["from_coin_name"] = first_coin_name
            transaction["to_coin_name"] = second_coin_name

        return transaction

        """
        Retrieves all transactions for a given wallet address.

        Parameters
        ----------
        wallet : str
            The wallet address to retrieve transactions for.

        Returns
        -------
        list of dict
            A list of dictionaries, each containing details of a swap
            transaction.
        """
        url = f"{self.blockscout_api_url}/addresses/{wallet}/transactions"

        response = requests.get(
            url, params={"filter": "to | from"}, timeout=10
        )
        items = response.json()["items"]

        swap_name = "processRouteWithTransferValueOutput"

        swap_count = 0
        swaps_df = pd.DataFrame(items).query(f"method == {swap_name}")

        swap_qty = swaps_df.shape[0]

        swaps = []

        self.logger.info("searching swaps...")

        for x in swaps_df.index.tolist():
            is_swap = items[x]["method"] == swap_name

            if is_swap:
                txn_fee = literal_eval(items[x]["priority_fee"]) / 10**18

                swap = self.get_transactions(items[x]["hash"], False)
                swap_count += 1

                self.logger.info(
                    "%.2f%% complete", (swap_count / swap_qty) * 100
                )

                swap["txn_fee"] = txn_fee
                swaps.append(swap)

            if swap_count == 0:
                logging.info("no swaps found")

            elif swap_count < swap_qty:
                logging.info("not all swaps found")

            else:
                logging.info("all swaps found")

            logging.info("swaps total: %d", swap_qty)
            logging.info("all searches complete")

        return swaps
