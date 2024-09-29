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

    def get_account_transactions(self, wallet: str, coin_names: bool = False):
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
        swaps_df = pd.DataFrame(items).query(
            "method == 'processRouteWithTransferValueOutput'"
        )

        fees_df = (
            pd.DataFrame(
                swaps_df["fee"].to_list(),
                index=swaps_df.index,
            )["value"].astype(float) / 10**18
        )

        swap_qty = swaps_df.shape[0]

        swaps = []

        self.logger.info("searching swaps...")

        for x in swaps_df.index.tolist():
            is_swap = items[x]["method"] == swap_name

            if is_swap:
                swap = self.get_transactions(items[x]["hash"], coin_names)
                swap_count += 1

                self.logger.info(
                    "%.2f%% complete", (swap_count / swap_qty) * 100
                )

                swap["txn_fee"] = fees_df.loc[x]
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

    def find_sells(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify and return sell transactions from a DataFrame of
        transactions. This method examines each transaction in the
        provided DataFrame to determine if it represents a sale. A sale
        is defined as a transaction where the 'from' address of one
        transaction matches the 'to' address of another transaction, and
        the 'from_coin_name' matches the 'to_coin_name'.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            A DataFrame containing transaction data. The DataFrame must
            include the following columns: 'from', 'to',
            'from_coin_name', and 'to_coin_name'.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing only the transactions identified as
            sells. The returned DataFrame includes the columns: 'from',
            'to', 'from_coin_name', and 'to_coin_name'.
        """

        def is_sale(row, df):
            is_exact_same_amout = df["from"] == row["to"]
            is_same_coin = df["from_coin_name"] == row["to_coin_name"]
            return (is_exact_same_amout & is_same_coin).any()

        sells_mask = transactions_df.apply(is_sale, df=transactions_df, axis=1)
        sells = transactions_df[sells_mask].reset_index(drop=True)

        return sells[["from", "to", "from_coin_name", "to_coin_name"]]

    def find_buys(self, transactions_df: pd.DataFrame):
        """
        Identify buy transactions from a DataFrame of transactions.
        This method analyzes a DataFrame of transactions and identifies
        which transactions are buy transactions. A buy transaction is
        defined as a transaction where the 'to' address of one
        transaction matches the 'from' address of another transaction
        and the 'to_coin_name' matches the 'from_coin_name'.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            A DataFrame containing transaction data. The DataFrame must
            include the following columns: 'from', 'to',
            'from_coin_name', and 'to_coin_name'.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing only the buy transactions. The
            returned DataFrame includes the columns: 'from', 'to',
            'from_coin_name', and 'to_coin_name'.
        """
        def is_buy(row, df):
            is_exact_same_amout = df["to"] == row["from"]
            is_same_coin = df["to_coin_name"] == row["from_coin_name"]
            return (is_exact_same_amout & is_same_coin).any()

        buys_mask = transactions_df.apply(is_buy, df=transactions_df, axis=1)
        buys = transactions_df[buys_mask].reset_index(drop=True)

        return buys[["from", "to", "from_coin_name", "to_coin_name"]]

    def find_trades(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify and return all trade transactions from a DataFrame of
        transactions. This method combines sell and buy transactions
        into a single DataFrame.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            A DataFrame containing transaction data. The DataFrame must
            include the following columns: 'from', 'to',
            'from_coin_name', and 'to_coin_name'.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing all identified trade transactions,
            sorted by their original index. The returned DataFrame
            includes the columns: 'from', 'to', 'from_coin_name', and
            'to_coin_name', and has an index named 'Trade Number'.
        """
        sells = self.find_sells(transactions_df)
        buys = self.find_buys(transactions_df)

        trades = pd.concat([sells, buys]).sort_index()
        trades.index.name = "Trade Number"

        return trades
