import unittest
from unittest.mock import patch, Mock
import pandas as pd
from api.blockscout_api import BlockscoutAPI


class TestBlockscoutAPI(unittest.TestCase):
    def setUp(self):
        self.api = BlockscoutAPI(verbose=False)
        self.transactions = (
            pd.DataFrame(
                [
                    {
                        "from": 8.676632,
                        "to": 0.0001332,
                        "USD Price": 65139.879879879874,
                        "from_coin_name": "USDT",
                        "to_coin_name": "WBTC",
                        "txn_fee": 0.00643746001609365,
                    },
                    {
                        "from": 0.00013673,
                        "to": 8.654798,
                        "USD Price": 63298.45681269655,
                        "from_coin_name": "WBTC",
                        "to_coin_name": "USDT",
                        "txn_fee": 0.004750170007600272,
                    },
                    {
                        "from": 8.0,
                        "to": 0.00012647,
                        "USD Price": 63256.10816794496,
                        "from_coin_name": "USDT",
                        "to_coin_name": "WBTC",
                        "txn_fee": 0.006340920010145472,
                    },
                    {
                        "from": 8.0,
                        "to": 0.00012689,
                        "USD Price": 63046.733391126174,
                        "from_coin_name": "USDT",
                        "to_coin_name": "WBTC",
                        "txn_fee": 0.007258425415653105,
                    },
                    {
                        "from": 8.0,
                        "to": 0.00013673,
                        "USD Price": 58509.471220653846,
                        "from_coin_name": "USDT",
                        "to_coin_name": "WBTC",
                        "txn_fee": 0.007532250006276875,
                    },
                    {
                        "from": 63.0,
                        "to": 24.021834,
                        "USD Price": 0.38129895238095235,
                        "from_coin_name": "WMATIC",
                        "to_coin_name": "USDT",
                        "txn_fee": 0.004755630003645983,
                    },
                ]
            )
            .iloc[::-1]
            .reset_index(drop=True)
        )

    @patch("requests.get")
    def test_get_transactions(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "8676632"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "13320"},
                },
            ]
        }

        mock_get.return_value = mock_response

        result = self.api.get_transactions(
            "0x1",
            False,
        )

        expected_result = {
            "from": 8.676632,
            "to": 0.0001332,
            "USD Price": 65139.879879879874,
        }

        self.assertDictEqual(result, expected_result)

    @patch("requests.get")
    def test_get_transactions_coin_names(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            "items": [
                {
                    "token": {"symbol": "USDT"},
                    "total": {"decimals": "6", "value": "8676632"},
                },
                {
                    "token": {"symbol": "WBTC"},
                    "total": {"decimals": "8", "value": "13320"},
                },
            ]
        }

        mock_get.return_value = mock_response

        result = self.api.get_transactions(
            "0x1",
            True,
        )

        expected_result = {
            "from": 8.676632,
            "to": 0.0001332,
            "USD Price": 65139.879879879874,
            "from_coin_name": "USDT",
            "to_coin_name": "WBTC",
        }

        self.assertDictEqual(result, expected_result)
