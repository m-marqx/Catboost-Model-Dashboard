import unittest

from pandas import Interval
import pandas as pd

def assert_count_series(
    test_count: dict[str, dict[Interval, int]],
    expected_count: dict[str, dict[Interval, int]],
):
    """
    Compare two dictionaries of count values and assert that their
    corresponding series are equal.

    Parameters
    ----------
    - test_count (dict[str, dict[Interval, int]]): The dictionary
    containing the test count values.
    - expected_count (dict[str, dict[Interval, int]]): The dictionary
    containing the expected count values.
    """
    test_count_series: dict[str, pd.Series] = {}
    expected_count_series: dict[str, pd.Series] = {}

    for key, value in zip(test_count.keys(), test_count.values()):
        test_count_series[key] = pd.Series(value)

    for key, value in zip(test_count.keys(), expected_count.values()):
        expected_count_series[key] = pd.Series(value)

    test_count_concat: pd.Series = pd.concat(test_count_series)
    expected_count_concat: pd.Series = pd.concat(expected_count_series)

    pd.testing.assert_series_equal(test_count_concat, expected_count_concat)


class TestAssertCount(unittest.TestCase):
    def setUp(self) -> None:
        self.test_count = {
            "count": {
                Interval(42.538, 51.888, closed="right"): 918,
                Interval(51.888, 61.237, closed="right"): 751,
                Interval(33.189, 42.538, closed="right"): 700,
                Interval(61.237, 70.586, closed="right"): 576,
                Interval(70.586, 79.936, closed="right"): 433,
                Interval(23.84, 33.189, closed="right"): 345,
                Interval(79.936, 89.285, closed="right"): 322,
                Interval(89.285, 98.634, closed="right"): 147,
                Interval(14.49, 23.84, closed="right"): 126,
                Interval(5.045999999999999, 14.49, closed="right"): 43,
            },
            "feat_count": {
                Interval(0.8, 1.6, closed="right"): 637,
                Interval(3.2, 4.0, closed="right"): 559,
                Interval(4.8, 5.6, closed="right"): 550,
                Interval(-0.009000000000000001, 0.8, closed="right"): 507,
                Interval(7.2, 8.0, closed="right"): 474,
                Interval(2.4, 3.2, closed="right"): 456,
                Interval(1.6, 2.4, closed="right"): 455,
                Interval(5.6, 6.4, closed="right"): 450,
                Interval(6.4, 7.2, closed="right"): 273,
                Interval(4.0, 4.8, closed="right"): 0,
            },
        }

    def test_assert_count_series(self):
        expected_count = {
            "count": {
                Interval(42.538, 51.888, closed="right"): 918,
                Interval(51.888, 61.237, closed="right"): 751,
                Interval(33.189, 42.538, closed="right"): 700,
                Interval(61.237, 70.586, closed="right"): 576,
                Interval(70.586, 79.936, closed="right"): 433,
                Interval(23.84, 33.189, closed="right"): 345,
                Interval(79.936, 89.285, closed="right"): 322,
                Interval(89.285, 98.634, closed="right"): 147,
                Interval(14.49, 23.84, closed="right"): 126,
                Interval(5.045999999999999, 14.49, closed="right"): 43,
            },
            "feat_count": {
                Interval(0.8, 1.6, closed="right"): 637,
                Interval(3.2, 4.0, closed="right"): 559,
                Interval(4.8, 5.6, closed="right"): 550,
                Interval(-0.009000000000000001, 0.8, closed="right"): 507,
                Interval(7.2, 8.0, closed="right"): 474,
                Interval(2.4, 3.2, closed="right"): 456,
                Interval(1.6, 2.4, closed="right"): 455,
                Interval(5.6, 6.4, closed="right"): 450,
                Interval(6.4, 7.2, closed="right"): 273,
                Interval(4.0, 4.8, closed="right"): 0,
            },
        }

        assert_count_series(self.test_count, expected_count)

    def test_assert_count_invalid(self):
        expected_count = {
            "count": {
                Interval(42.538, 51.888, closed="right"): 0,
                Interval(51.888, 61.237, closed="right"): 0,
                Interval(33.189, 42.538, closed="right"): 0,
                Interval(61.237, 70.586, closed="right"): 0,
                Interval(70.586, 79.936, closed="right"): 0,
                Interval(23.84, 33.189, closed="right"): 0,
                Interval(79.936, 89.285, closed="right"): 0,
                Interval(89.285, 98.634, closed="right"): 0,
                Interval(14.49, 23.84, closed="right"): 0,
                Interval(5.045999999999999, 14.49, closed="right"): 0,
            },
            "feat_count": {
                Interval(0.8, 1.6, closed="right"): 0,
                Interval(3.2, 4.0, closed="right"): 0,
                Interval(4.8, 5.6, closed="right"): 0,
                Interval(-0.009000000000000001, 0.8, closed="right"): 0,
                Interval(7.2, 8.0, closed="right"): 0,
                Interval(2.4, 3.2, closed="right"): 0,
                Interval(1.6, 2.4, closed="right"): 0,
                Interval(5.6, 6.4, closed="right"): 0,
                Interval(6.4, 7.2, closed="right"): 0,
                Interval(4.0, 4.8, closed="right"): 0,
            },
        }

        self.assertRaises(
            AssertionError,
            assert_count_series,
            self.test_count,
            expected_count,
        )

if __name__ == '__main__':
    unittest.main()