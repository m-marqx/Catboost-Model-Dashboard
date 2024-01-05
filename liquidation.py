"""
Module for calculating liquidation values.
"""


class Liquidation:
    """
    Class for calculating liquidation values.

    Parameters:
    ----------
    taker_fee : float
        The taker fee percentage.
    maintenance_margin : float
        The maintenance margin percentage.

    Methods:
    -------
    _initial_margin(leverage, funding_rate)
        Calculates the initial margin.
    calc_sell(entry, leverage, funding_rate)
        Calculates the liquidation value for a sell order.
    calc_buy(entry, leverage, funding_rate)
        Calculates the liquidation value and bankrupt price for a buy
        order.
    """

    def __init__(
        self,
        taker_fee: float = 0.05,
        maintenance_margin: float = 0.40,
    ) -> None:
        """
        Initialize a Liquidation object.

        Parameters
        ----------
        taker_fee : float, optional
            The taker fee percentage, expressed as a decimal.
            (default: 0.05)
        maintenance_margin : float, optional
            The maintenance margin percentage, expressed as a
            decimal.
            (default: 0.40)
        """
        self.taker_fee = taker_fee / 100
        self.maintenance_margin = (maintenance_margin / 100) - self.taker_fee

