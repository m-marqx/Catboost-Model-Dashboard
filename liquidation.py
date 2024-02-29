"""
Module for calculating liquidation values.
"""

from typing import Literal


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

    calculate_short(entry, leverage, funding_rate)
        Calculates the liquidation value for a sell order.

    calculate_long(entry, leverage, funding_rate)
        Calculates the liquidation value and bankrupt price for a buy
        order.
    """

    def __init__(
        self,
        taker_fee: float = 0.05,
        maintenance_margin: float = 0.40,
        return_type: Literal['both', 'liq', 'bankrupt'] = 'bankrupt',
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
        self.return_type = return_type

    def _initial_margin(self, leverage: float, funding_rate: float) -> float:
        """
        Calculates the initial margin.

        Parameters:
        ----------
        leverage : float
            The leverage value.

        funding_rate : float
            The funding rate.

        Returns:
        -------
        float
            The initial margin value.
        """
        return (1 + funding_rate) / leverage - self.taker_fee * 2

    def calculate_short(
        self,
        entry: float,
        leverage: float,
        funding_rate: float,
    ) -> tuple:
        """
        Calculates the liquidation value for a short position.

        Parameters:
        ----------
        entry : float
            The entry price.

        leverage : float
            The leverage value.

        funding_rate : float
            The funding rate.

        Returns:
        -------
        float
            The liquidation value.
        """
        bankrupt = entry / (1 - self._initial_margin(leverage, funding_rate))
        liq = bankrupt - (entry * (self.maintenance_margin - funding_rate))
        if self.return_type == 'both':
            return liq, bankrupt
        if self.return_type == 'liq':
            return liq
        return bankrupt

    def calculate_long(
        self,
        entry: float,
        leverage: float,
        funding_rate: float,
    ) -> tuple:  #! Warning: this method isn't accurate
        """
        Calculates the liquidation value and bankrupt price for a long
        position.

        Parameters:
        ----------
        entry : float
            The entry price.

        leverage : float
            The leverage value.

        funding_rate : float
            The funding rate.

        Returns:
        -------
        tuple
            The liquidation value and bankrupt price.
        """
        bankrupt = entry / (1 + self._initial_margin(leverage, funding_rate))
        liq = bankrupt + (entry * (self.maintenance_margin + funding_rate))

        if self.return_type == 'both':
            return liq, bankrupt
        if self.return_type == 'liq':
            return liq
        return bankrupt
