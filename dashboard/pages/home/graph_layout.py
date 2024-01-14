import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class GraphLayout:
    """
    A class for creating interactive line charts and layouts for
    financial data visualization.

    Methods:
    --------
    - __init__(
        self, data_frame: pd.DataFrame, symbol: str, \
        interval: str, api: str \
        ): Constructor method for initializing the GraphLayout class.
    - fig_layout(self, fig, column): Configure layout settings for the \
        figure.
    - custom_fig_layout(self, fig, column): Configure custom layout \
        settings for the figure.
    - plot_cumulative_results(self) -> go.Figure: \
        Plot cumulative results line chart.
    - plot_single_linechart(self, column) -> go.Figure: \
        Plot a single line chart for a specified column.
    - plot_close(self) -> go.Figure: Plot line chart for closing prices.
    - grouped_lines(self) -> go.Figure: Plot grouped lines for \
        multiple columns.

    Attributes:
    -----------
    - data_frame: pd.DataFrame
        The financial DataFrame for visualization.
    - symbol: str
        The symbol for the financial data.
    - interval: str
        The time interval for the financial data.
    - api: str
        The API type for the financial data.
    """
    def __init__(
        self,
        data_frame: pd.DataFrame,
        symbol: str,
        interval: str,
        api: str,
        normalization: float = 2.0,
    ):
        """
        Constructor method for initializing the GraphLayout class.

        Parameters:
        -----------
        data_frame : pd.DataFrame
            The financial DataFrame for visualization.
        symbol : str
            The symbol for the financial data.
        interval : str
            The time interval for the financial data.
        api : str
            The API type for the financial data.
        """
        self.data_frame = data_frame
        self.tranp_color = "rgba(0,0,0,0)"
        self.title_color = "rgba(255,255,255,0.85)"
        self.label_color = "rgba(255,255,255,0.65)"
        self.primary_color = "#8bbb11"
        self.grid_color = "#595959"
        self.symbol = symbol
        self.interval = interval
        self.api = api
        self.normalization = normalization

