import os
import dill

import ccxt
import dash
from dash import Output, Input, State, callback
import dash_ag_grid as dag
import pandas as pd
import numpy as np

from api.ccxt_api import CcxtAPI

from machine_learning.ml_utils import DataHandler

from dashboard.pages.home.graph_layout import GraphLayout
from dashboard.pages.home.graphs import display_linechart, add_outlier_lines


class DevRunModel:
    @callback(
        Output("upload-data", "children"),
        Output("upload-data", "className"),
        Input('upload-data', 'filename')
    )
    def update_upload_button(filename):
        if filename is not None:
            return filename, "btn btn-outline-secondary-filled"
        return "Upload Model", "btn btn-outline-secondary"

    @callback(
        Output("dev_model_text_output", "children"),
        Output("dev_model_text_output", "className"),
        Output("dev_signal_output", "children"),
        Output("dev_progress_bar", "className"),
        # Graph Charts
        Output("dev_ml_results", "figure"),
        Output("dev_ml_results2", "figure"),
        Output("drawdown_graph", "figure"),
        Output("expected_return_graph", "figure"),
        Output("payoff_graph", "figure"),
        Output("win_rate_graph", "figure"),
        inputs=[
            Input("dev_run_model", "n_clicks"),
            State('upload-data', 'filename'),
        ],
        background=True,
        cancel=Input("dev_cancel_model", "n_clicks"),
        running=[
            (Output("dev_run_model", "disabled"), True, False),
            (Output("dev_cancel_model", "disabled"), False, True),
            (Output("dev_progress_bar", "className"), "progress-info", "hidden"),
            (Output("dev_signal_output", "className"), "hidden", ""),
        ],
        progress=Output("dev_progress_bar", "children"),
        prevent_initial_call=True,
    )
    def get_model_predict(
        set_progress,
        run_button,  # run_button is necessary to track run_model clicks
        model_filename,
    ):
        ctx = dash.callback_context

        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        if "run_model" in ctx.triggered[0]["prop_id"]:
            set_progress("Creating model...")

            print(model_filename)
            with open(f"models/{model_filename}", "rb") as file:
                model_pickled = dill.load(file)

            result = model_pickled()
            validation_date = "23-03-2021"

            model_fig = GraphLayout(
                result["Liquid_Result"].loc[validation_date:].cumsum().to_frame(),
                "Model Result",
                "1D",
                "spot",
            ).plot_single_linechart("Liquid_Result")

            api_key = os.getenv("APIKEY")
            secret_key = os.getenv("SECRETKEY")

            keys = {"apiKey": api_key, "secret": secret_key}

            coin_api = CcxtAPI(
                symbol="BTCUSD_PERP",
                interval="1d",
                exchange=ccxt.binancecoinm(keys),
                since=None,
            )

            account_result = coin_api.get_account_result(130)

            account_fig = GraphLayout(
                account_result,
                "Account Result",
                "1D",
                "spot",
            ).plot_single_linechart("liquid_result_usd_aggr")

            win_rate = display_linechart(
                result["Gross_Return"],
                result["Liquid_Return"],
                validation_date,
                "winrate",
                "full",
                get_data=True,
            )

            win_rate_fig = GraphLayout(
                win_rate.loc[validation_date:],
                "Win Rate",
                "30D",
                "spot",
                0.618,
            ).plot_single_linechart("Liquid_Return")

            add_outlier_lines(win_rate, win_rate_fig)

            expected_return = display_linechart(
                result["Gross_Return"],
                result["Liquid_Return"],
                validation_date,
                "expected_return",
                "full",
                get_data=True,
            )

            expected_return_fig = GraphLayout(
                expected_return.loc[validation_date:],
                "Expected Return",
                "30D",
                "spot",
                0.5,
            ).plot_single_linechart("Liquid_Return")

            add_outlier_lines(expected_return, expected_return_fig)

            drawdown = display_linechart(
                result["Gross_Return"],
                result["Liquid_Return"],
                validation_date,
                "drawdown",
                "full",
                get_data=True,
            )

            drawdown_fig = GraphLayout(
                drawdown.loc[validation_date:],
                "Drawdown",
                "30D",
                "spot",
                0.75,
            ).plot_single_linechart("Liquid_Return")

            add_outlier_lines(drawdown, drawdown_fig, min_value=0)

            payoff = display_linechart(
                result["Gross_Return"],
                result["Liquid_Return"],
                validation_date,
                "payoff_mean",
                "full",
                get_data=True,
            )

            payoff_fig = GraphLayout(
                payoff.loc[validation_date:],
                "Payoff",
                "30D",
                "spot",
                0.618,
            ).plot_single_linechart("Liquid_Return")

            add_outlier_lines(payoff, payoff_fig)

            print(result.columns)
            predict = result["Predict"].copy()
            predict.index = predict.index.date
            predict = predict.rename_axis("date")

            print(DataHandler(result).result_metrics("Result"))

            recommendation = pd.DataFrame(predict).shift().reset_index()

            recommendation["Predict"] = np.where(
                recommendation["Predict"] > 0,
                "Long",
                np.where(recommendation["Predict"] == 0, "Do Nothing", "Short")
            )

            if predict.iloc[-1] > 0:
                signal = "Long"
            elif predict.iloc[-1] == 0:
                signal = "Do Nothing"
            elif predict.iloc[-1] < 0:
                signal = "Short"
            else:
                signal = "Error"

            new_signal = pd.DataFrame({"Predict": signal}, index=["Unconfirmed"])

            new_signal = new_signal.reset_index()
            new_signal.columns = ["date"] + list(new_signal.columns[1:])

            recommendation = pd.concat([recommendation, new_signal], axis=0)
            cols_def = [{"field": i} for i in recommendation.columns]
            cols_def[0]["sort"] = "desc"

            recommendation_table = dag.AgGrid(
                rowData=recommendation.to_dict("records"),
                getRowId="params.data.date",
                columnDefs=cols_def,
                defaultColDef={"resizable": True, "sortable": True, "filter": True},
                columnSize="responsiveSizeToFit",
                dashGridOptions={"pagination": False},
                className="bigger-table ag-theme-alpine-dark responsive",
                style={"z-index": "0"},
            )

            line_grid = dict(line_width=1, line_dash="dash", line_color="#595959")

            model_fig.add_vline(validation_date, **line_grid)
            model_fig.add_vline("2024-02-14 00:00:00", **line_grid)

            return (
                "",
                "hidden",
                recommendation_table,
                "hidden",
                # Graph Charts
                model_fig,
                account_fig,
                drawdown_fig,
                expected_return_fig,
                payoff_fig,
                win_rate_fig,
            )
