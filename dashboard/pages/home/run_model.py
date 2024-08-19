import dill

import dash
from dash import Output, Input, State, callback
import dash_ag_grid as dag
import pandas as pd
import numpy as np

from machine_learning.ml_utils import DataHandler

from dashboard.pages.home.graph_layout import GraphLayout
from dashboard.pages.home.graphs import (
    display_linechart,
    add_outlier_lines,
    calculate_sequencial_results,
)


class DevRunModel:
    @callback(
        Output("upload-data", "children"),
        Output("upload-data", "className"),
        Input("upload-data", "filename"),
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
        Output("dev_ml_results2", "children"),
        Output("drawdown_graph", "figure"),
        Output("expected_return_graph", "figure"),
        Output("payoff_graph", "figure"),
        Output("win_rate_graph", "figure"),
        inputs=[
            Input("dev_run_model", "n_clicks"),
            State("upload-data", "filename"),
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

            if result["Liquid_Result"].min() < 0:
                liquid_return = (
                    result["Liquid_Result"]
                    .loc[validation_date:]
                    .cumsum()
                    .rename("Liquid_Return")
                    .to_frame()
                )

            else:
                liquid_return = (
                    result["Liquid_Result"]
                    .loc[validation_date:]
                    .cumprod()
                    .rename("Liquid_Return")
                    .to_frame()
                )

            model_fig = GraphLayout(
                liquid_return,
                "Model Result",
                "1D",
                "spot",
            ).plot_single_linechart("Liquid_Return")

            win_rate = display_linechart(
                liquid_return["Liquid_Return"],
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

            drawdown = 1 - liquid_return / liquid_return.cummax()

            drawdown_fig = GraphLayout(
                drawdown.loc[validation_date:],
                "Drawdown",
                "Validation",
                "spot",
                0.75,
            ).plot_single_linechart("Liquid_Return")

            add_outlier_lines(drawdown, drawdown_fig, min_value=0)

            sequential_results = calculate_sequencial_results(
                result["Liquid_Result"].to_frame(),
                1,
            )['sequential_count'].to_frame()

            sequential_results_fig = GraphLayout(
                sequential_results.loc[validation_date:].ffill().fillna(0),
                "Sequential Results",
                "Validation",
                "spot",
                0.75,
            ).plot_single_linechart(sequential_results.columns[-1])

            add_outlier_lines(sequential_results, sequential_results_fig)

            print(result.columns)
            predict = result["Predict"].copy()
            predict.index = predict.index.date
            predict = predict.rename_axis("date")

            print(DataHandler(result).result_metrics("Result"))

            recommendation = pd.DataFrame(predict).shift().reset_index()

            recommendation["Predict"] = np.where(
                recommendation["Predict"] > 0,
                "Long",
                np.where(recommendation["Predict"] == 0, "Do Nothing", "Short"),
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

            sequential_count = (
                sequential_results["sequential_count"]
                .rename("sequence")
                .value_counts()
                .sort_index(ascending=False)
                .reset_index()
                .query("sequence != 0")
                .reset_index(drop=True)
            )

            gain_sequences = np.array([])
            loss_sequences = np.array([])

            for x in sequential_count["sequence"].to_numpy():
                if x > 0:
                    x = f"{x} Profits"
                    gain_sequences = np.append(gain_sequences, x)
                else:
                    x = f"{-x} Losses"
                    loss_sequences = np.append(loss_sequences, x)

            sequential_count["sequence"] = pd.concat(
                [
                    pd.Series(gain_sequences).rename("gain"),
                    pd.Series(loss_sequences).rename('loss')
                ],
                axis=0,
            ).reset_index(drop=True)

            # sequential_count = sequential_count.sort_values("sequence", ascending=False)
            cols_def = [{"field": i} for i in sequential_count.columns]
            # cols_def[0]["sort"] = "desc"

            sequential_results_table = dag.AgGrid(
                rowData=sequential_count.to_dict("records"),
                getRowId="params.data.count",
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
                sequential_results_table,
                drawdown_fig,
                expected_return_fig,
                sequential_results_fig,
                win_rate_fig,
            )
