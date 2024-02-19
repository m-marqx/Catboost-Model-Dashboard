import os
import json

import ccxt
import dash
from dash import Output, Input, State, callback
import dash_ag_grid as dag
import tradingview_indicators as ta
import pandas as pd


from api.ccxt_api import CcxtAPI

from machine_learning.ml_utils import DataHandler
from machine_learning.feature_params import FeaturesParams, FeaturesParamsComplete
from machine_learning.features_creator import FeaturesCreator

from dashboard.pages.home.graph_layout import GraphLayout
from dashboard.pages.home.graphs import display_linechart, add_outlier_lines

class DevRunModel:
    @callback(
        Output("dev_model_text_output", "children"),
        Output("dev_model_text_output", "className"),
        Output("dev_signal_output", "children"),
        Output("dev_progress_bar", "className"),
        #Graph Charts
        Output("dev_ml_results", "figure"),
        Output("dev_ml_results2", "figure"),
        Output("drawdown_graph", "figure"),
        Output("expected_return_graph", "figure"),
        Output("payoff_graph", "figure"),
        Output("win_rate_graph", "figure"),
        inputs=[
        Input("dev_run_model", "n_clicks"),
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
        run_button, #run_button is necessary to track run_model clicks
    ):
        ctx = dash.callback_context

        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        if "run_model" in ctx.triggered[0]["prop_id"]:
            loading_model_str = "Loading model..."

            set_progress(loading_model_str)

            try:
                updated_dataset = pd.read_parquet("data/dataset_updated.parquet")
                capi = CcxtAPI(symbol="BTC/USDT", interval="1d", exchange=ccxt.binance())
                updated_dataset = capi.update_klines(updated_dataset).drop(columns="volume")

            except Exception as e:
                set_progress("")

                return (
                    e, "",
                    None,
                    None,
                    "",

                    #Graph Charts
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )

            set_progress("Creating model...")

            BTCUSD = DataHandler(updated_dataset.copy()).calculate_targets()
            BTCUSD["RSI79"] = ta.RSI(BTCUSD["high"].pct_change(), 79)

            split_params = dict(
                target_input=BTCUSD["Target_1_bin"],
                column="temp_indicator",
                log_values=True,
                threshold=0.50
            )

            split_params_H = dict(
                target_input=BTCUSD["Target_1_bin"],
                column="temp_indicator",
                log_values=True,
                threshold=0.52
            )

            split_params_L = dict(
                target_input=BTCUSD["Target_1_bin"],
                column="temp_indicator",
                log_values=True,
                threshold=0.48,
                higher_than_threshold=False,
            )

            split_params = FeaturesParams(**split_params)
            high_params = FeaturesParams(**split_params_H)
            low_params = FeaturesParams(**split_params_L)

            complete_params = FeaturesParamsComplete(
                split_features=split_params,
                high_features=high_params,
                low_features=low_params
            )

            validation_date = "2020-04-11 00:00:00"

            model_predict = FeaturesCreator(
                BTCUSD,
                BTCUSD["Target_1"],
                BTCUSD["RSI79"],
                complete_params,
                validation_date
            )

            model_predict.calculate_features("RSI79", 1527)
            model_predict.data_frame['rolling_ratio_std'] = model_predict.temp_indicator(
                json.loads(os.getenv('indicators'))['rolling_ratio'],
                'rolling_ratio',
                BTCUSD["RSI79"]
            )

            features = json.loads(os.getenv('features'))
            set_progress("Calculating model returns...")
            model_predict.calculate_features("rolling_ratio_std", 1527)

            result = (
                model_predict
                .calculate_results(features, save_model=False)
            )

            predict = result["Predict"].to_frame()

            model_fig = GraphLayout(
                result["Liquid_Result"].loc[validation_date:].cumsum().to_frame(),
                "Model Result",
                "1D",
                "spot",
            ).plot_single_linechart("Liquid_Result")

            api_key = os.getenv('APIKEY')
            secret_key = os.getenv('SECRETKEY')

            keys = {'apiKey': api_key,'secret': secret_key}

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
                BTCUSD['Return'], result['Liquid_Return'],
                validation_date,  "winrate", "full", get_data=True,
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
                BTCUSD['Return'], result['Liquid_Return'],
                validation_date,  "expected_return", "full", get_data=True,
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
                BTCUSD['Return'], result['Liquid_Return'],
                validation_date,  "drawdown", "full", get_data=True,
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
                BTCUSD['Return'], result['Liquid_Return'],
                validation_date, "payoff_mean", "full", get_data=True,
            )

            payoff_fig = GraphLayout(
                payoff.loc[validation_date:],
                "Payoff",
                "30D",
                "spot",
                0.618,
            ).plot_single_linechart("Liquid_Return")

            add_outlier_lines(payoff, payoff_fig)

            predict.index = predict.index.date
            predict = predict.rename_axis("date")

            print(DataHandler(result).result_metrics("Result"))

            recommendation = predict.copy().shift()
            recommendation = (
                recommendation
                .where(recommendation < 0, 'Long')
                .where(recommendation > 0, 'Short')
                    .reset_index()
            )

            new_signal = pd.DataFrame({"Unconfirmed" : predict.iloc[-1]}).T

            new_signal = (
                new_signal
                .where(new_signal < 0, "Long")
                .where(new_signal > 0, "Short")
                .reset_index()
            )

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

            line_grid = dict(
                line_width=1,
                line_dash="dash",
                line_color="#595959"
            )

            model_fig.add_vline(validation_date, **line_grid)
            model_fig.add_vline("2023-11-29 00:00:00", **line_grid)

            return (
                "", "hidden",
                recommendation_table,
                "hidden",

                #Graph Charts
                model_fig,
                account_fig,
                drawdown_fig,
                expected_return_fig,
                payoff_fig,
                win_rate_fig,
            )
