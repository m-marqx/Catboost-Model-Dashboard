from dash import dcc, html, register_page
import dash_bootstrap_components as dbc
from dashboard.pages.lang import en_US, pt_BR


register_page(
    __name__,
    path="/",
    title="Machine Learning",
    name="Machine Learning",
    description="Machine Learning simple backtest.",
)

def layout(lang="en_US"):
    if lang == "en_US":
        lang = en_US
    elif lang == "pt_BR":
        lang = pt_BR

    return [
        dbc.Container(
            [
                dbc.Row([
                    dbc.Col(
                        [
                            dbc.Row([
                                    dcc.Graph(
                                        id="dev_ml_results",
                                        figure={
                                            "layout": {
                                                "paper_bgcolor": "rgba(0,0,0,0)",
                                                "plot_bgcolor": "rgba(0,0,0,0)",
                                                "xaxis": {
                                                    "showgrid": False,
                                                    "showticklabels": False,
                                                    "zeroline": False,
                                                    "title": "",
                                                },
                                                "yaxis": {
                                                    "showticklabels": False,
                                                    "zeroline": False,
                                                    "gridcolor": "#595959",
                                                    "griddash": "dash",
                                                    "title": "",
                                                    "exponentformat": "none",
                                                },
                                            }
                                        },
                                        className="graph",
                                        style={"width": "79%", "margin": "1svh 0svh 0px 1svh"}
                                    ),
                                    html.P(id='dev_ml_results2', style={"width": "20%", "margin-bottom": "0px"}),
                            ]),
                        ],
                        width=10,
                    ),
                    dbc.Col(
                        [
                            dbc.Col(
                                [
                                    dbc.Row(
                                        [
                                            html.P(
                                                id="dev_progress_bar",
                                                className="progress-inf",
                                            ),
                                        ],
                                        style={
                                            "margin-left": "auto",
                                            "margin-right": "auto",
                                        },
                                    ),
                                    dbc.Spinner(
                                        [
                                            html.P(
                                                lang["EMPTY_RESULT"],
                                                id="dev_model_text_output",
                                                style={
                                                    "margin-top": "1svh",
                                                    "border-radius": "2svh",
                                                    "display": "flex",
                                                    "flex-wrap": "wrap",
                                                    "align-content": "flex-start",
                                                    "justify-content": "center",
                                                }),
                                        ],
                                        id="dev_text_model_spinner",
                                        color="primary",
                                        spinner_class_name="spinner-loader",
                                    ),
                                    html.P(id="dev_signal_output", style={"margin-bottom": "0px"}),
                                    html.P(id="dev_new_signal_output"),
                                ]
                            ),
                            dbc.Col(id="dev_table_container"),
                        ],
                        width=2,
                        style={"max-height": "38svh"}
                    ),
                ]),
                dbc.Row([
                    dcc.Graph(
                        id="win_rate_graph",
                        figure={
                            "layout": {
                                "paper_bgcolor": "rgba(0,0,0,0)",
                                "plot_bgcolor": "rgba(0,0,0,0)",
                                "xaxis": {
                                    "showgrid": False,
                                    "showticklabels": False,
                                    "zeroline": False,
                                    "title": "",
                                },
                                "yaxis": {
                                    "showticklabels": False,
                                    "zeroline": False,
                                    "gridcolor": "#595959",
                                    "griddash": "dash",
                                    "title": "",
                                    "exponentformat": "none",
                                },
                            }
                        },
                        className="graph mini-graph",
                    ),
                    dcc.Graph(
                        id="expected_return_graph",
                        figure={
                            "layout": {
                                "paper_bgcolor": "rgba(0,0,0,0)",
                                "plot_bgcolor": "rgba(0,0,0,0)",
                                "xaxis": {
                                    "showgrid": False,
                                    "showticklabels": False,
                                    "zeroline": False,
                                    "title": "",
                                },
                                "yaxis": {
                                    "showticklabels": False,
                                    "zeroline": False,
                                    "gridcolor": "#595959",
                                    "griddash": "dash",
                                    "title": "",
                                    "exponentformat": "none",
                                },
                            }
                        },
                        className="graph mini-graph",
                    ),
                ]),
                dbc.Row([
                    dcc.Graph(
                        id="drawdown_graph",
                        figure={
                            "layout": {
                                "paper_bgcolor": "rgba(0,0,0,0)",
                                "plot_bgcolor": "rgba(0,0,0,0)",
                                "xaxis": {
                                    "showgrid": False,
                                    "showticklabels": False,
                                    "zeroline": False,
                                    "title": "",
                                },
                                "yaxis": {
                                    "showticklabels": False,
                                    "zeroline": False,
                                    "gridcolor": "#595959",
                                    "griddash": "dash",
                                    "title": "",
                                    "exponentformat": "none",
                                },
                            }
                        },
                        className="graph mini-graph",
                    ),
                    dcc.Graph(
                        id="payoff_graph",
                        figure={
                            "layout": {
                                "paper_bgcolor": "rgba(0,0,0,0)",
                                "plot_bgcolor": "rgba(0,0,0,0)",
                                "xaxis": {
                                    "showgrid": False,
                                    "showticklabels": False,
                                    "zeroline": False,
                                    "title": "",
                                },
                                "yaxis": {
                                    "showticklabels": False,
                                    "zeroline": False,
                                    "gridcolor": "#595959",
                                    "griddash": "dash",
                                    "title": "",
                                    "exponentformat": "none",
                                },
                            }
                        },
                        className="graph mini-graph",
                    ),
                ]),
            ],
            fluid=True,
            style={"font-family": "Open Sans"},
        )
    ]
