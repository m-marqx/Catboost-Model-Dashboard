import dash_bootstrap_components as dbc
from dash import dcc, html

menu = html.Div(
    dbc.NavLink(
        "DASHBOARD",
        href="/",
        active=True,
        id="home",
    ), className="nav-start"
)

lang_menu = dbc.Col(
    [
        dbc.NavItem(
            dbc.NavLink(
                "EN",
                href="?lang=en_US",
                id="en_US_lang",
                class_name="nav-link-lang",
                n_clicks_timestamp=1,
            )
        ),
        dbc.NavItem(
            dbc.NavLink(
                "PT",
                href="?lang=pt_BR",
                id="pt_BR_lang",
                class_name="nav-link-lang last",
                n_clicks_timestamp=0,
            )
        ),
        dcc.Store(id="lang_selection", storage_type="local"),
    ],
    id="lang_menu",
    width=5,
)

def model_buttons(lang):
    return html.Div([
    html.Button(
        lang["RUN_MODEL"],
        id="dev_run_model",
        style={
            "border-top-left-radius": "2svh",
            "border-bottom-left-radius": "2svh",
            "border-top-right-radius": "0px",
            "border-bottom-right-radius": "0px",
        },
        className="btn-primary",
    ),
    html.Button(
        lang["CANCEL_MODEL"],
        id="dev_cancel_model",
        style={
            "border-top-left-radius": "0px",
            "border-bottom-left-radius": "0px",
            "border-top-right-radius": "2svh",
            "border-bottom-right-radius": "2svh",
        },
        className="btn-primary",
        disabled=True,
    ),
], className="model-buttons")

model_upload = dcc.Upload(
    id="upload-data",
    children="Upload Model",
    style={
        "margin": "0 0.5svh 0 0.5svh",
        "display": "grid",
        "align-content": "center",
    },
    className="model-btn btn-outline-secondary",
)

def navbar_components(lang):
    ml_buttons = model_buttons(lang)
    return dbc.Navbar(
        [
            dbc.Collapse([menu, ml_buttons, html.Div()], id="navbar-collapse", navbar=True),
            dbc.DropdownMenu(
                lang_menu,
                id="navbar-dropdown",
                nav=True,
                align_end=True,
                label="文/A",
            ),
            model_upload,
        ],
    )
