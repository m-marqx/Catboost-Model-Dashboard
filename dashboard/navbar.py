import dash_bootstrap_components as dbc
from dash import dcc

menu = dbc.Nav(
    [
        dbc.NavItem(
            dbc.NavLink(
                "DASHBOARD",
                href="/",
                active=True,
                id="home",
            )
        ),
    ],
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

ml_buttons = [
    dbc.Button(
        "RUN_MODEL",
        id="dev_run_model",
        style={
            "border-top-left-radius": "2svh",
            "border-bottom-left-radius": "2svh",
            "border-top-right-radius": "0px",
            "border-bottom-right-radius": "0px",
            "margin-left": "auto",
        },
        color="primary",
        outline=False,
        className="model-btn",
    ),
    dbc.Button(
        "CANCEL_MODEL",
        id="dev_cancel_model",
        style={
            "border-top-left-radius": "0px",
            "border-bottom-left-radius": "0px",
            "border-top-right-radius": "2svh",
            "border-bottom-right-radius": "2svh",
            "margin-right": "auto",
        },
        color="primary",
        disabled=True,
        outline=False,
        className="model-btn",
    ),
]

navbar_components = dbc.Navbar(
    [
        dbc.Collapse([menu] + ml_buttons, id="navbar-collapse", navbar=True),
        dbc.DropdownMenu(
            lang_menu,
            id="navbar-dropdown",
            nav=True,
            align_end=True,
            label="文/A",
        ),
    ],
)
