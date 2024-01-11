import pandas as pd

import dash_bootstrap_components as dbc
from dash import html, dcc
import dash_ag_grid as dag

class MenuCollapse:
    """
    A class representing a collapsible menu item.

    Parameters
    ----------
    lang : dict
        A dictionary containing language translations.
    label : str
        The label used to retrieve the translated name from the "lang"
        dictionary.
    component : dbc._components.Row
        The component to be displayed inside the collapsible menu item.
    id_prefix : str
        A prefix used to generate unique IDs for the collapse and button
        components.

    Attributes
    ----------
    label_name : str
        The translated label name.
    component : dbc._components.Row
        The component to be displayed inside the collapsible menu item.
    id_prefix : str
        A prefix used to generate unique IDs for the collapse and button
        components.

    Methods
    -------
    menu_collapse()
        Create a collapsible menu item.

        Returns
        -------
        tuple
            A tuple containing the collapse and button components.
    """

    def __init__(
        self,
        lang: dict,
        label: str,
        component,
        id_prefix: str,
        is_open: bool = False,
        **kwargs
    ):
        """
        Initialize a MenuCollapse instance.

        Parameters
        ----------
        lang : dict
            A dictionary containing language translations.
        label : str
            The label used to retrieve the translated name from the
            "lang" dictionary.
        component : dbc._components.Row
            The component to be displayed inside the collapsible menu
            item.
        id_prefix : str
            A prefix used to generate unique IDs for the collapse and
            button components.
        """

        self.label_name = lang[label]
        self.component = component
        self.id_prefix = id_prefix
        self.is_open = is_open

        self.button = dbc.Button(
            [
                self.label_name,
                html.I(
                    className="fa fa-chevron-down ml-2",
                    id=f"{self.id_prefix}_icon",
                    style={"transformY": "2px"},
                ),
            ],
            id=f"{id_prefix}_button",
            className="d-grid gap-2 col-6 mx-auto w-100",
            outline=True,
            color="secondary",
            **kwargs,
        )

    @property
    def simple_collapse(self):
        """
        Create a collapsible menu item.

        Returns
        -------
        tuple
            A tuple containing the collapse and button components.
        """
        collapse = dbc.Collapse(
            dbc.Card(
                dbc.CardBody(
                    self.component,
                )
            ),
            id=f"{self.id_prefix}_collapse",
            is_open=self.is_open,
        )

        return dbc.Col([self.button, collapse])

    def collapse_with_inside_collapse(self, inside_component):
        """
        Create a collapsible menu item with another collpsible menu
        inside.

        Returns
        -------
        dbc.Col
            A column component containing the collapse and button
            components.
        """
        collapse = dbc.Collapse(
            dbc.Card(
                [
                    dbc.CardBody(
                        self.component,
                    ),
                    dbc.Card(
                        dbc.CardBody(
                            inside_component,
                        ),
                    ),
                ]
            ),
            id=f"{self.id_prefix}_collapse",
            is_open=self.is_open,
        )

        return dbc.Col([self.button, collapse])

