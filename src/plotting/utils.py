import pandas as pd
from configs import ANALYSIS_PATHS, FILENAMES_CLASS
from typing import Tuple, Any, List
from rich.tree import Tree
import rich


def as_float(x: Any) -> float:
    return float(pd.to_numeric(x, errors="raise"))


def import_plotting_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function to return the necessary data for plotting

    Parameters
    ----------
    None

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The dataframes necessary to plot the findings
        - monthly_factor_loadings
        - monthly_predicted_returns
        - factor_loadings_overtime"""
    # Import files and set index
    monthly_factor_loadings: pd.DataFrame = pd.read_csv(
        ANALYSIS_PATHS.results_read(FILENAMES_CLASS.Factor_loadings_monthly),
        index_col=[0, 1],
    )

    monthly_predicted_returns: pd.DataFrame = pd.read_csv(
        ANALYSIS_PATHS.results_read(FILENAMES_CLASS.Comp_pred_actual_portfolio),
        header=[0, 1],
        index_col=0,
        parse_dates=True,
    )

    factor_loadings_overtime: pd.DataFrame = pd.read_csv(
        ANALYSIS_PATHS.results_read(FILENAMES_CLASS.Factor_loadings_differentperiods),
        header=[0, 1],
        index_col=[0, 1],
    )

    return monthly_factor_loadings, monthly_predicted_returns, factor_loadings_overtime


def tree_plot_portfolios(portfolio_list: List[str]) -> None:
    """
    Function to show the available portfolios in a tree like fashion.

    Parameters
    ----------
    portfolio_list: List[str]
        List of the different portfolios

    Returns
    -------
    None
        Prints to the console"""

    large_cap_id: str = " - Large Cap"
    small_cap_id: str = " - Small Cap"

    main_tree = Tree("Portfolios")

    for portfolio in portfolio_list:

        # Skip the large and small cap portfolios
        if large_cap_id in portfolio or small_cap_id in portfolio:
            continue

        else:
            main_portfolio: Tree = main_tree.add(portfolio)
            small_cap_subportfolio: str = portfolio + small_cap_id
            large_cap_subportfolio: str = portfolio + large_cap_id

            if small_cap_subportfolio in portfolio_list:
                main_portfolio.add(small_cap_subportfolio)
            if large_cap_subportfolio in portfolio_list:
                main_portfolio.add(large_cap_subportfolio)

    rich.print(main_tree)
