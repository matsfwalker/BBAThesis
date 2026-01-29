import pandas as pd
from configs import ANALYSIS_PATHS, FILENAMES
from typing import Tuple, Any


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
        ANALYSIS_PATHS.results_read(FILENAMES.Factor_loadings_monthly), index_col=[0, 1]
    )

    monthly_predicted_returns: pd.DataFrame = pd.read_csv(
        ANALYSIS_PATHS.results_read(FILENAMES.Comp_pred_actual_portfolio),
        header=[0, 1],
        index_col=0,
        parse_dates=True,
    )

    factor_loadings_overtime: pd.DataFrame = pd.read_csv(
        ANALYSIS_PATHS.results_read(FILENAMES.Factor_loadings_differentperiods),
        header=[0, 1],
        index_col=[0, 1],
        parse_dates=[0],
    )

    return monthly_factor_loadings, monthly_predicted_returns, factor_loadings_overtime
