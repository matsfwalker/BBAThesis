from enum import StrEnum
from pathlib import Path
from typing import Literal

from .schema import PATH_ANALYSIS, PATH_CONFIG, PROJECT_ROOT

# Paths that are hidden from the PATHCONFIG
DATA_DIR: Path = PROJECT_ROOT / "data"
SQL_DIR: Path = PROJECT_ROOT / "sql"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
LOGGING_DIR: Path = PROJECT_ROOT / "logs"

# Paths for the analysis
ANALYSIS_PATHS = PATH_ANALYSIS(
    PORTFOLIO_DATA_DIR=DATA_DIR / "portfolios",
    # Result directories in the RESULTS_DIR
    RESULT_DATA_DIR=RESULTS_DIR / "data",
    RESULT_IMAGES_DIR=RESULTS_DIR / "images",
    LOGGING_DIR=PROJECT_ROOT / "logs",
)

# Paths for the main
PATHCONFIG = PATH_CONFIG(
    # SQL directory
    SQL_DIR=PROJECT_ROOT / "sql",
    # Data directories in the DATA_DIR
    RAW_DATA_DIR=DATA_DIR / "raw",
    PROCESSED_DATA_DIR=DATA_DIR / "processed",
    PORTFOLIO_DATA_DIR=DATA_DIR / "portfolios",
    # Result directories in the RESULTS_DIR
    RESULT_DATA_DIR=RESULTS_DIR / "data",
    RESULT_IMAGES_DIR=RESULTS_DIR / "images",
    LOGGING_DIR=PROJECT_ROOT / "logs",
)


# Enum for the analysis pathnames
class FILENAMES_ANALYSIS(StrEnum):
    # Portfolio Info
    Portfolio_returns = "portfolio_returns_monthly"
    Portfolio_construction_details = "portfolio_construction_details"

    # Model outcomes
    Comp_pred_actual_portfolio = "factor_model_predictions_monthly_obs_universe"
    Factor_loadings_monthly = "factor_loadings_monthly_portfolios"
    Factor_loadings_differentperiods = "factor_loadings_overtime_portfolios"


# Enum for all pathnames
class FILENAMES(StrEnum):
    # Fama French Files
    FF5_factors_monthly = "fama_french_monthly_factors"
    FF5_factors_yearly = "fama_french_yearly_factors"

    # FF-Portfolio
    ff_portfolios: Literal["Siccodes5", "Siccodes17", "Siccodes30", "Siccodes38", "Siccodes48", "Siccodes49"] = "Siccodes30"

    # Stock info files
    Stock_prices = "stock_prices_wrds_monthly_obs_universe"

    # Firm info
    Firm_info = "firm_info_wrds"

    # Sic Description
    Sic_description = "sic_description"

    # Portfolio Info
    Portfolio_returns = "portfolio_returns_monthly"
    Portfolio_construction_details = "portfolio_construction_details"

    # Inflation information
    Inflation_info_monthly = "monthly_inflation_info"

    # Model outcomes
    Comp_pred_actual_portfolio = "factor_model_predictions_monthly_obs_universe"
    Factor_loadings_monthly = "factor_loadings_monthly_portfolios"
    Factor_loadings_differentperiods = "factor_loadings_overtime_portfolios"
