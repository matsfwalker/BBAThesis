from enum import StrEnum
from pathlib import Path

from .schema import PATH_ANALYSIS_CLASS, PATH_CONFIG_CLASS, PROJECT_ROOT

# Paths that are hidden from the PATHCONFIG
DATA_DIR: Path = PROJECT_ROOT / "data"
SQL_DIR: Path = PROJECT_ROOT / "sql"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
LOGGING_DIR: Path = PROJECT_ROOT / "logs"

# Paths for the analysis
ANALYSIS_PATHS = PATH_ANALYSIS_CLASS(
    PORTFOLIO_DATA_DIR=DATA_DIR / "portfolios",
    # Result directories in the RESULTS_DIR
    RESULT_DATA_DIR=RESULTS_DIR / "data",
    RESULT_IMAGES_DIR=RESULTS_DIR / "images",
    RESULT_TABLES_DIR=RESULTS_DIR / "tables",
    LOGGING_DIR=PROJECT_ROOT / "logs",
)

# Paths for the main
PATHCONFIG = PATH_CONFIG_CLASS(
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
class FILENAMES_ANALYSIS_CLASS(StrEnum):
    # Portfolio Info
    Portfolio_info = "portfolio_returns_monthly"
    Portfolio_construction_details = "portfolio_construction_details"

    # Model outcomes
    Comp_pred_actual_portfolio = "factor_model_predictions_monthly_obs_universe"
    Factor_loadings_monthly = "factor_loadings_monthly_portfolios"
    Factor_loadings_differentperiods = "factor_loadings_overtime_portfolios"


# Enum for all pathnames
class FILENAMES_CLASS(StrEnum):
    # Fama French Files
    FF5_factors_monthly = "fama_french_monthly_factors"
    FF5_factors_yearly = "fama_french_yearly_factors"

    # FF-portfolios
    FF5_industry_portfolios = "fama_french_industry_portfolios"

    # Stock info files
    Stock_prices = "stock_prices_wrds_monthly_obs_universe"

    # Firm info
    Firm_info = "firm_info_wrds"

    # Sic Description
    Sic_description = "sic_description"

    # Portfolio Info
    Portfolio_info = "portfolio_returns_monthly"

    # Inflation information
    Inflation_info_monthly = "monthly_inflation_info"

    # Model outcomes
    Comp_pred_actual_portfolio = "factor_model_predictions_monthly_obs_universe"
    Factor_loadings_monthly = "factor_loadings_monthly_portfolios"
    Factor_loadings_differentperiods = "factor_loadings_overtime_portfolios"
