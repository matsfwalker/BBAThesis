# File for the configurations of the application
# This are constants that may change depending on the build and are not essential for the core functionality.

from .constants import (
    END_PANDEMIC,
    FACTORS_DATA_SOURCE,
    FACTORS_LIB,
    START_PANDEMIC,
    INFLATION_LIB,
    INFLATION_SOURCE,
    ANALYSIS_START_DATE,
    ANALYSIS_END_DATE,
    BREAK_DATE_PERIODS,
)
from .paths import PATHCONFIG
from .schema import CONFIGURATION_CLASS, PLOTTING_CONFIGURATIONS_CLASS
from .logging_configs import setup_logging

PROJ_CONFIG = CONFIGURATION_CLASS(
    ################
    # Main Configs #
    ################
    THRESHOLD_MISSING_SHARESOUTSTANDING=0.5,  # Relative threshold of missing sharesoutstanding to drop a ticker
    MIN_STOCK_PRICE=1.0,  # Eliminate Penny-stocks
    MAX_MONTHLY_RETURN=1.0,  # Clip at 1000% return per month
    CUTOFF_FIRMS_PER_PORTFOLIO=10,  # Number of firms needed per portfolio
    MIN_MARKETCAP_FIRM=10_000_000.0,  # Minimum market cap needed for a firm to be considered
    MIN_OCCURANCES_PORTFOLIOS_ENTIRE=26,  # Number of occurances of a portfolio throughout the entire period
    MIN_OCCURANCES_PORTFOLIOS_SUB=10,  # Minimum number of occurances of a portfolio in the data to be included in the analysis
    CREATE_MARKETCAP_PORTFOLIOS=False,  # Whether to create market cap portfolios in addition to the industry portfolios
    MARKETCAP_PORTFOLIO_PERCENTILE=None,
    MARKETCAP_PORTFOLIO_EXCHANGE=None,  # "New York Stock Exchange",  # Exchange from which to pull the market cap for the cutoff
    #########
    # Paths #
    #########
    paths=PATHCONFIG,
    ###########
    # Logging #
    ###########
    LOG_INFO=True,
    logger=setup_logging(
        name="Thesis", log_file=PATHCONFIG.LOGGING_DIR / "logging.log", level="DEBUG"
    ),
    ###########
    # Sources #
    ###########
    FACTORS_LIB=FACTORS_LIB,
    FACTORS_DATA_SOURCE=FACTORS_DATA_SOURCE,
    INFLATION_LIB=INFLATION_LIB,
    INFLATION_SOURCE=INFLATION_SOURCE,
    ############################
    # Data downloading configs #
    ############################
    START_DATE_ANALYSIS=ANALYSIS_START_DATE,
    END_DATE_ANALYSIS=ANALYSIS_END_DATE,
    #########################
    # Data-cleaning configs #
    #########################
    EXCHANGES_TO_KEEP=[
        "New York Stock Exchange",
        "Nasdaq Stock Market",
        "NYSE American",
        "Cboe BZX Exchange",
        "NYSEArca",
    ],
    #######################################
    # Industry Portfolio creation configs #
    #######################################
    INDUSTRY_CLASSIFICATION_METHOD="Fama-French_portfolios",
    FAMA_FRENCH_INDUSTRY_PORTFOLIOS="Siccodes48",
    SIC_LEVEL=None,
    ##############################
    # Other portfolio creation configs #
    ##############################
    DISCOUNT_MARKETCAP_FIRM_INFLATION=True,  # Discount the marketcap of firms. If this is used, then the minimum market cap is in real terms, not nominal and applied to each period.
    PORTFOLIO_AGGREGATION_METHOD="MarketCap",  # Method to aggregate firms into portfolios
    ########################
    # Model configurations #
    ########################
    BREAK_DATE_PERIODS=BREAK_DATE_PERIODS,
    INCLUDE_END_DATE_PERIOD=True,
    INCLUDE_START_DATE_PERIOD=True,
    PERIOD_WINDOW_LENGTH_MONTHS=None,
    INCLUDE_WHOLE_PERIOD_MODEL=True,  # Whether to compute the model for the entire period in addition to subperiods
    MARKETCAP_PORTFOLIO_NUMBER_FIRMS=None,  # Number of firms to include in top and bottom market cap portfolios.
    ##############################
    # Statistical configurations #
    ##############################
    T_TEST_FACTORS="all",  # Factors to perform t-tests on
    T_TEST_SIGNIFICANCE_LEVEL=2.0,  # Significance level for t-tests
    P_THRESHOLD=0.05,  # p-value threshold for Jarque-Bera test to reject normality
)


PLOTTING_CONFIG = PLOTTING_CONFIGURATIONS_CLASS(
    TIMESPANS_TO_PLOT=[
        {
            "name": "Pandemic",
            "start": START_PANDEMIC,
            "end": END_PANDEMIC,
            "color": "grey",
        }
    ]
)
