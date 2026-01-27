# File for the configurations of the application
# This are constants that may change depending on the build and are not essential for the core functionality.

import datetime as dt

from .constants import (END_PANDEMIC, FACTORS_DATA_SOURCE, FACTORS_LIB,
                        START_PANDEMIC)
from .paths import PATHCONFIG
from .schema import CONFIGURATION, PLOTTING_CONFIGURATIONS

PLOTTING_CONFIG = PLOTTING_CONFIGURATIONS(
    TIMESPANS_TO_PLOT=[
        {
            "name": "Pandemic",
            "start": START_PANDEMIC,
            "end": END_PANDEMIC,
            "color": "grey",
        }
    ]
)


CONFIG = CONFIGURATION(
    # Paths
    paths=PATHCONFIG,
    # Constants
    FACTORS_LIB=FACTORS_LIB,
    FACTORS_DATA_SOURCE=FACTORS_DATA_SOURCE,
    # Data downloading configs
    START_DATE_FACTORS_DOWNLOAD=dt.datetime(2008, 1, 1),
    END_DATE_FACTORS_DOWNLOAD=dt.datetime.now(),
    # Data-cleaning configs
    THRESHOLD_MISSING_SHARESOUTSTANDING=0.5,  # Relative threshold of missing sharesoutstanding to drop a ticker
    # Portfolio creation configs
    CUTOFF_FIRMS_PER_PORTFOLIO=10,  # Number of firms needed per portfolio
    MIN_MARKETCAP_FIRM=100_000.0,  # Minimum latest market cap needed for a firm to be considered
    SIC_LEVEL=2,  # SIC code level to use for industry portfolios. The larger, the more granular.
    PORTFOLIO_AGGREGATION_METHOD="MarketCap",  # Method to aggregate firms into portfolios
    # Model configurations
    BREAK_DATE_PERIODS=[
        dt.datetime(2015, 1, 1),
        dt.datetime(2019, 1, 1),
        dt.datetime(2022, 1, 1),
    ],
    INCLUDE_END_DATE_PERIOD=True,
    INCLUDE_START_DATE_PERIOD=True,
    PERIOD_WINDOW_LENGTH_MONTHS=None,
    INCLUDE_WHOLE_PERIOD_MODEL=True,  # Whether to compute the model for the entire period in addition to subperiods
    MARKETCAP_PORTFOLIO_PERCENTILE=0.2,  # Percentile threshold to define market cap portfolios (e.g., 0.4 means bottom 40% vs top 40%).
    MARKETCAP_PORTFOLIO_NUMBER_FIRMS=None,  # Number of firms to include in top and bottom market cap portfolios.
    # Statistical configurations
    T_TEST_FACTORS="all",  # Factors to perform t-tests on
    T_TEST_SIGNIFICANCE_LEVEL=2.0,  # Significance level for t-tests
)
