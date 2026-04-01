# Constants are those values that remain fixed and are essential to ensure the program runs.
# They may not change based on different configurations or builds.
import pandas as pd
from typing import List

# 1. Factor model constants
# Source: Kenneth R. French Data Library (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
# Name of the library in pandas datareader
FACTORS_LIB: str = "F-F_Research_Data_5_Factors_2x3"

# Name of the data source within the library
FACTORS_DATA_SOURCE: str = "famafrench"

# Name of the library for inflation info in pandas datareader
INFLATION_LIB: str = "fred"

# Name of the data source for monthly inflation info in pandas datareader
INFLATION_SOURCE: str = "CPIAUCSL"

# Information for covid 19
START_PANDEMIC: pd.Timestamp = pd.Timestamp("2019-12-01")
END_PANDEMIC: pd.Timestamp = pd.Timestamp("2021-06-01")

# Information about the start of the period
ANALYSIS_START_DATE: pd.Timestamp = pd.Timestamp("2009-06-01")
ANALYSIS_END_DATE: pd.Timestamp = pd.Timestamp("2026-02-01")

# Break dates for the time-series comparison
BREAK_DATE_PERIODS: List[pd.Timestamp] = [
    pd.Timestamp("2015-01-01"),
    pd.Timestamp("2020-02-01"),
    pd.Timestamp("2023-12-01"),
]
