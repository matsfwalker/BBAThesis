# Constants are those values that remain fixed and are essential to ensure the program runs.
# They may not change based on different configurations or builds.
import pandas as pd

# 1. Factor model constants
# Source: Kenneth R. French Data Library (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
# Name of the library in pandas datareader
FACTORS_LIB: str = "F-F_Research_Data_5_Factors_2x3"

# Name of the data source within the library
FACTORS_DATA_SOURCE: str = "famafrench"

# Name of the library for inflation info in pandas datareader
INFLATION_LIB: str = "fred"

# Name of the data source for monthly inflation info in pandas datareader
INFLATION_SOURCE: str = "CPIAUCSL"

# Information for covid 19
START_PANDEMIC: pd.Timestamp = pd.Timestamp("2019-12-01")
END_PANDEMIC: pd.Timestamp = pd.Timestamp("2021-06-01")
