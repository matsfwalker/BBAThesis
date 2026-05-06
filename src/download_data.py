import datetime as dt
import warnings
from typing import Any, Dict, Tuple, Union, cast, Optional, List
import re
import pandas as pd
import wrds  # Wharton Research Data Services

from configs import (
    PROJ_CONFIG,
    CONFIGURATION_CLASS,
    FILENAMES_CLASS,
    DATAFRAME_CONTAINER,
)
from .utils import chunkify_dates

####################
# Helper Functions #
####################


def connect_wrds(config: CONFIGURATION_CLASS) -> wrds.Connection:
    """
    Function to connect to the WRDS database using the credentials specified in the configuration.

    Parameters
    ----------
    config : CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    wrds.Connection
        Connection object to the WRDS database
    """
    # Connect to wrds Databank
    wrds_credentials: Dict[str, str] = config.get_wrds_data()

    db = wrds.Connection(wrds_username=wrds_credentials["username"], autoconnect=True)
    pd.set_option("future.no_silent_downcasting", True)

    if config.LOG_INFO:
        config.logger.info("Successfully connected to WRDS database")

    return db


###########
# Factors #
###########


# Sub-download functions
def download_fama_french_factors(
    config: CONFIGURATION_CLASS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads Fama-French factors from the specified data source.

    Parameters
    ----------
    config : CONFIGURATION_CLASS:
        Configuration object containing parameters for the download.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]:
        Two data frames containing first the monthly and then the yearly data
    """
    # Retrieve the necessary info from config
    lib_name: str = config.FACTORS_LIB
    data_course: str = config.FACTORS_DATA_SOURCE
    start_date: dt.datetime = config.START_DATE_ANALYSIS
    end_date: dt.datetime = config.END_DATE_ANALYSIS

    import pandas_datareader.data as web

    # Filter out the internal deprecation warning of web.DataReader
    warnings.filterwarnings(
        "ignore",
        message="The argument 'date_parser' is deprecated",
        category=FutureWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message="The argument 'date_parser' is deprecated",
        category=DeprecationWarning,
    )

    factors: Dict[Union[int, str], Any] = web.DataReader(
        name=lib_name, data_source=data_course, start=start_date, end=end_date
    )

    monthly_factors: pd.DataFrame = factors[0]
    yearly_factors: pd.DataFrame = factors[1]

    monthly_factors.index.name = "date"
    yearly_factors.index.name = "date"

    if config.LOG_INFO:
        config.logger.info(
            f"Successfully downloaded Fama-French factors from {lib_name} from {data_course}"
        )
        config.logger.debug(f"Monthly factors sample:\n\n{monthly_factors.sample(5)}\n")

    return monthly_factors, yearly_factors


##########################
# FF Industry Portfolios #
##########################


def import_ff_portfolios(config: CONFIGURATION_CLASS) -> pd.DataFrame:
    """
    Function to import the Fama-French portfolios from the downloaded files.

    Parameters
    ----------
    config : CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    pd.DataFrame
        DataFrame containing the Fama-French portfolios
    """

    rows = []

    current_industry_id = None
    current_short = None
    current_name = None

    header_blueprint: str = r"^\s*(\d+)\s+(\w+)\s+(.*)$"
    sic_blueprint: str = r"^\s*(\d{4})-(\d{4})\s+(.*)$"

    with open(
        config.paths.read_raw_txt(cast(str, config.FAMA_FRENCH_INDUSTRY_PORTFOLIOS)),
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.rstrip()

            if not line.strip():
                continue

            # Match industry header line (e.g. "1 Food   Food")
            header_match = re.match(header_blueprint, line)
            if header_match:
                current_industry_id = int(header_match.group(1))
                current_short = header_match.group(2)
                current_name = header_match.group(3).strip()
                continue

            # Match SIC range line (e.g. "0100-0199 Agricultural production - crops")
            sic_match = re.match(sic_blueprint, line)
            if sic_match:
                sic_start = int(sic_match.group(1))
                sic_end = int(sic_match.group(2))
                sic_desc = sic_match.group(3).strip()

                rows.append(
                    {
                        "industry_id": current_industry_id,
                        "industry_short": current_short,
                        "industry_name": current_name,
                        "sic_start": sic_start,
                        "sic_end": sic_end,
                        "sic_description": sic_desc,
                    }
                )

    result: pd.DataFrame = pd.DataFrame(rows)

    if config.LOG_INFO:
        config.logger.info(
            f"Successfully imported Fama-French industry portfolios from {config.FAMA_FRENCH_INDUSTRY_PORTFOLIOS}.txt"
        )
        config.logger.debug(
            f"Fama-French industry portfolios sample:\n\n{result.sample(5)}\n"
        )

    return result


#############
# Inflation #
#############


def download_monthly_inflation(config: CONFIGURATION_CLASS) -> pd.Series:
    """
    Function to download monthly inflation data for the entire period to discount values using the time value of money.
    This is the MoM inflation, not inflation compared to previous year.

    Parameters
    ----------
    config : CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    pd.Series
        Series containing MoM inflation

    """
    # Unpack the config
    start_date: dt.datetime = config.START_DATE_ANALYSIS - dt.timedelta(days=31)
    end_date: dt.datetime = config.END_DATE_ANALYSIS

    inflation_lib: str = config.INFLATION_LIB
    inflation_source: str = config.INFLATION_SOURCE

    import pandas_datareader.data as web

    cpi: Any = web.DataReader(
        name=inflation_source, data_source=inflation_lib, start=start_date, end=end_date
    )

    if not isinstance(cpi, pd.DataFrame):
        raise ValueError(
            f"cpi object downloaded from {inflation_lib} is of type {type(cpi)} and not pd.DataFrame"
        )

    monthly_inflation: pd.Series = cpi["CPIAUCSL"].pct_change(1, fill_method=None)

    monthly_inflation.name = "MoM inflation"
    monthly_inflation.index.name = "date"

    if config.LOG_INFO:
        config.logger.info(
            f"Successfully downloaded inflation info from {inflation_lib} from {inflation_source} from {start_date} to {end_date}"
        )
        config.logger.debug(
            f"Monthly inflation sample:\n\n{monthly_inflation.sample(5)}\n"
        )

    return monthly_inflation


##################
# WRDS Downloads #
##################


def download_monthly_market_info_wrds(
    con: wrds.Connection, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to query the monthly info, including prices, returns, etc. for the observable universe of stocks from WRDS.

    Parameters
    ----------
    con : wrds.Connection
        Connection object to the WRDS database.
    config : CONFIGURATION_CLASS
        Configuration of the project
    Returns
    -------
    pd.DataFrame
        DataFrame containing the daily prices for the observable universe of stocks.
    """

    # Unpack the config
    start_date: dt.datetime = config.START_DATE_ANALYSIS - dt.timedelta(days=31)
    end_date: dt.datetime = config.END_DATE_ANALYSIS  # - dt.timedelta(days=31)

    # Get the SQL query
    sql_query_monthly_price: str = config.paths.sql_query(
        "monthly_market_prices"
    ).read_text()

    frames: List[pd.DataFrame] = []

    for price_query_params in chunkify_dates(start_date, end_date):
        result_subquery: pd.DataFrame = con.raw_sql(
            sql_query_monthly_price, params=price_query_params, date_cols=["date"]
        )
        frames.append(result_subquery)

        if config.LOG_INFO:
            config.logger.debug(
                f"Downloaded monthly prices for the observable universe of stocks from WRDS from {price_query_params[0]} to {price_query_params[1]}"
            )

    result = pd.concat(frames, ignore_index=True)

    if config.LOG_INFO:
        config.logger.info(
            f"Successfully downloaded monthly prices for the observable universe of stocks ({len(result['gvkey'].unique())} firms) from WRDS from {start_date} to {end_date}"
        )
        config.logger.debug(f"Monthly prices sample:\n\n{result.sample(5)}\n")

    return result


def download_firm_info_wrds(
    con: wrds.Connection, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to query information for the firms in the observable universe of stocks from WRDS.
    The function reads the SQL query from the external file SQL_QUERY_FIRM_INFO.

    Parameters
    ----------
    con : wrds.Connection
        Connection object to the WRDS database.
    config : CONFIGURATION_CLASS
        configuration of the project
    Returns
    -------
    pd.DataFrame
        DataFrame containing the information for the firms in the observable universe of stocks.
    """
    sql_query_firm_info: str = config.paths.sql_query("firm_info").read_text()

    result: pd.DataFrame = con.raw_sql(sql_query_firm_info)

    if config.LOG_INFO:
        config.logger.info(
            f"Successfully downloaded firm information for the observable universe ({len(result["gvkey"].unique())} firms) of stocks from WRDS"
        )
        config.logger.debug(f"Firm info sample:\n\n{result.sample(5)}\n")

    return result


def download_sic_description_wrds(
    con: wrds.Connection, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to retrieve information about different sic codes from WRDS.
    SIC codes are used to classify firms by industry.
    These can be grouped by their first 1, 2, 3 or 4 digits depending on precision.
    The function reads the SQL query from the external file SQL_QUERY_SIC_CODES.

    Parameters
    ----------
    con : wrds.Connection
        Connection object to the WRDS database.
    config : CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    pd.DataFrame
        DataFrame containing the sic code descriptions.
    """
    sql_query_sic_codes: str = config.paths.sql_query("sic_codes").read_text()

    result: pd.DataFrame = con.raw_sql(sql_query_sic_codes)

    if config.LOG_INFO:
        config.logger.info(
            f"Successfully downloaded firm information for the observable universe of stocks ({len(result["siccode"].unique())} firms) from WRDS"
        )
        config.logger.debug(f"SIC codes sample:\n\n{result.sample(5)}\n")

    return result


##################
# Main functions #
##################


def download_data(config: CONFIGURATION_CLASS) -> DATAFRAME_CONTAINER:
    """Donwloads the entire data necessary for the project.
    Parameters
    ----------
    config : CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    DATAFRAME_CONTAINER
        Container containing 5 pd.DataFrames with the entire data"""

    if config.LOG_INFO:
        config.logger.info(
            "Starting to download all necessary data for the project...\n" + "-" * 80
        )

    # Connect to WRDS database
    db: wrds.Connection = connect_wrds(config)

    # Download the prices data
    prices_obs_universe = download_monthly_market_info_wrds(db, config)

    # Download the firm info
    firm_info: pd.DataFrame = download_firm_info_wrds(db, config)

    # Download sic codes
    sic_codes: pd.DataFrame = download_sic_description_wrds(db, config)

    db.close()

    # Download fama french data
    ff5_monthly, ff5_yearly = download_fama_french_factors(config)

    # Import the fama-french portfolios from the downloaded txt file
    ff_industry_portfolios: pd.DataFrame = import_ff_portfolios(config)

    # Download inflation information
    monthly_inflation: pd.Series = download_monthly_inflation(config)

    if config.LOG_INFO:
        config.logger.info("Successfully downloaded all necessary data for the project")

    return DATAFRAME_CONTAINER(
        monthly_fama_french=ff5_monthly,
        yearly_fama_french=ff5_yearly,
        monthly_stock_info=prices_obs_universe,
        firm_info=firm_info,
        sic_info=sic_codes,
        monthly_inflation=monthly_inflation,
        ff_industry_portfolios=ff_industry_portfolios,
    )


def save_data(data: DATAFRAME_CONTAINER, config: CONFIGURATION_CLASS) -> None:
    """
    Function to save the data in the right location.
    Parameters
    ---------
    data : DATAFRAME_CONTAINER
        Container of 5 pd.DataFrames of the different info

    config: CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    None"""

    if config.LOG_INFO:
        config.logger.info("Starting to save all raw files....\n" + "-" * 80)

    # Unpack the container
    ff5_monthly: Optional[pd.DataFrame] = data.monthly_fama_french
    ff5_yearly: Optional[pd.DataFrame] = data.yearly_fama_french
    if ff5_monthly is None or ff5_yearly is None:
        raise ValueError("Fama-French factors data frames cannot be None")

    config.paths.raw_save(df=ff5_monthly, stem=FILENAMES_CLASS.FF5_factors_monthly)

    config.paths.raw_save(df=ff5_yearly, stem=FILENAMES_CLASS.FF5_factors_yearly)

    config.paths.raw_save(
        df=data.monthly_stock_info, stem=FILENAMES_CLASS.Stock_prices, index=False
    )

    config.paths.raw_save(
        df=data.firm_info, stem=FILENAMES_CLASS.Firm_info, index=False
    )

    config.paths.raw_save(
        df=data.sic_info, stem=FILENAMES_CLASS.Sic_description, index=False
    )

    config.paths.raw_save(
        df=data.monthly_inflation,
        stem=FILENAMES_CLASS.Inflation_info_monthly,
    )

    config.paths.raw_save(
        df=data.ff_industry_portfolios,
        stem=FILENAMES_CLASS.FF5_industry_portfolios,
        index=False,
    )

    if config.LOG_INFO:
        config.logger.info("Successfully saved all raw data files")

    return


def download_save_raw_data(config: CONFIGURATION_CLASS) -> None:
    """
    Function to download and then save all necessary data.
    This function orchestrates the other functions.

    Parameters
    ----------
    config : CONFIGURATION_CLASS
        Configuration of the file

    Returns
    -------
    None"""
    data: DATAFRAME_CONTAINER = download_data(config)
    save_data(data, config)

    return


if __name__ == "__main__":
    download_save_raw_data(PROJ_CONFIG)
