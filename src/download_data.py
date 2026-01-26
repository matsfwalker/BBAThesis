import pandas as pd
import pandas_datareader.data as web
import wrds  # Wharton Research Data Services
import datetime as dt
from typing import Dict, Tuple, List, Any, Union
import warnings
from configs import CONFIG, CONFIGURATION, FILENAMES

type EntireData = Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]


# Sub-download functions
def download_fama_french_factors(
    config: CONFIGURATION,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Downloads Fama-French factors from the specified data source.

    Parameters
    ----------
    config : CONFIGURATION:
        Configuration object containing parameters for the download.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]:
        Two data frames containing first the monthly and then the yearly data
    """
    # Retrieve the necessary info from config
    lib_name: str = config.FACTORS_LIB
    data_course: str = config.FACTORS_DATA_SOURCE
    start_date: dt.datetime = config.START_DATE_FACTORS_DOWNLOAD
    end_date: dt.datetime = config.END_DATE_FACTORS_DOWNLOAD

    # Filter out the internal deprecation warning of web.DataReader
    warnings.filterwarnings(
        "ignore",
        message="The argument 'date_parser' is deprecated",
        category=FutureWarning,
    )

    factors: Dict[Union[int, str], Any] = web.DataReader(
        name=lib_name, data_source=data_course, start=start_date, end=end_date
    )
    return factors[0], factors[1]


def download_prices_daily_wrds(
    con: wrds.Connection, config: CONFIGURATION
) -> pd.DataFrame:
    """
    Function to query the daily prices for the observable universe of stocks from WRDS.
    The function reads the SQL query from the external file SQL_QUERY_MARKET_PRICE.

    Parameters
    ----------
    con : wrds.Connection
        Connection object to the WRDS database.
    config : CONFIGURATION
        Configuration of the project
    Returns
    -------
    pd.DataFrame
        DataFrame containing the daily prices for the observable universe of stocks.
    """

    # Unpack the config
    start_date: dt.datetime = config.START_DATE_FACTORS_DOWNLOAD - dt.timedelta(days=31)
    end_date: dt.datetime = config.END_DATE_FACTORS_DOWNLOAD - dt.timedelta(days=31)
    price_query_params: Tuple[str, str] = (
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

    # Get the SQL query
    sql_query_daily_price: str = config.paths.sql_query(
        "daily_market_prices"
    ).read_text()

    return con.raw_sql(
        sql_query_daily_price, params=price_query_params, date_cols=["date"]
    )


def download_firm_info_wrds(
    con: wrds.Connection, config: CONFIGURATION
) -> pd.DataFrame:
    """
    Function to query information for the firms in the observable universe of stocks from WRDS.
    The function reads the SQL query from the external file SQL_QUERY_FIRM_INFO.

    Parameters
    ----------
    con : wrds.Connection
        Connection object to the WRDS database.
    config : CONFIGURATION
        configuration of the project
    Returns
    -------
    pd.DataFrame
        DataFrame containing the information for the firms in the observable universe of stocks.
    """
    sql_query_firm_info: str = config.paths.sql_query("firm_info").read_text()

    return con.raw_sql(sql_query_firm_info)


def download_sic_description_wrds(
    con: wrds.Connection, config: CONFIGURATION
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
    config : CONFIGURATION
        Configuration of the project

    Returns
    -------
    pd.DataFrame
        DataFrame containing the sic code descriptions.
    """
    sql_query_sic_codes: str = config.paths.sql_query("sic_codes").read_text()

    return con.raw_sql(sql_query_sic_codes)


# Main functions
def download_data(config: CONFIGURATION) -> EntireData:
    """Donwloads the entire data necessary for the project.
    Parameters
    ----------
    config : CONFIGURATION
        Configuration of the project

    Returns
    -------
    EntireData
        5 pd.DataFrames containing the entire data"""

    # Connect to wrds Databank
    db = wrds.Connection(wrds_username=config.WRDS_USERNAME, autoconnect=True)
    pd.set_option("future.no_silent_downcasting", True)

    # Download fama french data
    ff5_monthly, ff5_yearly = download_fama_french_factors(CONFIG)

    # Download the prices data
    prices_obs_universe = download_prices_daily_wrds(db, CONFIG)

    # Download the firm info
    firm_info: pd.DataFrame = download_firm_info_wrds(db, CONFIG)

    # Download sic codes
    sic_codes: pd.DataFrame = download_sic_description_wrds(db, CONFIG)

    db.close()

    return ff5_monthly, ff5_yearly, prices_obs_universe, firm_info, sic_codes


def save_data(data: EntireData) -> None:
    """
    Function to save the data in the right location.
    Parameters
    ---------
    data : EntireData
        Tuple of 5 pd.DataFrames of the different info

    Returns
    -------
    None"""
    ff5_monthly, ff5_yearly, prices_obs_universe, firm_info, sic_codes = data

    ff5_monthly.to_csv(CONFIG.paths.raw_out(FILENAMES.FF5_factors_monthly))
    ff5_yearly.to_csv(CONFIG.paths.raw_out(FILENAMES.FF5_factors_yearly))

    prices_obs_universe.to_csv(
        CONFIG.paths.raw_out(FILENAMES.Stock_prices), index=False
    )

    firm_info.to_csv(CONFIG.paths.raw_out(FILENAMES.Firm_info), index=False)

    sic_codes.to_csv(CONFIG.paths.raw_out(FILENAMES.Sic_description), index=False)

    return


def download_save_raw_data(config: CONFIGURATION) -> None:
    """
    Function to download and then save all necessary data.
    This function orchestrates the other functions.

    Parameters
    ----------
    config : CONFIGURATION
        Configuration of the file

    Returns
    -------
    None"""
    data: EntireData = download_data(config)
    save_data(data)

    return


if __name__ == "__main__":
    download_save_raw_data(CONFIG)
