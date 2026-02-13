import datetime as dt
import warnings
from typing import Any, Dict, Tuple, Union

import pandas as pd
import pandas_datareader.data as web
import wrds  # Wharton Research Data Services

from configs import CONFIG, CONFIGURATION, FILENAMES, DATAFRAME_CONTAINER


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

    if config.LOG_INFO:
        config.logger.info(
            f"Successfully downloaded Fama-French factors from {lib_name} from {data_course}"
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

    result: pd.DataFrame = con.raw_sql(
        sql_query_daily_price, params=price_query_params, date_cols=["date"]
    )

    if config.LOG_INFO:
        config.logger.info(
            f"Successfully downloaded daily prices for the observable universe of stocks from WRDS from {start_date} to {end_date}"
        )

    return result


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

    result: pd.DataFrame = con.raw_sql(sql_query_firm_info)

    if config.LOG_INFO:
        config.logger.info(
            "Successfully downloaded firm information for the observable universe of stocks from WRDS"
        )

    return result


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

    result: pd.DataFrame = con.raw_sql(sql_query_sic_codes)

    if config.LOG_INFO:
        config.logger.info(
            "Successfully downloaded firm information for the observable universe of stocks from WRDS"
        )
    
    return result


def download_monthly_inflation(
    config: CONFIGURATION
)->pd.Series:
    """
    Function to download monthly inflation data for the entire period to discount values using the time value of money.
    This is the MoM inflation, not inflation compared to previous year.

    Parameters
    ----------
    config : CONFIGURATION
        Configuration of the project

    Returns
    -------
    pd.Series
        Series containing MoM inflation
    
    """
    # Unpack the config
    start_date: dt.datetime = config.START_DATE_FACTORS_DOWNLOAD - dt.timedelta(days=31)
    end_date: dt.datetime = config.END_DATE_FACTORS_DOWNLOAD

    inflation_lib: str = config.INFLATION_LIB
    inflation_source: str = config.INFLATION_SOURCE

    cpi: Any = web.DataReader(name = inflation_source,
                         data_source=inflation_lib, 
                         start= start_date,
                         end=end_date)
    
    if not isinstance(cpi, pd.DataFrame):
        raise ValueError(f"cpi object downloaded from {inflation_lib} is of type {type(cpi)} and not pd.DataFrame")
    
    monthly_inflation: pd.Series = cpi["CPIAUCSL"].pct_change(1, fill_method=None)

    monthly_inflation.name = "MoM inflation"
    monthly_inflation.index.name = "date"

    if config.LOG_INFO:
        config.logger.info(
            f"Successfully downloaded inflation info from {inflation_lib} from {inflation_source} from {start_date} to {end_date}"
        )

    return monthly_inflation


def connect_wrds(config: CONFIGURATION) -> wrds.Connection:
    """
    Function to connect to the WRDS database using the credentials specified in the configuration.
    
    Parameters
    ----------
    config : CONFIGURATION
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


# Main functions
def download_data(config: CONFIGURATION) -> DATAFRAME_CONTAINER:
    """Donwloads the entire data necessary for the project.
    Parameters
    ----------
    config : CONFIGURATION
        Configuration of the project

    Returns
    -------
    DATAFRAME_CONTAINER
        Container containing 5 pd.DataFrames with the entire data"""

    # Connect to WRDS database
    db: wrds.Connection = connect_wrds(config)

    # Download fama french data
    ff5_monthly, ff5_yearly = download_fama_french_factors(CONFIG)

    # Download the prices data
    prices_obs_universe = download_prices_daily_wrds(db, CONFIG)

    # Download the firm info
    firm_info: pd.DataFrame = download_firm_info_wrds(db, CONFIG)

    # Download sic codes
    sic_codes: pd.DataFrame = download_sic_description_wrds(db, CONFIG)

    # Download inflation information
    monthly_inflation: pd.Series = download_monthly_inflation(CONFIG)

    db.close()

    return DATAFRAME_CONTAINER(
        monthly_fama_french=ff5_monthly,
        yearly_fama_french=ff5_yearly,
        stock_market_info=prices_obs_universe,
        firm_info=firm_info,
        sic_info=sic_codes,
        monthly_inflation=monthly_inflation
    )


def save_data(data: DATAFRAME_CONTAINER, config: CONFIGURATION) -> None:
    """
    Function to save the data in the right location.
    Parameters
    ---------
    data : DATAFRAME_CONTAINER
        Container of 5 pd.DataFrames of the different info

    config: CONFIGURATION
        Configuration of the project

    Returns
    -------
    None"""

    # Unpack the container
    ff5_monthly: pd.DataFrame = data.monthly_fama_french
    ff5_yearly: pd.DataFrame = data.yearly_fama_french
    prices_obs_universe: pd.DataFrame = data.stock_market_info
    firm_info: pd.DataFrame = data.firm_info
    sic_codes: pd.DataFrame = data.sic_info
    monthly_inflation: pd.DataFrame = data.monthly_inflation

    ff5_monthly.to_csv(CONFIG.paths.raw_out(FILENAMES.FF5_factors_monthly))
    ff5_yearly.to_csv(CONFIG.paths.raw_out(FILENAMES.FF5_factors_yearly))

    prices_obs_universe.to_csv(
        CONFIG.paths.raw_out(FILENAMES.Stock_prices), index=False
    )

    firm_info.to_csv(CONFIG.paths.raw_out(FILENAMES.Firm_info), index=False)

    sic_codes.to_csv(CONFIG.paths.raw_out(FILENAMES.Sic_description), index=False)

    monthly_inflation.to_csv(CONFIG.paths.raw_out(FILENAMES.Inflation_info_monthly), index=True)

    if config.LOG_INFO:
        config.logger.info(
            "Successfully saved all raw data files"
        )

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
    data: DATAFRAME_CONTAINER = download_data(config)
    save_data(data, config)

    return


if __name__ == "__main__":
    download_save_raw_data(CONFIG)
