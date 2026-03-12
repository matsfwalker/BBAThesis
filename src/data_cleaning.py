from typing import Tuple, List, Dict, Union, Optional, cast
import pandas as pd
from configs import (
    PROJ_CONFIG,
    CONFIGURATION_CLASS,
    FILENAMES_CLASS,
    DATAFRAME_CONTAINER,
)
import warnings

######################
# Import and Exports #
######################


def download_raw_data(config: CONFIGURATION_CLASS) -> DATAFRAME_CONTAINER:
    """
    Function to download all of the raw data.

    Parameters
    ----------
    config : CONFIGURATION_CLASS
        COnfiguration of the project

    Returns
    ------
    DATAFRAME_CONTAINER
        The 5 different raw dataframes
        - factors_monthly_raw
        - factors_yearly_raw
        - monthly_stock_prices_raw
        - firm_info_raw
        - sic_desc_raw"""

    if config.LOG_INFO:
        config.logger.info("Starting importing raw data....\n" + "-" * 80)

    factors_monthly_raw: pd.DataFrame = pd.read_csv(
        config.paths.raw_read(FILENAMES_CLASS.FF5_factors_monthly),
        parse_dates=["date"],
        index_col="date",
    )
    factors_yearly_raw: pd.DataFrame = pd.read_csv(
        config.paths.raw_read(FILENAMES_CLASS.FF5_factors_yearly),
        parse_dates=["date"],
        index_col="date",
    )

    monthly_stock_prices_raw: pd.DataFrame = pd.read_csv(
        config.paths.raw_read(FILENAMES_CLASS.Stock_prices),
        parse_dates=["date"],
        index_col="date",
    )

    firm_info_raw: pd.DataFrame = pd.read_csv(
        config.paths.raw_read(FILENAMES_CLASS.Firm_info)
    )

    sic_desc_raw: pd.DataFrame = pd.read_csv(
        config.paths.raw_read(FILENAMES_CLASS.Sic_description)
    )

    monthly_inflation_info_raw: pd.DataFrame = pd.read_csv(
        config.paths.raw_read(FILENAMES_CLASS.Inflation_info_monthly),
        parse_dates=["date"],
        index_col="date",
    )

    ff_industry_portfolios_raw: pd.DataFrame = pd.read_csv(
        config.paths.raw_read(FILENAMES_CLASS.FF5_industry_portfolios)
    )

    if config.LOG_INFO:
        config.logger.info("Downloaded all raw data")

    return DATAFRAME_CONTAINER(
        monthly_fama_french=factors_monthly_raw,
        yearly_fama_french=factors_yearly_raw,
        monthly_stock_info=monthly_stock_prices_raw,
        firm_info=firm_info_raw,
        sic_info=sic_desc_raw,
        monthly_inflation=monthly_inflation_info_raw,
        ff_industry_portfolios=ff_industry_portfolios_raw,
    )


def save_processed_data(
    data_processed: DATAFRAME_CONTAINER,
    config: CONFIGURATION_CLASS,
) -> None:
    """
    Function to save all of the processed data in the data/processed dir.

    Parameters
    ----------
    data_processed: DATAFRAME_CONTAINER
        Container containing the all processed dataframes of the project
    config: CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    None
    """
    if config.LOG_INFO:
        config.logger.info("Starting saving processed files....\n" + "-" * 80)

    # Unpack the data if necessary
    monthly_factors_processed: Optional[pd.DataFrame] = (
        data_processed.monthly_fama_french
    )
    factors_yearly_processed: Optional[pd.DataFrame] = data_processed.yearly_fama_french
    if monthly_factors_processed is None or factors_yearly_processed is None:
        raise ValueError("Factors dataframes cannot be None")

    config.paths.processed_save(
        df=monthly_factors_processed, stem=FILENAMES_CLASS.FF5_factors_monthly
    )

    config.paths.processed_save(
        df=factors_yearly_processed, stem=FILENAMES_CLASS.FF5_factors_yearly
    )

    config.paths.processed_save(
        df=data_processed.monthly_stock_info, stem=FILENAMES_CLASS.Stock_prices
    )

    config.paths.processed_save(
        df=data_processed.firm_info, stem=FILENAMES_CLASS.Firm_info, index=False
    )

    config.paths.processed_save(
        df=data_processed.sic_info, stem=FILENAMES_CLASS.Sic_description, index=False
    )

    config.paths.processed_save(
        df=data_processed.monthly_inflation,
        stem=FILENAMES_CLASS.Inflation_info_monthly,
    )

    config.paths.processed_save(
        df=data_processed.ff_industry_portfolios,
        stem=FILENAMES_CLASS.FF5_industry_portfolios,
        index=False,
    )

    if config.LOG_INFO:
        config.logger.info("Finished saving processed files")

    return


#########
# Utils #

#########


def _inflation_discount(inflation: pd.DataFrame, value: float) -> pd.Series:
    """
    Function to discount a given value based on the inflation discount multiples.

    Parameters
    ----------
    inflation : pd.DataFrame
        DataFrame containing the inflation discount multiples with a datetime index.
    value : float
        The value to be discounted.

    Returns
    -------
    pd.Series
        A Series containing the discounted values for each date.
    """
    inflation = inflation.copy()
    return inflation["Inflation multiple"] * value


def _present_value_inflation(inflation: pd.DataFrame, values: pd.Series) -> pd.Series:
    """
    Function to adjust past prices to their present value based on the inflation discount multiples.

    Parameters
    ----------

    inflation : pd.DataFrame
        DataFrame containing the inflation discount multiples with a datetime index.
    values : pd.Series
        A Series containing the values to be adjusted, indexed by date.

    Returns
    -------
    pd.Series
        A Series containing the inflation-adjusted values for each date.
    """
    assert isinstance(
        inflation.index, pd.DatetimeIndex
    ), "The index of the inflation dataframe should be a DatetimeIndex."
    assert isinstance(
        values.index, pd.DatetimeIndex
    ), "The index of the values series should be a DatetimeIndex."

    inflation_multiple = inflation["Inflation multiple"]
    matched_inflation = values.index.to_series().map(inflation_multiple)

    return values / matched_inflation.to_numpy()


# Get the returns from portfolios (No longer Used)
def __get_returns_stocks(stock_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Function to compute the returns of individual stocks.
    Parameters
    ----------
    stock_prices : pd.DataFrame
        DataFrame containing stock prices with a datetime index.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the returns of individual stocks in addition to the existing info.
    """

    warnings.warn(
        "__get_returns_stocks() is deprecated, as the returns are received directly from wrds",
        DeprecationWarning,
        stacklevel=2,
    )

    # Reset the index of the dataframe
    prices_noindex: pd.DataFrame = stock_prices.reset_index(names=["date"])

    # Sort the prices, first by gvkey and then by date
    sorted_prices: pd.DataFrame = prices_noindex.sort_values(
        ["gvkey", "date"]
    ).drop_duplicates(subset=["gvkey", "date"])

    # Calculate their return
    sorted_prices["return"] = sorted_prices.groupby("gvkey")["close"].pct_change(
        fill_method=None
    )

    # Reset the date as index and sort it
    returns: pd.DataFrame = (
        sorted_prices.set_index("date").dropna(subset=["return"]).sort_index()
    )

    return returns


#############
# MarketCap #
#############


def _compute_market_cap(
    df_info: pd.DataFrame, price_column: str, shares_column: str
) -> pd.Series:
    """
    Function to compute the market cap of firms.

    Parameters
    ----------
    df_info : pd.DataFrame
        DataFrame containing firm information including price and shares outstanding.
    price_column : str
        The name of the column containing stock prices.
    shares_column : str
        The name of the column containing shares outstanding.

    Returns
    -------
    pd.Series
        A Series containing the market cap of each firm.
    """
    return df_info[price_column] * df_info[shares_column]


def get_market_cap(
    stock_prices: pd.DataFrame, inflation: pd.DataFrame, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to add the market cap and present value market cap of firms to the dataframe.

    Parameters
    ----------
    stock_prices : pd.DataFrame
        DataFrame containing stock prices with a datetime index.
    inflation : pd.DataFrame
        DataFrame containing the inflation discount multiples with a datetime index.
    config : CONFIGURATION_CLASS
        Configuration of the project.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the original stock prices information along with the market cap and present value market cap (if applicable) of each firm.
    """
    # Asserts
    assert (
        "close" in stock_prices.columns
    ), "Close price column is missing in the stock prices dataframe."
    assert (
        "sharesoutstanding" in stock_prices.columns
    ), "Shares outstanding column is missing in the stock prices dataframe."
    assert isinstance(
        stock_prices.index, pd.DatetimeIndex
    ), "The index of the stock prices dataframe should be a DatetimeIndex."

    stock_prices = stock_prices.copy()
    stock_prices["market_cap"] = _compute_market_cap(
        stock_prices, "close", "sharesoutstanding"
    )
    stock_prices["Lagged_MarketCap"] = stock_prices.groupby("gvkey")[
        "market_cap"
    ].shift(1)

    if config.DISCOUNT_MARKETCAP_FIRM_INFLATION:
        stock_prices["market_cap_present_value"] = _present_value_inflation(
            inflation=inflation, values=stock_prices["market_cap"]
        )

    return stock_prices


# MarketCap cutoff
def _apply_marketcap_cutoff_latestperiod(
    stock_prices: pd.DataFrame, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to apply a market cap cutoff to the latest stock prices.
    Parameters
    ----------
    stock_prices : pd.DataFrame
        DataFrame containing stock prices with a datetime index.
    config : CONFIGURATION_CLASS
        Configuration of the project.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the firms that meet the market cap cutoff at the latest date.
    """
    # Unpack the config:
    marketcap_cutoff: float = config.MIN_MARKETCAP_FIRM

    num_firms_start: int = stock_prices["gvkey"].nunique()

    # Get the latest date and corresponding entries
    latest_date: pd.Timestamp = stock_prices.index.max()
    latest_prices: pd.DataFrame = (
        stock_prices.loc[latest_date].reset_index().drop_duplicates(subset=["gvkey"])
    )

    surviving_gvkeys = latest_prices.loc[
        latest_prices["Lagged_MarketCap"] >= marketcap_cutoff, "gvkey"
    ].unique()

    result = stock_prices[stock_prices["gvkey"].isin(surviving_gvkeys)].copy()

    num_firms_end: int = result["gvkey"].nunique()

    if config.LOG_INFO:
        config.logger.info(
            f"Applied market cap cutoff of {marketcap_cutoff} to the latest period ({latest_date.date()}).\
            \nNumber of firms dropped: {num_firms_start - num_firms_end} ({num_firms_start}->{num_firms_end}"
        )

    return result


def _apply_marketcap_cutoff_allperiods(
    stock_prices: pd.DataFrame, inflation: pd.DataFrame, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to apply a market cap cutoff to the stock prices of all periods, discounting the market cap by inflation.
    Only the entries with a market capitalisation above the cutoff in each period are kept.
    Parameters
    ----------
    stock_prices : pd.DataFrame
        DataFrame containing stock prices with a datetime index.
    inflation : pd.DataFrame
        DataFrame containing inflation discount multiples with a datetime index.
    config : CONFIGURATION_CLASS
        Configuration of the project.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the market entries where the market cap meets the cutoff.
    """
    # Unpack the config:
    min_marketcap: float = config.MIN_MARKETCAP_FIRM

    num_entries_before: int = stock_prices.shape[0]

    # Discount the market cap by inflation
    min_marketcap_discounted: pd.Series = _inflation_discount(
        inflation=inflation, value=min_marketcap
    )

    # Ensure alignment
    min_marketcap_discounted = min_marketcap_discounted.reindex(
        stock_prices.index
    ).ffill()

    mask = stock_prices["Lagged_MarketCap"] >= min_marketcap_discounted

    filtered_stock_prices: pd.DataFrame = stock_prices[mask]

    num_entries_end: int = filtered_stock_prices.shape[0]

    if config.LOG_INFO:
        config.logger.info(
            f"Applied market cap cutoff of {min_marketcap} to all periods using inflation discounted values.\
            \nNumber of entries dropped: {num_entries_before - num_entries_end}"
        )

    return filtered_stock_prices


def apply_marketcap_cutoff(
    stock_prices: pd.DataFrame, inflation: pd.DataFrame, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to apply the market cap cutoff.
    This can either be cutoff on the latest date or across the entire period after discounting for inflation, depending on the configuration.

    Parameters
    ----------
    stock_prices : pd.DataFrame
        DataFrame containing stock prices with a datetime index.
    inflation : pd.DataFrame
        DataFrame containing inflation discount multiples with a datetime index.
    config : CONFIGURATION_CLASS
        Configuration of the project.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the market entries where the market cap meets the cutoff according to the configuration.
    """

    # Apply market cap cutoff to filter the firms
    if config.DISCOUNT_MARKETCAP_FIRM_INFLATION:
        return _apply_marketcap_cutoff_allperiods(
            stock_prices=stock_prices, inflation=inflation, config=config
        )
    else:
        return _apply_marketcap_cutoff_latestperiod(
            stock_prices=stock_prices, config=config
        )


###################
# Factor Cleaning #
###################


def clean_factors(
    factors_monthly_raw: pd.DataFrame,
    factors_yearly_raw: pd.DataFrame,
    config: CONFIGURATION_CLASS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to clean the monthly and yearly factor info

    Parameters
    ----------
    factors_monthly_raw : pd.DataFrame
        Dataframe containing the monthly data for the factors
    factors_yearly_raw : pd.DataFrame
        Dataframe containing the yearly data for the factors
    config : CONFIGURATION_CLASS
        Configuration of the project

    Returns
    ------
    Tuple[pd.DataFrame,pd.DataFrame]
        Cleaned monthly and yearly factor data
    """
    factors_monthly_raw_newindex: pd.DataFrame = factors_monthly_raw.rename_axis(
        "date", axis="index"
    )
    factors_yearly_raw_newindex: pd.DataFrame = factors_yearly_raw.rename_axis(
        "date", axis="index"
    )

    factors_monthly_raw_decimal = factors_monthly_raw_newindex / 100
    factors_yearly_raw_decimal = factors_yearly_raw_newindex / 100

    if config.LOG_INFO:
        config.logger.info("Cleaned the factor data")
        config.logger.debug(
            f"Monthly factor data sample:\n\n{factors_monthly_raw_decimal.head(5)}\n"
        )

    return factors_monthly_raw_decimal, factors_yearly_raw_decimal


#####################
# Industry Cleaning #
#####################


def clean_ff_industry_portfolio(
    ff_industry_portfolios_raw: pd.DataFrame, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to clean the Fama French industry portfolio data.
    Aggregates the data to a single row per industry portfolio.

    Parameters
    ----------
    ff_industry_portfolios_raw: pd.DataFrame
        Raw Fama French industry portfolio data
    config: CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    pd.DataFrame
        Cleaned Fama French industry portfolio data with one row per industry and date
    """

    # Different sic codes
    sic_codes: List[Dict[str, Union[str, int]]] = []

    for _, row in ff_industry_portfolios_raw.iterrows():
        # Case when the range of sic is just one key
        if row["sic_start"] == row["sic_end"]:
            sic_codes.append(
                {
                    "siccode": row["sic_start"],
                    "industry_name": row["industry_name"],
                    "industry_id": row["industry_id"],
                }
            )
        else:
            for i in range(row["sic_start"], row["sic_end"] + 1):
                sic_codes.append(
                    {
                        "siccode": i,
                        "industry_name": row["industry_name"],
                        "industry_id": row["industry_id"],
                    }
                )
    result: pd.DataFrame = pd.DataFrame(sic_codes)

    if config.LOG_INFO:
        config.logger.info(
            "Finished cleaning Fama French industry portfolio data by aggregating to one row per industry and date"
        )
        config.logger.debug(
            f"Cleaned Fama French industry portfolio data sample:\n\n{result.head(5)}\n"
        )

    return result


######################
# Inflation Cleaning #
######################


def calculate_cum_inflation_multiplier(
    raw_monthly_inflation: pd.DataFrame, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to convert MoM inflation into the dicount multiplier from present value.
    This number can be multiplied to a monetary amount to get the equivalent amount at a past date.

    Parameters
    ----------
    raw_monthly_inflation: pd.DataFrame
       MoM inflation
    config: CONFIGURATION_CLASS
        Configuration of the Project

    Returns
    -------
    pd.DataFrame
        DataFrame containing the monthyl inflation discount multiplier
    """

    # Clean the data
    monthly_inflation_processed: pd.DataFrame = (
        raw_monthly_inflation.dropna().sort_index()
    )

    # Calculate the MoM multiplier
    monthly_multiplier: pd.Series = monthly_inflation_processed["MoM inflation"] + 1

    # Cumulate this to today
    cum_mult: pd.Series = monthly_multiplier.cumprod()

    # Normalise to last month = 1
    cum_mult_normalised = cum_mult / cum_mult.iloc[-1]
    monthly_inflation_processed["Inflation multiple"] = cum_mult_normalised

    if config.LOG_INFO:
        config.logger.info(
            "Calculated the cumulative inflation multiplier from the MoM inflation"
        )
        config.logger.debug(
            f"Cumulative inflation multiplier sample:\n\n{monthly_inflation_processed.head(5)}\n"
        )

    return monthly_inflation_processed


def intersect_stockprices_inflation(
    df_idx_to_keep: pd.DataFrame,
    monthly_inflation: pd.DataFrame,
    config: CONFIGURATION_CLASS,
) -> pd.DataFrame:
    """
    Function to intersect the stockprices with the inflation.
    Missing values are filled with the mean of previous and past info.

    Parameters
    ----------
    df_idx_to_keep: pd.DataFrame
        Dataframe with the already intersected index. This index values should be kept.
    monthly_inflation: pd.DataFrame
        Dataframe of the monthyl inflation that needs to be intersected
    config: CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    pd.DataFrame
        Dataframe containing the intersected inflation data"""

    # Reindex based on stockprices dates
    inflation_reindexed: pd.DataFrame = monthly_inflation.reindex(df_idx_to_keep.index)

    # Fill missing values with linear interpolation
    inflation_filled: pd.DataFrame = inflation_reindexed.sort_index().interpolate(
        method="time"
    )

    if config.LOG_INFO:
        config.logger.info(
            "Intersected stock prices and inflation data on common dates index"
        )
        config.logger.debug(
            f"Inflation data after intersection and filling missing values sample:\n\n{inflation_filled.head(5)}\n"
        )

    return inflation_filled


#######################
# Stock Info Cleaning #
#######################


def _remove_firms_missing_sharesoutstanding(
    stock_price: pd.DataFrame, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to remove those firms that have less shares outstanding than the threshold.
    Used to remove illiquid firms

    Parameters
    ----------
    stock_price : pd.DataFrame
        Dataframe containing the info about the stock and shares outstanding
    config : CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    pd.DataFrame
        New dataframe without the illiquid stocks"""
    # Unpack the configurations
    threshold_missing_shares: float = config.THRESHOLD_MISSING_SHARESOUTSTANDING

    # Remove columns with over threshold unactive trading (sharesoutstanding = 0)
    mask_activity = stock_price.groupby("gvkey")["sharesoutstanding"].apply(
        lambda s: s.le(0).sum() / s.size < threshold_missing_shares
    )

    result: pd.DataFrame = stock_price[
        stock_price["gvkey"].isin(mask_activity[mask_activity].index)
    ]

    if config.LOG_INFO:
        config.logger.info(
            f"Removed firms with more than {threshold_missing_shares * 100} of missing shares outstanding data"
        )
    return result


def _remove_non_significant_exchanges(
    stock_price: pd.DataFrame, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to remove firms listed on stock exchanges that should not be included.

    Parameters
    ----------
    stock_price : pd.DataFrame
        Dataframe containing the info about the stock and shares outstanding
    config : CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    pd.DataFrame
        New dataframe without the firms listed on non significant stock exchanges"""

    result: pd.DataFrame = stock_price[
        ~stock_price["exchange"].isin(config.EXCHANGES_TO_REMOVE)
    ]

    if config.LOG_INFO:
        config.logger.info(
            f"Removed firms listed on non significant stock exchanges (e.g. Toronto Stock Exchange). Removed {stock_price['gvkey'].nunique() - result['gvkey'].nunique()} firms"
        )

    return result


def _remove_small_firms(
    monthly_stock_prices_raw: pd.DataFrame, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to remove the entries for the firms that have a too small stock price.

    Parameters
    ----------
    monthly_stock_prices_raw: pd.DataFrame
        Monthly stock prices
    config
        Configuration of the project

    Returns
    -------
    pd.DataFrame
        The monthly entries without the firms"""

    num_entries_start: int = monthly_stock_prices_raw.shape[0]

    result: pd.DataFrame = monthly_stock_prices_raw[
        monthly_stock_prices_raw["close"] > config.MIN_STOCK_PRICE
    ]

    num_entries_end: int = result.shape[0]

    if config.logger:
        config.logger.info(
            f"Removed firms with share price of less than {config.MIN_STOCK_PRICE}.\n\
            Removed {num_entries_start-num_entries_end} entries"
        )

    return result


def _clip_monthly_return(
    monthly_stock_returns: pd.DataFrame, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to clip the return of an asset at a certain value.
    This avoids excessively large returns.
    Performs MIN(return_i, max_return) operation on each period.
    For sanity check, it also clips losses at 100%.

    Parameters
    ----------
    monthly_stock_returns: pd.DataFrame
        Monthly stock prices with return in decimal format
    config: CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    pd.DataFrame
        DataFrame containing the clipped returns.
        Note that the shape stays the same.
    """
    monthly_stock_returns.loc[:, "return"] = monthly_stock_returns["return"].clip(
        -1, config.MAX_MONTHLY_RETURN
    )
    return monthly_stock_returns


def _format_data_monthly_stock_prices(
    monthly_stock_prices: pd.DataFrame, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to format the data of the stock prices

    Parameters
    ----------
    monthly_stock_prices: pd.DataFrame
        Raw monthly stock prices to format
    config: CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    pd.DataFrame
        Dataframe of the returns in the right format"""

    # Convert numeric columns to numbers
    numeric_cols = ["close", "sharesoutstanding", "return"]
    monthly_stock_prices[numeric_cols] = monthly_stock_prices[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # Convert return from percentage to decimal
    monthly_stock_prices["return"] = monthly_stock_prices["return"] / 100

    return monthly_stock_prices


def clean_stock_prices(
    monthly_stock_prices_raw: pd.DataFrame, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to clean the stock prices

    Parameters
    ----------
    monthly_stock_prices_raw : pd.DataFrame
        Dataframe containing the raw stock prices
    config : CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    pd.DataFrame
        Processed stock prices
    """

    monthly_stock_prices_raw = monthly_stock_prices_raw.reset_index()

    # Drop duplicate gvkey-date entries
    monthly_stock_prices_raw = monthly_stock_prices_raw.drop_duplicates(
        subset=["gvkey", "date"]
    )

    monthly_stock_prices_formatted = _format_data_monthly_stock_prices(
        monthly_stock_prices_raw, config
    )

    # Remove firms with missing shares outstanding
    monthly_stock_prices_cleaned1 = _remove_firms_missing_sharesoutstanding(
        monthly_stock_prices_formatted, config
    )

    # remove the firms from non-significant exchanges
    monthly_stock_prices_cleaned2 = _remove_non_significant_exchanges(
        monthly_stock_prices_cleaned1, config
    )

    # remove the firms whose close price is too small
    monthly_stock_prices_cleaned3 = _remove_small_firms(
        monthly_stock_prices_cleaned2, config
    )

    # Clip the monthly return
    monthly_stock_prices_clipped_returns_cleaned: pd.DataFrame = _clip_monthly_return(
        monthly_stock_prices_cleaned3, config=config
    )

    if config.LOG_INFO:
        config.logger.info(
            f"Cleaned the stock prices. {monthly_stock_prices_raw['gvkey'].nunique()} -> {monthly_stock_prices_clipped_returns_cleaned['gvkey'].nunique()} firms"
        )
        config.logger.debug(
            f"Cleaned stock prices sample:\n\n{monthly_stock_prices_clipped_returns_cleaned.head(5)}\n"
        )

    return monthly_stock_prices_clipped_returns_cleaned


def _fill_missing_values(
    stock_price: pd.DataFrame, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to fill the missing dates (weekends are always missing) with the friday's data

    Parameters
    ----------
    stock_price : pd.DataFrame
        Dataframe containing the prices and other info that will be filled
    config: CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    pd.DataFrame
        Dataframe with the filled dates"""
    # Sort values again
    stock_price = stock_price.reset_index(names=["date"]).sort_values(["gvkey", "date"])

    # Fill the nas in shares outstanding with the previous value
    stock_price["sharesoutstanding"] = stock_price.groupby("gvkey")[
        "sharesoutstanding"
    ].ffill(limit=1)

    # Fill the nas in close with the last available value (usually weekends)
    stock_price["close"] = stock_price.groupby("gvkey")["close"].ffill(limit=1)

    # Reset date as index
    stock_price.set_index("date", inplace=True)

    if config.LOG_INFO:
        config.logger.info(
            "Filled missing values in stock prices with forward fill for shares outstanding and forward fill for close price to adjust for missing weekend data."
        )
    return stock_price


def intersect_stockprices_monthlyfactors(
    monthly_stock_prices_cleaned: pd.DataFrame,
    monthly_factors_processed: pd.DataFrame,
    config: CONFIGURATION_CLASS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to intersect the dates of the stockprices and monthlyfactors and only keep those dates
    that are present in both

    Parameters
    ----------
    monthly_stock_prices_cleaned : pd.DataFrame,
        Cleaned stock prices
    monthly_factors_processed : pd.DataFrame
        Processed monthly factors
    config : CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        - Stock prices with the common dates
        - Factor returns with common dates
    """
    # Resample the stock values based on a monthly frequency
    monthly_stock_prices = monthly_stock_prices_cleaned.set_index("date", drop=True)

    # Shift the days of the returns to the first of the next month (FF standard)
    idx = cast(pd.DatetimeIndex, monthly_stock_prices.index)
    monthly_stock_prices.index = idx.to_period("M").to_timestamp()

    # Align on common index
    common_idx: pd.Index = monthly_stock_prices.index.intersection(
        cast(pd.DatetimeIndex, monthly_factors_processed.index)
    )
    monthly_stock_prices_aligned = monthly_stock_prices.loc[common_idx]
    monthly_factors_aligned = monthly_factors_processed.loc[common_idx]

    # Check on how much data was dropped
    dates_num_ff_before: int = monthly_factors_processed.shape[0]
    dates_num_ff_after: int = monthly_factors_aligned.shape[0]
    date_num_stocks_before: int = monthly_stock_prices.index.nunique()
    date_num_stocks_after: int = monthly_stock_prices.index.nunique()

    if config.LOG_INFO:
        config.logger.info(
            "Intersected stock prices and monthly factors on common dates index."
        )
        config.logger.debug(
            f"Dropped {dates_num_ff_before-dates_num_ff_after} dates for the FF-data."
        )
        config.logger.debug(
            f"Dropped {date_num_stocks_before-date_num_stocks_after} dates for the market entries during alignment."
        )
        config.logger.debug(
            f"Stock prices after intersection and filling missing values sample:\n\n{monthly_stock_prices_aligned.head(5)}\n"
        )

    return monthly_stock_prices_aligned, monthly_factors_aligned


######################
# Firm Info Cleaning #
######################


def clean_firm_info(
    firm_info_raw: pd.DataFrame, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to clean the firm info

    Parameters
    ----------
    firm_info_raw : pd.DataFrame
        Raw firm info
    config : CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    pd.DataFrame
        Processed firm info"""

    if config.LOG_INFO:
        config.logger.info("Cleaned the firm info")
        config.logger.debug(f"Cleaned firm info sample:\n\n{firm_info_raw.head(5)}\n")

    return firm_info_raw


############################
# SIC Description Cleaning #
############################


def clean_sic_desc_raw(
    sic_desc_raw: pd.DataFrame, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to clean the sic codes

    Parameters
    ----------
    sic_desc_raw : pd.DataFrame
        Raw SIC description
    config : CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    pd.DataFrame
        Processed SIC description"""
    # Remove all inactive SIC codes
    sic_desc_raw = sic_desc_raw[sic_desc_raw["status"] == "A"]

    # Remove the status column
    sic_desc_raw = sic_desc_raw.drop(columns=["status"])

    if config.LOG_INFO:
        config.logger.info(
            "Cleaned the SIC description data by removing inactive codes and status column"
        )
        config.logger.debug(
            f"Cleaned SIC description sample:\n\n{sic_desc_raw.head(5)}\n"
        )

    return sic_desc_raw


##################
# Main functions #
##################


def clean_data(config: CONFIGURATION_CLASS) -> DATAFRAME_CONTAINER:
    """
    Function to clean all of the data.

    Parameters
    ----------
    config : CONFIGURATION_CLASS
        COnfiguration of the project

    Returns
    -------
    DATAFRAME_CONTAINER
        Container with the cleaned dataframes"""

    # Download the data
    raw_data: DATAFRAME_CONTAINER = download_raw_data(config)
    if raw_data.monthly_fama_french is None or raw_data.yearly_fama_french is None:
        raise ValueError("Factors dataframes cannot be None")
    if not isinstance(raw_data.monthly_inflation, pd.DataFrame):
        raise ValueError("Monthly inflation data must be a DataFrame")

    if config.LOG_INFO:
        config.logger.info("Starting data cleaning process....\n" + "-" * 80)

    # Process the factor data
    monthly_factors_processed, factors_yearly_processed = clean_factors(
        raw_data.monthly_fama_french, raw_data.yearly_fama_french, config
    )
    # Process the industry portfolio data
    ff_industry_portfolio_processed: pd.DataFrame = clean_ff_industry_portfolio(
        raw_data.ff_industry_portfolios, config
    )

    # Clean the stock data
    monthly_monthly_stock_prices_cleaned: pd.DataFrame = clean_stock_prices(
        raw_data.monthly_stock_info, config
    )

    # Intersect the stock prices with the monthly factors
    monthly_stock_prices_intersected, monthly_factors_processed = (
        intersect_stockprices_monthlyfactors(
            monthly_monthly_stock_prices_cleaned, monthly_factors_processed, config
        )
    )

    # Process the firm info
    firm_info_processed: pd.DataFrame = clean_firm_info(raw_data.firm_info, config)

    # Process the SIC description
    sic_desc_processed: pd.DataFrame = clean_sic_desc_raw(raw_data.sic_info, config)

    # Process the inflation info
    cum_inflation_multiplier: pd.DataFrame = calculate_cum_inflation_multiplier(
        raw_data.monthly_inflation, config
    )

    # Intersect the stock prices with the inflation data
    cum_inflation_multiplier_intersected: pd.DataFrame = (
        intersect_stockprices_inflation(
            monthly_factors_processed, cum_inflation_multiplier, config
        )
    )

    # Calculate the marketcap for all entries
    monthly_stock_prices_with_marketcap: pd.DataFrame = get_market_cap(
        stock_prices=monthly_stock_prices_intersected,
        inflation=cum_inflation_multiplier_intersected,
        config=config,
    )

    # Apply the marketcap cutoff
    monthly_stock_prices_market_cap_filtered: pd.DataFrame = apply_marketcap_cutoff(
        stock_prices=monthly_stock_prices_with_marketcap,
        inflation=cum_inflation_multiplier_intersected,
        config=config,
    )

    if config.LOG_INFO:
        config.logger.info("Completed cleaning process\n")

    return DATAFRAME_CONTAINER(
        monthly_fama_french=monthly_factors_processed,
        yearly_fama_french=factors_yearly_processed,
        monthly_stock_info=monthly_stock_prices_market_cap_filtered,
        firm_info=firm_info_processed,
        sic_info=sic_desc_processed,
        monthly_inflation=cum_inflation_multiplier_intersected,
        ff_industry_portfolios=ff_industry_portfolio_processed,
    )


def clean_save_data(config: CONFIGURATION_CLASS) -> None:
    """
    Function to clean and save the entire data.

    Parameters
    ----------
    config : CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    None"""

    data: DATAFRAME_CONTAINER = clean_data(config)

    save_processed_data(
        data,
        config,
    )


if __name__ == "__main__":
    clean_save_data(PROJ_CONFIG)
