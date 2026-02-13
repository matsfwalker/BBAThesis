from typing import Tuple

import pandas as pd

# Import the configurations
from configs import CONFIG, CONFIGURATION, FILENAMES, DATAFRAME_CONTAINER


def download_raw_data(config: CONFIGURATION) -> DATAFRAME_CONTAINER:
    """
    Function to download all of the raw data.

    Parameters
    ----------
    config : CONFIGURATION
        COnfiguration of the project

    Returns
    ------
    DATAFRAME_CONTAINER
        The 5 different raw dataframes
        - factors_monthly_raw
        - factors_yearly_raw
        - stock_prices_raw
        - firm_info_raw
        - sic_desc_raw"""

    factors_monthly_raw: pd.DataFrame = pd.read_csv(
        config.paths.raw_read(FILENAMES.FF5_factors_monthly),
        parse_dates=["date"],
        index_col="date",
    )
    factors_yearly_raw: pd.DataFrame = pd.read_csv(
        config.paths.raw_read(FILENAMES.FF5_factors_yearly),
        parse_dates=["date"],
        index_col="date",
    )

    stock_prices_raw: pd.DataFrame = pd.read_csv(
        config.paths.raw_read(FILENAMES.Stock_prices),
        parse_dates=["date"],
        index_col="date",
    )

    firm_info_raw: pd.DataFrame = pd.read_csv(
        config.paths.raw_read(FILENAMES.Firm_info)
    )

    sic_desc_raw: pd.DataFrame = pd.read_csv(
        config.paths.raw_read(FILENAMES.Sic_description)
    )

    monthly_inflation_info_raw: pd.DataFrame = pd.read_csv(
        config.paths.raw_read(FILENAMES.Inflation_info_monthly),
        parse_dates=["date"],
        index_col="date",
    )

    return DATAFRAME_CONTAINER(
        monthly_fama_french=factors_monthly_raw,
        yearly_fama_french=factors_yearly_raw,
        stock_market_info=stock_prices_raw,
        firm_info=firm_info_raw,
        sic_info=sic_desc_raw,
        monthly_inflation=monthly_inflation_info_raw
    )


def clean_factors(
    factors_monthly_raw: pd.DataFrame,
    factors_yearly_raw: pd.DataFrame,
    config: CONFIGURATION,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to clean the monthly and yearly factor info

    Parameters
    ----------
    factors_monthly_raw : pd.DataFrame
        Dataframe containing the monthly data for the factors
    factors_yearly_raw : pd.DataFrame
        Dataframe containing the yearly data for the factors
    config : CONFIGURATION
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

    return factors_monthly_raw_decimal, factors_yearly_raw_decimal


def remove_firms_missing_sharesoutstanding(
    stock_price: pd.DataFrame, config: CONFIGURATION
) -> pd.DataFrame:
    """
    Function to remove those firms that have less shares outstanding than the threshold.
    Used to remove illiquid firms

    Parameters
    ----------
    stock_price : pd.DataFrame
        Dataframe containing the info about the stock and shares outstanding
    config : CONFIGURATION
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

    return stock_price[stock_price["gvkey"].isin(mask_activity[mask_activity].index)]


def clean_stock_prices(
    stock_prices_raw: pd.DataFrame, config: CONFIGURATION
) -> pd.DataFrame:
    """
    Function to clean the stock prices

    Parameters
    ----------
    stock_prices_raw : pd.DataFrame
        Dataframe containing the raw stock prices
    config : CONFIGURATION
        Configuration of the project

    Returns
    -------
    pd.DataFrame
        Processed stock prices
    """

    stock_prices_raw = stock_prices_raw.reset_index()

    # Drop duplicate gvkey-date entries
    stock_prices_raw = stock_prices_raw.drop_duplicates(subset=["gvkey", "date"])

    # Convert numeric columns to numbers
    numeric_cols = ["close", "sharesoutstanding"]
    stock_prices_raw[numeric_cols] = stock_prices_raw[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # Remove firms with missing shares outstanding
    stock_prices_cleaned = remove_firms_missing_sharesoutstanding(
        stock_prices_raw, config
    )

    return stock_prices_cleaned


def fill_missing_values(
    stock_price: pd.DataFrame,
) -> pd.DataFrame:
    """
    Function to fill the missing dates (weekends are always missing) with the friday's data

    Parameters
    ----------
    stock_price : pd.DataFrame
        Dataframe containing the prices and other info that will be filled

    Returns
    -------
    pd.DataFrame
        Dataframe with the filled dates"""
    # Sort values again
    stock_price = stock_price.reset_index(names=["date"]).sort_values(["gvkey", "date"])

    # Fill the nas in shares outstanding with the previous value
    stock_price["sharesoutstanding"] = stock_price.groupby("gvkey")[
        "sharesoutstanding"
    ].ffill()

    # Fill the nas in close with the mean of previous and last
    stock_price["close"] = stock_price.groupby("gvkey")["close"].transform(
        lambda s: s.interpolate(method="linear", limit_area="inside")
    )

    return stock_price


def intersect_stockprices_monthlyfactors(
    stock_prices_cleaned: pd.DataFrame,
    factors_monthly_processed: pd.DataFrame,
    config: CONFIGURATION,
) -> pd.DataFrame:
    """
    Function to intersect the dates of the stockprices and monthlyfactors and only keep those dates
    that are present in both

    Parameters
    ----------
    stock_prices_cleaned : pd.DataFrame,
        Cleaned stock prices
    factors_monthly_processed : pd.DataFrame
        Processed monthly factors
    config : CONFIGURATION
        Configuration of the project

    Returns
    -------
    pd.DataFrame
        Stock prices with the common dates
    """

    # Create full calendar-day index from min to max date
    full_idx: pd.DatetimeIndex = pd.date_range(
        stock_prices_cleaned["date"].min(),
        factors_monthly_processed.index.max(),
        freq="D",
    )

    stock_prices_reindexed = (
        stock_prices_cleaned.set_index(["gvkey", "date"])
        .sort_index()
        .groupby(level="gvkey", group_keys=False)
        .apply(
            lambda g: g.reindex(
                pd.MultiIndex.from_product(
                    [[g.index.get_level_values("gvkey")[0]], full_idx],
                    names=["gvkey", "date"],
                )
            ).ffill(limit=2)
        )
        .reset_index(level="gvkey")
    )

    # Intersect the dates with the Fama French factors
    common_dates = stock_prices_reindexed.index.intersection(
        factors_monthly_processed.index
    )

    # Only keep the common dates
    stock_prices_common_date: pd.DataFrame = stock_prices_reindexed.loc[
        common_dates
    ].sort_index()

    # Fill the missing values
    stock_prices_filled: pd.DataFrame = fill_missing_values(stock_prices_common_date)

    return stock_prices_filled


def clean_firm_info(firm_info_raw: pd.DataFrame, config: CONFIGURATION) -> pd.DataFrame:
    """
    Function to clean the firm info

    Parameters
    ----------
    firm_info_raw : pd.DataFrame
        Raw firm info
    config : CONFIGURATION
        Configuration of the project

    Returns
    -------
    pd.DataFrame
        Processed firm info"""
    return firm_info_raw


def clean_sic_desc_raw(
    sic_desc_raw: pd.DataFrame, config: CONFIGURATION
) -> pd.DataFrame:
    """
    Function to clean the sic codes

    Parameters
    ----------
    sic_desc_raw : pd.DataFrame
        Raw SIC description
    config : CONFIGURATION
        Configuration of the project

    Returns
    -------
    pd.DataFrame
        Processed SIC description"""
    # Remove all inactive SIC codes
    sic_desc_raw = sic_desc_raw[sic_desc_raw["status"] == "A"]

    # Remove the status column
    sic_desc_raw = sic_desc_raw.drop(columns=["status"])

    return sic_desc_raw


def calculate_cum_inflation_multiplier(
    raw_monthly_inflation: pd.DataFrame,
    config: CONFIGURATION
)->pd.DataFrame:
    """
    Function to convert MoM inflation into the dicount multiplier from present value.
    This number can be multiplied to a monetary amount to get the equivalent amount at a past date.
    
    Parameters
    ----------
    raw_monthly_inflation: pd.DataFrame
       MoM inflation
    config: CONFIGURATION
        Configuration of the Project

    Returns
    -------
    pd.DataFrame
        DataFrame containing the monthyl inflation discount multiplier
    """

    # Clean the data
    monthly_inflation_processed: pd.DataFrame = raw_monthly_inflation.dropna().sort_index()

    # Calculate the MoM multiplier
    monthly_multiplier: pd.Series = monthly_inflation_processed["MoM inflation"] + 1

    # Cumulate this to today
    cum_mult: pd.DataFrame = monthly_multiplier.cumprod()

    # Normalise to last month = 1
    cum_mult_normalised = cum_mult / cum_mult.iloc[-1]
    monthly_inflation_processed["Inflation multiple"] = cum_mult_normalised

    return monthly_inflation_processed 


def intersect_stockprices_inflation(
    df_idx_to_keep: pd.DataFrame,
    monthly_inflation: pd.DataFrame,
    config: CONFIGURATION
)-> pd.DataFrame:
    """
    Function to intersect the stockprices with the inflation.
    Missing values are filled with the mean of previous and past info.
    
    Parameters
    ----------
    df_idx_to_keep: pd.DataFrame
        Dataframe with the already intersected index. This index values should be kept.
    monthly_inflation: pd.DataFrame
        Dataframe of the monthyl inflation that needs to be intersected
    config: CONFIGURATION
        Configuration of the project
        
    Returns
    -------
    pd.DataFrame
        Dataframe containing the intersected inflation data"""
    
    # Reindex based on stockprices dates
    inflation_reindexed: pd.DataFrame = monthly_inflation.reindex(df_idx_to_keep.index)

    # Fill missing values with linear interpolation
    inflation_filled: pd.DataFrame = inflation_reindexed.sort_index().interpolate(method="time")

    return inflation_filled


def save_processed_data(
    data: DATAFRAME_CONTAINER,
    config: CONFIGURATION,
) -> None:
    """
    Function to save all of the processed data in the data/processed dir.

    Parameters
    ----------
    data: DATAFRAME_CONTAINER
        Container containing the all processed dataframes of the project
    config: CONFIGURATION
        Configuration of the project

    Returns
    -------
    None
    """

    # Unpack the data
    factors_monthly_processed: pd.DataFrame = data.monthly_fama_french
    factors_yearly_processed: pd.DataFrame = data.yearly_fama_french
    stock_prices_intersected: pd.DataFrame = data.stock_market_info
    firm_info_processed: pd.DataFrame = data.firm_info
    sic_desc_processed: pd.DataFrame = data.sic_info
    inflation_processed: pd.DataFrame = data.monthly_inflation

    factors_monthly_processed.to_csv(
        config.paths.processed_out(FILENAMES.FF5_factors_monthly)
    )
    factors_yearly_processed.to_csv(
        config.paths.processed_out(FILENAMES.FF5_factors_yearly)
    )

    stock_prices_intersected.to_csv(config.paths.processed_out(FILENAMES.Stock_prices))

    firm_info_processed.to_csv(
        config.paths.processed_out(FILENAMES.Firm_info), index=False
    )

    sic_desc_processed.to_csv(
        config.paths.processed_out(FILENAMES.Sic_description), index=False
    )

    inflation_processed.to_csv(
        config.paths.processed_out(FILENAMES.Inflation_info_monthly)
    )

    return


def clean_data(config: CONFIGURATION) -> DATAFRAME_CONTAINER:
    """
    Function to clean all of the data.

    Parameters
    ----------
    config : CONFIGURATION
        COnfiguration of the project

    Returns
    -------
    DATAFRAME_CONTAINER
        Container with the cleaned dataframes"""

    # Download the data
    raw_data: DATAFRAME_CONTAINER = download_raw_data(config)

    # Process the factor data
    factors_monthly_processed, factors_yearly_processed = clean_factors(
        raw_data.monthly_fama_french, raw_data.yearly_fama_french, config
    )

    # Clean the stock data
    stock_prices_cleaned: pd.DataFrame = clean_stock_prices(raw_data.stock_market_info, config)

    # Intersect the stock prices with the monthly factors
    stock_prices_intersected: pd.DataFrame = intersect_stockprices_monthlyfactors(
        stock_prices_cleaned, factors_monthly_processed, config
    )

    firm_info_processed: pd.DataFrame = clean_firm_info(raw_data.firm_info, config)

    sic_desc_processed: pd.DataFrame = clean_sic_desc_raw(raw_data.sic_info, config)

    cum_inflation_multiplier: pd.DataFrame = calculate_cum_inflation_multiplier(raw_data.monthly_inflation, config)

    # Intersect the stock prices with the inflation data
    cum_inflation_multiplier_intersected: pd.DataFrame = intersect_stockprices_inflation(
        factors_monthly_processed, cum_inflation_multiplier, config
    )

    return DATAFRAME_CONTAINER(
        monthly_fama_french=factors_monthly_processed,
        yearly_fama_french=factors_yearly_processed,
        stock_market_info=stock_prices_intersected,
        firm_info=firm_info_processed,
        sic_info=sic_desc_processed,
        monthly_inflation=cum_inflation_multiplier_intersected
    )


def clean_save_data(config: CONFIGURATION) -> None:
    """
    Function to clean and save the entire data.

    Parameters
    ----------
    config : CONFIGURATION
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
    clean_save_data(CONFIG)
