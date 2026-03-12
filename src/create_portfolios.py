from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
import warnings

from configs import PROJ_CONFIG, CONFIGURATION_CLASS, FILENAMES_CLASS, DATAFRAME_CONTAINER

####################
# Download/Exports #
####################

def download_processed_data(
    config: CONFIGURATION_CLASS,
) -> DATAFRAME_CONTAINER:
    """
    Function to read the stock prices, firm info, and SIC code descriptions from the processed data directory.

    Parameters
    ----------
    config : CONFIGURATION_CLASS
        Configuration of the model

    Returns
    -------
    DATAFRAME_CONTAINER
        A container for the dataframes.
    """

    if config.LOG_INFO:
        config.logger.info("Starting to download processed data...\n" + "-" * 80)

    stock_prices: pd.DataFrame = pd.read_csv(
        config.paths.processed_read(FILENAMES_CLASS.Stock_prices),
        parse_dates=["date"],
        index_col="date",
    )

    firm_info: pd.DataFrame = pd.read_csv(
        config.paths.processed_read(FILENAMES_CLASS.Firm_info)
    )

    sic_codes: pd.DataFrame = pd.read_csv(
        config.paths.processed_read(FILENAMES_CLASS.Sic_description)
    )

    inflation: pd.DataFrame = pd.read_csv(
        config.paths.processed_read(FILENAMES_CLASS.Inflation_info_monthly),
        parse_dates=["date"],
        index_col="date",
    )

    ff_industry_portfolios: pd.DataFrame = pd.read_csv(
        config.paths.processed_read(FILENAMES_CLASS.FF5_industry_portfolios)
    )

    if config.LOG_INFO:
        config.logger.info("Successfully downloaded the processed data")

    return DATAFRAME_CONTAINER(
        monthly_stock_info=stock_prices,
        firm_info=firm_info,
        sic_info=sic_codes,
        monthly_inflation=inflation,
        ff_industry_portfolios=ff_industry_portfolios,
    )


def save_portfolio_returns_constitution(
    portfolios: pd.DataFrame,
    config: CONFIGURATION_CLASS,
) -> None:
    """
    Function to save the portfolio returns and constitution details to CSV files.

    Parameters
    ----------
    portfolios : pd.DataFrame
        DataFrame containing the returns and info of the portfolios.
    config : CONFIGURATION_CLASS
        Configuration of the project.

    Returns
    -------
    None
        This function saves the dataframes to CSV files and does not return anything.
    """

    if config.LOG_INFO:
        config.logger.info(
            "Saving the portfolio returns and constitution details to CSV files...\n"
            + "-" * 80
        )

    # Save the results
    portfolios.to_csv(config.paths.portfolios_out(FILENAMES_CLASS.Portfolio_info))

    if config.LOG_INFO:
        config.logger.info(
            "Successfully saved the portfolio returns and constitution details"
        )

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


def _present_value_inflation(
    inflation: pd.DataFrame, values: pd.Series
) -> pd.Series:
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
    assert isinstance(inflation.index, pd.DatetimeIndex), "The index of the inflation dataframe should be a DatetimeIndex."
    assert isinstance(values.index, pd.DatetimeIndex), "The index of the values series should be a DatetimeIndex."

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
        stacklevel=2
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
    returns: pd.DataFrame = sorted_prices.set_index("date").dropna(subset=["return"]).sort_index()

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
    assert "close" in stock_prices.columns, "Close price column is missing in the stock prices dataframe."
    assert "sharesoutstanding" in stock_prices.columns, "Shares outstanding column is missing in the stock prices dataframe."
    assert isinstance(stock_prices.index, pd.DatetimeIndex), "The index of the stock prices dataframe should be a DatetimeIndex."

    stock_prices = stock_prices.copy()
    stock_prices["market_cap"] = _compute_market_cap(
        stock_prices, "close", "sharesoutstanding"
    )
    stock_prices["Lagged_MarketCap"] = stock_prices.groupby("gvkey")["market_cap"].shift(1)

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

    surviving_gvkeys = latest_prices.loc[latest_prices["Lagged_MarketCap"] >= marketcap_cutoff, "gvkey"].unique()

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
    min_marketcap_discounted = min_marketcap_discounted.reindex(stock_prices.index).ffill()

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


#######################
# Industry Assignment #
#######################

def _format_sic_codes(
    sic_descr: pd.DataFrame, level: Literal[1, 2, 3, 4], sic_col: str = "siccode"
) -> pd.DataFrame:
    """
    Helper function to format SIC codes to a given level.
    Keep only SIC codes that belong EXACTLY to the chosen level, dropping all coarser levels.

    Parameters
    ----------
    sic_descr : pd.DataFrame
        DataFrame containing SIC codes and their descriptions.
    level : Literal[1,2,3,4]
        The desired SIC code level (1 to 4).
    sic_col : str = "siccode"
        The name of the column containing SIC codes.
        Default is "siccode".

    Returns
    -------
    pd.DataFrame
        A DataFrame containing SIC codes and their descriptions at the specified level.
    """
    # Copy and clean the DataFrame
    df = sic_descr.copy()
    df.dropna(subset=[sic_col], inplace=True)

    # Convert SIC codes to integers
    df["sic_int"] = pd.to_numeric(df[sic_col], errors="coerce").astype("Int64")

    # Condition for belonging to level L
    step_L = 10 ** (4 - level)
    cond = df["sic_int"] % step_L == 0

    df.rename(columns={sic_col: "sic_level"}, inplace=True)

    return df[cond][["sic_level", "sicdescription"]]


def _format_firms_sic(
    entries: pd.DataFrame, level: Literal[1, 2, 3, 4], sic_col: str = "siccode"
) -> pd.DataFrame:
    """
    Helper function to format firms' SIC codes to a given level.

    Parameters
    ----------
    entries : pd.DataFrame
        DataFrame containing firm SIC code and gvkey
    level : Literal[1,2,3,4]
        The desired SIC code level (1 to 4).
    sic_col : str = "siccode"
        The name of the column containing SIC codes.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing firms with SIC codes normalized to the specified level.
    """
    # Make sure the necessary information is present
    assert sic_col in entries.columns, f"{sic_col} column is missing in the entries dataframe."
    assert "gvkey" in entries.columns, "gvkey column is missing in the entries dataframe."

    # Copy and format the firms
    entries: pd.DataFrame = entries.copy()
    entries.dropna(subset=[sic_col, "gvkey"], inplace=True)

    # Normalise the sic code using floor division
    entries["sic_level"] = np.floor(entries[sic_col] / (10 ** (4 - level))) * (
        10 ** (4 - level)
    )
    entries["sic_level"] = entries["sic_level"].astype("Int64")

    return entries


def _assign_industry_to_firms_siclevel(
    entries: pd.DataFrame, sic_descr: pd.DataFrame, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to assign the firms to a specific industry by sic code level.
    This is done using their sic codes and a sic code description dataframe.
    The firms can be grouped according to different levels of the sic code hierarchy.

    Parameters
    ----------
    entries : pd.DataFrame
        Dataframe containing firm information per period including sic codes.
    sic_descr : pd.DataFrame
        Dataframe containing sic code descriptions.
    config : CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    pd.DataFrame
        Dataframe with the gvkey of the firms, the sic level and the corresponding sic code description at the specified level.
    """
    if config.SIC_LEVEL is None:
        raise ValueError(
            "SIC_LEVEL must be specified when using SIC code level classification."
        )
    # Unpack the config:
    sic_level: Literal[1, 2, 3, 4] = config.SIC_LEVEL

    # Format firms and sic codes according to the level
    firms: pd.DataFrame = _format_firms_sic(entries, sic_level)
    descr: pd.DataFrame = _format_sic_codes(sic_descr, sic_level)

    # Merge the two dataframes
    result:pd.DataFrame = (firms
                            .join(descr.set_index("sic_level"), on="sic_level", how="left")
                            .rename(
                                columns={"sic_level": "key", "sicdescription": "industry_name"}
                            ))

    if config.LOG_INFO:
        config.logger.info(
            f"Successfully assigned an industry to firms using SIC codes at level {sic_level}"
        )

    return result


def _assign_industry_firms_ffindustries(
    entries: pd.DataFrame, ff_industries: pd.DataFrame, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to assign to each firm its industry based on the Fama-French industry classification.

    Parameters
    ----------
    entries : pd.DataFrame
        DataFrame containing market entries of the firms including their SIC codes.
    ff_industries : pd.DataFrame
        DataFrame containing Fama-French industry classifications for each SIC-code.
    config : CONFIGURATION_CLASS
        Configuration of the project.

    Returns
    -------
    pd.DataFrame
        DataFrame with the gvkey of the firms and their corresponding Fama-French industry classification.
    """
    # Asserts to make sure the necessary information is present
    assert "gvkey" in entries.columns, "gvkey column is missing in the market entries dataframe."
    assert "siccode" in entries.columns, "siccode column is missing in the market entries dataframe."

    # Merge the two dataframes
    result: pd.DataFrame = (entries
                            .join(ff_industries.set_index("siccode"), on="siccode", how="left")
                            .rename(columns={"industry_id": "key"}))

    if config.LOG_INFO:
        config.logger.info(
            "Successfully assigned industries to firms according to the Fama-French industry portfolios"
        )

    return result


def assign_industry(
    entries: pd.DataFrame,
    sic_descr: pd.DataFrame,
    ff_industry_portfolios: pd.DataFrame,
    config:CONFIGURATION_CLASS,
) -> pd.DataFrame:
    """
    Function to assign an industry to each firm based on the configuration.
    This ca either be done according to the SIC code level or the Fama-French industry classification.

    Parameters
    ----------
    entries : pd.DataFrame
        DataFrame containing stock info per period including SIC codes.
    sic_descr : pd.DataFrame
        DataFrame containing SIC code descriptions.
    ff_industry_portfolios : pd.DataFrame
        DataFrame containing Fama-French industry classifications for each SIC-code.
    config : CONFIGURATION_CLASS
        Configuration of the project.

    Returns
    -------
    pd.DataFrame
        DataFrame of the market entries with a new column "industry" containing the industry classification of each firm according to the configuration.
    """

    if config.INDUSTRY_CLASSIFICATION_METHOD == "Sic_level":
        return _assign_industry_to_firms_siclevel(entries, sic_descr, config)
    else:
        return _assign_industry_firms_ffindustries(
            entries, ff_industry_portfolios, config
        )


######################
# Portfolio Creation #
######################

def portfolio_information(
    portfolio_subset: pd.DataFrame,
    industry_name: str,
    MarketCapID: str,
    config: CONFIGURATION_CLASS
)->pd.Series:
    """
    Function to add additional information to the portfolio, such as the number of firms and total market cap.

    Parameters
    ----------
    portfolio_subset : pd.DataFrame
        DataFrame containing the subset of the portfolio for a specific industry and date.
    industry_name: str
        The name of the industry for which the portfolio is being created.
    MaerketCapID: str
        The market cap based sub-portfolio identifier (e.g. "all", "large_cap", "small_cap").
    config : CONFIGURATION_CLASS
        Configuration of the project.

    Returns
    -------
    pd.Series
        A Series containing the  information for the industry portfolio.
    """

    return pd.Series({
        "industry_name": industry_name,
        "num_firms": portfolio_subset["gvkey"].nunique(),
        "total_market_cap": portfolio_subset["market_cap"].sum(),
        "total_presentvalue_marketcap": (portfolio_subset["market_cap_present_value"].sum() if "market_cap_present_value" in portfolio_subset.columns else np.nan),
        "gvkeys": portfolio_subset["gvkey"].unique().tolist(),
        "firm_names": portfolio_subset["companyname"].unique().tolist(),
        "return": calculate_portfolio_return(portfolio_subset, config),
        "MarketCapID": MarketCapID,
        "date": portfolio_subset.index[0],
    })


def marketcap_subportfolio_information(
    industry_subset: pd.DataFrame,
    industry_name: str,
    config: CONFIGURATION_CLASS
)->List[pd.Series]:
    """
    Function to create the market cap based sub-portfolio information for a given industry portfolio subset.
    
    Parameters
    ----------
    industry_subset : pd.DataFrame
        DataFrame containing the subset of the portfolio for a specific industry and date.
    industry_name: str
        The name of the industry for which the portfolio is being created.
    config : CONFIGURATION_CLASS
        Configuration of the project.
    
    Returns
    -------
    List[pd.Series]
        A list of Series, each containing the information for a market cap based sub-portfolio (large cap and small cap) within the industry portfolio.
    """
    num_firms_portfolio = config.MARKETCAP_PORTFOLIO_NUMBER_FIRMS
    percentile_portfolio = config.MARKETCAP_PORTFOLIO_PERCENTILE

    # determine cutoff
    if percentile_portfolio is not None:
        cutoff = int(len(industry_subset) * percentile_portfolio)
    elif num_firms_portfolio is not None:
        cutoff = min(num_firms_portfolio, len(industry_subset) // 2)
    else:
        raise ValueError(
            "Either percentile or number of firms for portfolios needs to be provided."
        )
    if cutoff == 0:
        return []
    
    # order firms by marketcap
    industry_subset_sorted: pd.DataFrame = industry_subset.sort_values("market_cap", ascending=False)

    top_firms = industry_subset_sorted.iloc[:cutoff].copy()
    bottom_firms = industry_subset_sorted.iloc[-cutoff:].copy()

    return [
        portfolio_information(top_firms, industry_name + " - Large Cap", "large_cap", config),
        portfolio_information(bottom_firms, industry_name + " - Small Cap", "small_cap", config)
    ]


def calculate_portfolio_return(
    portfolio_subset: pd.DataFrame, config: CONFIGURATION_CLASS
) -> float:
    """
    Function to calculate the return of a portfolio subset based on the returns of the individual stocks and their weights.

    Parameters
    ----------
    portfolio_subset : pd.DataFrame
        DataFrame containing the subset of the portfolio for a specific industry and date, including stock returns and market caps.
    config : CONFIGURATION_CLASS
        Configuration of the project.

    Returns
    -------
    float
        The calculated return of the portfolio.
    """
    # Calculate the weights of the stocks in the portfolio
    portfolio_subset["Weight"] = _calculate_weight_in_portfolio(
        firm_subset=portfolio_subset, config=config
    )

    # Calculate the return of the portfolio as the weighted sum of the returns of the individual stocks
    portfolio_return: float = (portfolio_subset["Weight"] * portfolio_subset["return"]).sum()

    return portfolio_return


def create_all_portfolios(
    market_industry_info: pd.DataFrame,
    config: CONFIGURATION_CLASS,
)->Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Function to create the industry portfolios based on the market entries and their assigned industries.
    Creates the standard industry portfolios and their sub-portfolios based on market capitalization.
    Returns first the description of the portfolios and then the returns.
    
    Parameters
    ----------
    market_industry_info : pd.DataFrame
        DataFrame containing the market entries of the firms and their assigned industries.
    config : CONFIGURATION_CLASS
        Configuration of the project.
    
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing two DataFrames:
        - The first DataFrame contains the description of the portfolios, including their industry classification and market cap based sub-portfolios.
        - The second DataFrame contains the returns of the portfolios.
    """

    # Asserts
    assert "industry_name" in market_industry_info.columns, "The industry_name column is missing in the market industry info dataframe."
    assert "gvkey" in market_industry_info.columns, "The gvkey column is missing in the market industry info dataframe."
    assert "market_cap_present_value" in market_industry_info.columns, "The market_cap_present_value column is missing in the market industry info dataframe."
    assert "market_cap" in market_industry_info.columns, "The market_cap column is missing in the market industry info dataframe."
    assert "close" in market_industry_info.columns, "The close column is missing in the market industry info dataframe."

    market_industry_info = market_industry_info.sort_values(["gvkey", "date"])

    # Get the lagged market cap to calculate the weight if necessary
    market_industry_info = market_industry_info.dropna(subset=["Lagged_MarketCap"])

    dates: pd.DatetimeIndex = market_industry_info.index.unique(level=0).sort_values()
    industries: pd.Series = market_industry_info["industry_name"].unique()

    rows: List[pd.Series] = []

    for date in dates:  # Skip the first date, since there is no data
        subset_date: pd.DataFrame = market_industry_info.loc[date]
        for industry in industries:
            subset = subset_date[subset_date["industry_name"] == industry].copy()

            if subset.empty:
                continue
        
            rows.append(portfolio_information(portfolio_subset=subset, 
                                              industry_name=industry,
                                              MarketCapID="all", 
                                              config=config))

            rows.extend(marketcap_subportfolio_information(subset, industry, config))
            
    industry_data = pd.DataFrame(rows).set_index(["date", "industry_name"])

    return industry_data


def drop_small_portfolios(
    portfolio_df: pd.DataFrame, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to drop the portfolios with less firms than the minimum required.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        DataFrame containing the portfolio information.
    config : CONFIGURATION_CLASS
        Configuration of the project.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the significant portfolios.
    """
    min_firms_per_portfolio: int = config.CUTOFF_FIRMS_PER_PORTFOLIO

    portfolio_df = portfolio_df.reset_index()

    num_portfolios_start: int = portfolio_df["industry_name"].nunique()

    result: pd.DataFrame = portfolio_df[
        portfolio_df["num_firms"] >= min_firms_per_portfolio
    ]

    num_portfolios_end: int = result["industry_name"].nunique()

    if config.LOG_INFO:
        config.logger.info(
            f"Dropped non-significant portfolios with less than {min_firms_per_portfolio} firms.\
            \nNumber of portfolios dropped: {num_portfolios_start - num_portfolios_end} ({num_portfolios_start}->{num_portfolios_end})"
        )

    return result.set_index(["date", "industry_name"])


def drop_sparse_portfolios(
    portfolio_df: pd.DataFrame, config: CONFIGURATION_CLASS
) -> pd.DataFrame:
    """
    Function to drop portfolios that do not appear often enough in the data, 
    meaning that they do not have enough returns data to be included in the analysis.
    
    Parameters
    ----------
    portfolio_df : pd.DataFrame
        DataFrame containing the portfolio information.
    config : CONFIGURATION_CLASS
        Configuration of the project.
        
    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the portfolios that appear in at least a minimum number of periods.
     """
     
    min_count_occurances: int = config.MIN_OCCURANCES_PORTFOLIOS 

    portfolio_df = portfolio_df.reset_index()

    num_portfolios_start: int = portfolio_df["industry_name"].nunique()

    counts: pd.Series = portfolio_df["industry_name"].value_counts()
    result = portfolio_df[portfolio_df["industry_name"].isin(counts[counts >= min_count_occurances].index)]

    num_portfolios_end: int = result["industry_name"].nunique()

    if config.LOG_INFO:
        config.logger.info(
            f"Dropped non-significant portfolios with less than {min_count_occurances} occurances.\
            \nNumber of portfolios dropped: {num_portfolios_start - num_portfolios_end} ({num_portfolios_start}->{num_portfolios_end})"
        )

    return result.set_index(["date", "industry_name"])



def _calculate_weight_in_portfolio_marketcap(firm_subset: pd.DataFrame) -> pd.Series:
    """
    Function to compute weights of firms in a portfolio based on their market capitalization.
    This is the one period lagged market cap weight, meaning that the market cap used to compute the weights is from the previous period to avoid look-ahead bias.

    Parameters
    ----------
    firm_subset : pd.DataFrame
        DataFrame containing the information of the firms in the portfolio, including their market cap and date.

    Returns
    -------
    pd.Series
        A Series containing the weight of each firm in the portfolio based on their market capitalization.
    """
    # Compute the lagged Market Cap
    #firm_subset["Lagged_MarketCap"] = firm_subset.groupby("gvkey")["market_cap"].shift(
        #1
    #)

    # Compute the weight as the MarketCap weight in the portfolio
    weights: pd.Series = firm_subset["Lagged_MarketCap"] / firm_subset.groupby("date")[
        "Lagged_MarketCap"
    ].transform("sum")
    return weights


def _calculate_weight_in_portfolio_equal(firm_subset: pd.DataFrame) -> pd.Series:
    """
    Function to calculate the equal weight of firms in a portfolio.
    This is, all firms have the same weight.

    Parameters
    ----------
    firm_subset : pd.DataFrame
        DataFrame containing the information of the firms in the portfolio.

    Returns
    -------
    pd.Series
        A Series containing the equal weight of each firm in the portfolio."""

    # Get the number of firms in the portfolio on each date
    num_firms: pd.Series = firm_subset.groupby("date")["gvkey"].transform("nunique")

    weights: pd.Series = 1.0 / num_firms

    return weights


def _calculate_weight_in_portfolio(
    firm_subset: pd.DataFrame, config=CONFIGURATION_CLASS
) -> pd.Series:
    """
    Function to calculate the weight of different firms in a portfolio according to the configuration.

    Parameters
    ----------
    firm_subset : pd.DataFrame
        DataFrame containing the information of the firms in the portfolio.
    config : CONFIGURATION_CLASS
        Configuration of the project.

    Returns
    -------
    pd.Series
        A Series containing the weight of each firm in the portfolio.
    """
    # Unpack the config
    how: Literal["MarketCap", "Equal"] = config.PORTFOLIO_AGGREGATION_METHOD

    if how == "MarketCap":
        return _calculate_weight_in_portfolio_marketcap(firm_subset)

    elif how == "Equal":
        return _calculate_weight_in_portfolio_equal(firm_subset)
    else:
        raise ValueError("Invalid aggregation method.")


def create_portfolios(
    sic_descr: pd.DataFrame,
    ff_industry_portfolios: pd.DataFrame,
    entries: pd.DataFrame,
    config: CONFIGURATION_CLASS,
) -> pd.DataFrame:
    """
    Function to orchestrate the creation and formatting of industry and marketcap portfolios.

    Parameters
    ----------
    sic_descr : pd.DataFrame
        DataFrame containing SIC code descriptions.
    ff_industry_portfolios : pd.DataFrame
        DataFrame containing Fama-French industry classifications for each SIC-code.
    entries: pd.DataFrame
        DataFrame containing the market entries of the firms to include in the index.
    config : CONFIGURATION_CLASS
        Configuration of the project.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the final portfolios with their industry classification and market cap based sub-portfolios.
        These are generated for each period and might differ.
    """

    # Assign each firm to an industry
    industry_assignment_per_period: pd.DataFrame = assign_industry(
        entries=entries,
        sic_descr=sic_descr,
        ff_industry_portfolios=ff_industry_portfolios,
        config=config,
    )


    portfolios: pd.DataFrame = create_all_portfolios(industry_assignment_per_period, config)

    # Drop non-significant portfolios
    industry_marketcap_portfolios_filtered: pd.DataFrame = drop_small_portfolios(
        portfolio_df=portfolios, config=config
    )

    industry_marketcap_portfolios_filtered = drop_sparse_portfolios(
        portfolio_df=industry_marketcap_portfolios_filtered, config=config
    )
    return industry_marketcap_portfolios_filtered


# Save the results

# Main pipeline function
def create_portfolios_and_returns(
    config: CONFIGURATION_CLASS,
) -> pd.DataFrame:
    """
    Main function to orchestrate the creation of portfolios based on SIC codes and market capitalization.
    This function acts as a pipeline and calls all other helper functions defined above.

    Parameters
    ----------
    config : CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    pd.DataFrame
        DataFrame containing the info and returns per portfolios
    """
    # Dowload the data
    data: DATAFRAME_CONTAINER = download_processed_data(config)
    if not isinstance(data.monthly_inflation, pd.DataFrame):
        raise ValueError("Inflation data should be a DataFrame.")

    if config.LOG_INFO:
        config.logger.info("Starting to create portfolios....\n" + "-" * 80)


    stock_market_merged_info: pd.DataFrame = (data.monthly_stock_info
        .reset_index()
        .merge(
            data.firm_info[["gvkey", "siccode", "companyname"]],
            on="gvkey",
            how="left",)
        .set_index("date")
        .dropna())

    # Compute the market cap
    monthly_stock_info_marketcap: pd.DataFrame = get_market_cap(
        stock_prices=stock_market_merged_info,
        inflation=data.monthly_inflation,
        config=config
    )

    # Cutoff the firms based on their market cap
    stock_entries_to_keep: pd.DataFrame = apply_marketcap_cutoff(
        stock_prices=monthly_stock_info_marketcap,
        inflation=data.monthly_inflation,
        config=config,
    )

    # Create the portfolios
    portfolios: pd.DataFrame = create_portfolios(
        sic_descr=data.sic_info,
        ff_industry_portfolios=data.ff_industry_portfolios,
        entries=stock_entries_to_keep,
        config=config,
    )

    if config.LOG_INFO:
        config.logger.info("Successfully created portfolios")

    return portfolios


def create_save_portfolios(config: CONFIGURATION_CLASS) -> None:
    """
    Function to run the entire portfolio creation and return calculation pipeline.

    Parameters
    ----------
    config : CONFIGURATION_CLASS
        Configuration of the project.

    Returns
    -------
    None
        This function runs the pipeline and saves the results to CSV files.
    """
    # Create the portfolios and their returns
    portfolios = create_portfolios_and_returns(config)

    # Save the results
    save_portfolio_returns_constitution(
        portfolios = portfolios,
        config=config,
    )


if __name__ == "__main__":
    create_save_portfolios(PROJ_CONFIG)
