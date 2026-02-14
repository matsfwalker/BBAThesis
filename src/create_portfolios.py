from typing import Any, List, Literal, Tuple

import numpy as np
import pandas as pd

from configs import CONFIG, CONFIGURATION, FILENAMES, DATAFRAME_CONTAINER


# utils function to discount values based on inflation
def _inflation_discount(inflation_info: pd.DataFrame, value: float) -> pd.Series:
    """
    Function to discount a given value based on the inflation discount multiples.

    Parameters
    ----------
    inflation_info : pd.DataFrame
        DataFrame containing the inflation discount multiples with a datetime index.
    value : float
        The value to be discounted.

    Returns
    -------
    pd.Series
        A Series containing the discounted values for each date.
    """
    inflation_info = inflation_info.copy()
    return inflation_info["Inflation multiple"] * value


def _present_value_inflation(
    inflation_info: pd.DataFrame, values: pd.Series
) -> pd.Series:
    """
    Function to adjust past prices to their present value based on the inflation discount multiples.

    Parameters
    ----------

    inflation_info : pd.DataFrame
        DataFrame containing the inflation discount multiples with a datetime index.
    values : pd.Series
        A Series containing the values to be adjusted, indexed by date.

    Returns
    -------
    pd.Series
        A Series containing the inflation-adjusted values for each date.
    """
    inflation_info = inflation_info.copy()
    values = values.copy()
    inflation_info = inflation_info.reindex(values.index).ffill()
    return values / inflation_info["Inflation multiple"]


# Import the data
def download_processed_data(
    config: CONFIGURATION,
) -> DATAFRAME_CONTAINER:
    """
    Function to read the stock prices, firm info, and SIC code descriptions from the processed data directory.

    Parameters
    ----------
    config : CONFIGURATION
        Configuration of the model

    Returns
    -------
    DATAFRAME_CONTAINER
        A container for the dataframes.
    """

    if config.LOG_INFO:
        config.logger.info("Starting to download processed data...\n" + "-" * 80)

    stock_prices: pd.DataFrame = pd.read_csv(
        config.paths.processed_read(FILENAMES.Stock_prices),
        parse_dates=["date"],
        index_col="date",
    )

    firm_info: pd.DataFrame = pd.read_csv(
        config.paths.processed_read(FILENAMES.Firm_info)
    )

    sic_codes: pd.DataFrame = pd.read_csv(
        config.paths.processed_read(FILENAMES.Sic_description)
    )

    inflation: pd.DataFrame = pd.read_csv(
        config.paths.processed_read(FILENAMES.Inflation_info_monthly),
        parse_dates=["date"],
        index_col="date",
    )

    ff_industry_portfolios: pd.DataFrame = pd.read_csv(
        config.paths.processed_read(FILENAMES.FF5_industry_portfolios)
    )

    if config.LOG_INFO:
        config.logger.info("Successfully downloaded the processed data")

    return DATAFRAME_CONTAINER(
        stock_market_info=stock_prices,
        firm_info=firm_info,
        sic_info=sic_codes,
        monthly_inflation=inflation,
        ff_industry_portfolios=ff_industry_portfolios,
    )


# Compute MarketCap
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


def get_market_cap(stock_prices: pd.DataFrame, inflation: pd.DataFrame, config: CONFIGURATION) -> pd.DataFrame:
    """
    Function to add the market cap and present value market cap of firms to the dataframe.

    Parameters
    ----------

    stock_prices : pd.DataFrame
        DataFrame containing stock prices with a datetime index.
    inflation : pd.DataFrame
        DataFrame containing the inflation discount multiples with a datetime index.
    config : CONFIGURATION
        Configuration of the project.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the original stock prices information along with the market cap and present value market cap (if applicable) of each firm.
    """
    stock_prices = stock_prices.copy()
    stock_prices["market_cap"] = _compute_market_cap(stock_prices, "close", "sharesoutstanding")

    if config.DISCOUNT_MARKETCAP_FIRM_INFLATION:
        stock_prices["market_cap_present_value"] = _present_value_inflation(
            inflation_info=inflation, values=stock_prices["market_cap"]
        )

    return stock_prices


# MarketCap cutoff
def _apply_marketcap_cutoff_latestperiod(
    stock_prices: pd.DataFrame, config: CONFIGURATION
) -> pd.DataFrame:
    """
    Function to apply a market cap cutoff to the latest stock prices.
    Parameters
    ----------
    stock_prices : pd.DataFrame
        DataFrame containing stock prices with a datetime index.
    config : CONFIGURATION
        Configuration of the project.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the firms that meet the market cap cutoff at the latest date.
    """
    # Unpack the config:
    marketcap_cutoff: float = config.MIN_MARKETCAP_FIRM

    num_firms_start: int = stock_prices['gvkey'].nunique()

    # Get the latest date and corresponding entries
    latest_date: pd.Timestamp = stock_prices.index.max()
    latest_prices: pd.DataFrame = (
        stock_prices.loc[latest_date].reset_index().drop_duplicates(subset=["gvkey"])
    )

    result: pd.DataFrame = latest_prices[
        latest_prices["market_cap"] >= marketcap_cutoff
    ]

    num_firms_end: int = result['gvkey'].nunique()

    if config.LOG_INFO:
        config.logger.info(
            f"Applied market cap cutoff of {marketcap_cutoff} to the latest period ({latest_date.date()}).\
            \nNumber of firms dropped: {num_firms_start - num_firms_end} ({num_firms_start}->{num_firms_end})"
        )

    return result


def _apply_marketcap_cutoff_allperiods(
    stock_prices: pd.DataFrame, inflation: pd.DataFrame, config: CONFIGURATION
) -> pd.DataFrame:
    """
    Function to apply a market cap cutoff to the stock prices of all periods, discounting the market cap by inflation.
    Parameters
    ----------
    stock_prices : pd.DataFrame
        DataFrame containing stock prices with a datetime index.
    inflation : pd.DataFrame
        DataFrame containing inflation discount multiples with a datetime index.
    config : CONFIGURATION
        Configuration of the project.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the firms that meet the market cap cutoff in each period after discounting for inflation.
    """
    # Unpack the config:
    min_marketcap: float = config.MIN_MARKETCAP_FIRM

    num_firms_start: int = stock_prices['gvkey'].nunique()

    # Discount the market cap by inflation
    min_marketcap_discounted: pd.Series = _inflation_discount(
        inflation_info=inflation, value=min_marketcap
    )

    # Ensure alignment
    min_marketcap_discounted = min_marketcap_discounted.reindex(stock_prices.index)

    mask = stock_prices["market_cap"] >= min_marketcap_discounted
    survivors = mask.groupby(stock_prices["gvkey"]).all()
    filtered_stock_prices = stock_prices[
        stock_prices["gvkey"].isin(survivors[survivors].index)
    ]

    # Order the filtered stock prices by date and keep only the latest entry for each gvkey
    filtered_stock_prices_lastval = (
        filtered_stock_prices.sort_index()
        .drop_duplicates(subset=["gvkey"], keep="last")
        .reset_index()
    )

    num_firms_end: int = filtered_stock_prices_lastval['gvkey'].nunique()

    if config.LOG_INFO:
        config.logger.info(
            f"Applied market cap cutoff of {min_marketcap} to all periods using inflation discounted values.\
            \nNumber of firms dropped: {num_firms_start - num_firms_end} ({num_firms_start}->{num_firms_end})"
        )

    return filtered_stock_prices_lastval


def apply_marketcap_cutoff(
    stock_prices: pd.DataFrame, inflation: pd.DataFrame, config: CONFIGURATION
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
    config : CONFIGURATION
        Configuration of the project.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the firms that meet the market cap cutoff according to the configuration.
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


# Create Industry Portfolios
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
    firms_df: pd.DataFrame, level: Literal[1, 2, 3, 4], sic_col: str = "siccode"
) -> pd.DataFrame:
    """
    Helper function to format firms' SIC codes to a given level.

    Parameters
    ----------
    firms_df : pd.DataFrame
        DataFrame containing firm information including SIC codes.
    level : Literal[1,2,3,4]
        The desired SIC code level (1 to 4).
    sic_col : str = "siccode"
        The name of the column containing SIC codes.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing firms with SIC codes normalized to the specified level.
    """
    # Copy and format the firms
    firms: pd.DataFrame = firms_df.copy()
    firms.dropna(subset=[sic_col, "gvkey"], inplace=True)

    # Normalise the sic code using floor division
    firms["sic_level"] = np.floor(firms[sic_col] / (10 ** (4 - level))) * (
        10 ** (4 - level)
    )
    firms["sic_level"] = firms["sic_level"].astype("Int64")

    return firms


def _assign_industry_to_firms_siclevel(
    firm_descr: pd.DataFrame, sic_descr: pd.DataFrame, config: CONFIGURATION
) -> pd.DataFrame:
    """
    Function to assign the firms to a specific industry by sic code level.
    This is done using their sic codes and a sic code description dataframe.
    The firms can be grouped according to different levels of the sic code hierarchy.

    Parameters
    ----------
    firm_descr : pd.DataFrame
        Dataframe containing firm information including sic codes.
    sic_descr : pd.DataFrame
        Dataframe containing sic code descriptions.
    config : CONFIGURATION
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
    firms: pd.DataFrame = _format_firms_sic(firm_descr, sic_level)
    descr: pd.DataFrame = _format_sic_codes(sic_descr, sic_level)

    # Merge the two dataframes
    merged = firms.merge(descr, on="sic_level", how="left")

    result: pd.DataFrame = merged[["gvkey", "sic_level", "sicdescription"]].rename(
        columns={"sic_level": "key", "sicdescription": "industry_name"}
    )

    if config.LOG_INFO:
        config.logger.info(
            f"Successfully assigned an industry to firms using SIC codes at level {sic_level}"
        )

    return result


def _assign_industry_firms_ffindustries(
    firm_descr: pd.DataFrame, ff_industries: pd.DataFrame, config: CONFIGURATION
) -> pd.DataFrame:
    """
    Function to assign to each firm its industry based on the Fama-French industry classification.

    Parameters
    ----------
    firm_descr : pd.DataFrame
        DataFrame containing firm information including SIC codes.
    ff_industries : pd.DataFrame
        DataFrame containing Fama-French industry classifications for each SIC-code.
    config : CONFIGURATION
        Configuration of the project.

    Returns
    -------
    pd.DataFrame
        DataFrame with the gvkey of the firms and their corresponding Fama-French industry classification.
    """

    # Merge the two dataframes
    merged = firm_descr.merge(ff_industries, on="siccode", how="right")

    result: pd.DataFrame = merged[["gvkey", "industry_id", "industry_name"]].rename(
        columns={"industry_id": "key"}
    )

    if config.LOG_INFO:
        config.logger.info(
            "Successfully assigned industries to firms according to the Fama-French industry portfolios"
        )

    return result


def assign_industry_to_firms(
    firm_descr: pd.DataFrame,
    sic_descr: pd.DataFrame,
    ff_industry_portfolios: pd.DataFrame,
    config=CONFIGURATION,
) -> pd.DataFrame:
    """
    Function to assign an industry to each firm based on the configuration.
    This ca either be done according to the SIC code level or the Fama-French industry classification.

    Parameters
    ----------
    firm_descr : pd.DataFrame
        DataFrame containing firm information including SIC codes.
    sic_descr : pd.DataFrame
        DataFrame containing SIC code descriptions.
    ff_industry_portfolios : pd.DataFrame
        DataFrame containing Fama-French industry classifications for each SIC-code.
    config : CONFIGURATION
        Configuration of the project.

    Returns
    -------
    pd.DataFrame
        DataFrame with the gvkey of the firms and their corresponding industry classification according to the configuration.
    """

    if config.INDUSTRY_CLASSIFICATION_METHOD == "Sic_level":
        return _assign_industry_to_firms_siclevel(firm_descr, sic_descr, config)
    else:
        return _assign_industry_firms_ffindustries(
            firm_descr, ff_industry_portfolios, config
        )


# Portfolio formatting
def intersect_portfolios_price_companyinfo(
    firm_industry_assignment: pd.DataFrame,
    firms_to_keep: pd.DataFrame,
    firm_descr: pd.DataFrame,
) -> pd.DataFrame:
    """
    Function to intersect the portfolios dataframe with the firm information dataframe.

    Parameters
    ----------
    firm_industry_assignment : pd.DataFrame
        DataFrame containing information of which industry to assign a firm to.
    firms_to_keep: pd.DataFrame
        Dataframe containing the prices info
    firm_descr : pd.DataFrame
        DataFrame containing firm information.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the firms that are present in both firms_to_keep and firm information dataframes.
    """
    final_portfolio: pd.DataFrame = firm_industry_assignment.merge(
        firms_to_keep, on="gvkey", how="inner"
    ).merge(firm_descr[["gvkey", "companyname"]], on="gvkey", how="left")
    return final_portfolio


def get_portfolio_constitution(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to create a dataframe to keep track of the portfolio constitution.
    Groups these together and sorts them by industry total market cap and individual market cap.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        DataFrame containing portfolio information.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the portfolio constitution details.
    """
    # Create a new dataframe that tracks the constitution of the portfolio
    final_portfolio_constitution: pd.DataFrame = portfolio_df.drop_duplicates(
        subset="gvkey", keep="last"
    ).set_index(["industry_name", "gvkey"])[
        ["companyname", "market_cap", "close", "sharesoutstanding", "key", "date", "market_cap_present_value"]
    ]

    group_totals: pd.Series = final_portfolio_constitution.groupby(level=0)[
        "market_cap"
    ].sum()
    group_key: pd.Index = final_portfolio_constitution.index.get_level_values(0).map(
        group_totals.to_dict()
    )

    sorted_main_portfolios = (
        final_portfolio_constitution.assign(_group_total_mcap=group_key)
        .sort_values(
            by=["_group_total_mcap", "market_cap"],
            ascending=[False, False],
            kind="mergesort",
        )
        .drop(columns="_group_total_mcap")
    )

    return sorted_main_portfolios


# Get market cap based sub-portfolios
def compute_marketcap_portfolios(
    main_portfolios: pd.DataFrame, config: CONFIGURATION
) -> pd.DataFrame:
    """
    Function to create sub-portfolios based on market capitalization within each industry.
    Creates a large-cap and a small-cap portfolio for each industry.

    Parameters
    ----------
    main_portfolios : pd.DataFrame
        DataFrame containing the main portfolio information.
    config : CONFIGURATION
        Configuration of the project.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the sub-portfolios based on market capitalization."""
    num_firms_portfolio = config.MARKETCAP_PORTFOLIO_NUMBER_FIRMS
    percentile_portfolio = config.MARKETCAP_PORTFOLIO_PERCENTILE

    df = main_portfolios.copy()
    df["MarketCapID"] = "all"

    new_blocks = []

    industries = df.index.get_level_values(0).unique()

    for industry in industries:
        industry_df = df.xs(industry, level=0)

        # determine cutoff
        if percentile_portfolio is not None:
            cutoff = int(len(industry_df) * percentile_portfolio)
        elif num_firms_portfolio is not None:
            cutoff = min(num_firms_portfolio, len(industry_df) // 2)
        else:
            raise ValueError(
                "Either percentile or number of firms for portfolios needs to be provided."
            )
        if cutoff == 0:
            continue

        top_firms = industry_df.iloc[:cutoff].copy()
        bottom_firms = industry_df.iloc[-cutoff:].copy()

        top_firms["MarketCapID"] = "large_cap"
        bottom_firms["MarketCapID"] = "small_cap"

        # assign new level-0 labels
        top_firms.index = pd.MultiIndex.from_product(
            [[f"{industry} - Large Cap"], top_firms.index], names=df.index.names
        )

        bottom_firms.index = pd.MultiIndex.from_product(
            [[f"{industry} - Small Cap"], bottom_firms.index], names=df.index.names
        )

        new_blocks.extend([top_firms, bottom_firms])

    if new_blocks:
        df = pd.concat([df, *new_blocks])

    return df.reset_index()


def aggregate_portfolio(
    portfolio: pd.DataFrame, config: CONFIGURATION
) -> pd.DataFrame:
    """
    Function to aggregate the portfolio by SIC description and sort them.

    Parameters
    ----------
    portfolio : pd.DataFrame
        DataFrame containing the portfolio information.
    config: CONFIGURATION
        COnfiguration of the project

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the aggregated and sorted SIC portfolios.
    """
    sic_portfolios: pd.DataFrame = (
        portfolio.groupby("industry_name")
        .agg(
            key=("key", "first"),
            num_firms=("gvkey", "nunique"),
            total_market_cap=("market_cap", "sum"),
            total_presentvalue_marketcap=("market_cap_present_value", "sum"), 
            gvkeys=("gvkey", lambda x: list(x)),
            marketcap_id=("MarketCapID", "first"),
        )
        .reset_index()
    )

    if config.LOG_INFO:
        config.logger.info("Successfully aggregated the portfolios by SIC description")

    return sic_portfolios


def drop_small_portfolios(
    portfolio_df: pd.DataFrame, config: CONFIGURATION
) -> pd.DataFrame:
    """
    Function to drop the portfolios with less firms than the minimum required.

    Parameters
    ----------
    portfolio_df : pd.DataFrame
        DataFrame containing the portfolio information.
    config : CONFIGURATION
        Configuration of the project.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the significant portfolios.
    """
    min_firms_per_portfolio: int = config.CUTOFF_FIRMS_PER_PORTFOLIO

    num_portfolios_start: int = portfolio_df['industry_name'].nunique()

    result: pd.DataFrame = portfolio_df[portfolio_df["num_firms"] >= min_firms_per_portfolio]

    num_portfolios_end: int = result['industry_name'].nunique()

    if config.LOG_INFO:
        config.logger.info(
            f"Dropped non-significant portfolios with less than {min_firms_per_portfolio} firms.\
            \nNumber of portfolios dropped: {num_portfolios_start - num_portfolios_end} ({num_portfolios_start}->{num_portfolios_end})"
        )

    return result


# Get the returns from portfolios
def get_returns_stocks(stock_prices: pd.DataFrame) -> pd.DataFrame:
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
    returns: pd.DataFrame = sorted_prices.set_index("date").sort_index()

    return returns


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
        A Series containing the weight of each firm in the portfolio based on their market capitalization."""
    # Compute the lagged Market Cap
    firm_subset["Lagged_MarketCap"] = (
        firm_subset.groupby("gvkey")["market_cap"].shift(1)
    )

    # Compute the weight as the MarketCap weight in the portfolio
    weights: pd.Series = firm_subset[
        "Lagged_MarketCap"
    ] / firm_subset.groupby("date")["Lagged_MarketCap"].transform(
        "sum"
    )
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
    num_firms: pd.Series = firm_subset.groupby("date")[
        "gvkey"
    ].transform("nunique")

    weights: pd.Series = 1.0 / num_firms

    return weights


def _calculate_weight_in_portfolio(firm_subset: pd.DataFrame, config=CONFIGURATION)->pd.Series:
    """
    Function to calculate the weight of different firms in a portfolio according to the configuration.
    
    Parameters
    ----------
    firm_subset : pd.DataFrame
        DataFrame containing the information of the firms in the portfolio.
    config : CONFIGURATION
        Configuration of the project.
    
    Returns
    -------
    pd.Series
        A Series containing the weight of each firm in the portfolio.
    """
    # Unpack the config
    how: Literal["MarketCap", "Equal"] = config.PORTFOLIO_AGGREGATION_METHOD

    if how == "MarketCap":
        weights: pd.Series = _calculate_weight_in_portfolio_marketcap(firm_subset)

    elif how == "Equal":
        weights: pd.Series = _calculate_weight_in_portfolio_equal(firm_subset)
    else:
        raise ValueError("Invalid aggregation method.")
    
    return weights

def calculate_portfolio_returns(
    portfolios_df: pd.DataFrame, prices_df: pd.DataFrame, config: CONFIGURATION
) -> pd.DataFrame:
    """
    Function to calculate the returns of the different portfolios.

    Parameters
    ----------
    portfolios_df : pd.DataFrame
        DataFrame containing the portfolio information.
    prices_df : pd.DataFrame
        DataFrame containing stock prices with a datetime index.
    config : CONFIGURATION
        Configuration of the project.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the returns of the portfolios.
    """

    # Get the stock returns and format the portfolios
    prices_and_returns: pd.DataFrame = get_returns_stocks(prices_df)
    portfolios_df = portfolios_df.reset_index()

    # Initialize the dataframe to store portfolio prices
    col_names: List[str] = portfolios_df["industry_name"].drop_duplicates().tolist()
    portfolio_returns = pd.DataFrame(columns=col_names)

    for _, row in portfolios_df.iterrows():
        # Get the prices and returns for the gvkeys in the portfolio
        gvkeys: List[Any] = row["gvkeys"]
        prices_and_returns_subset: pd.DataFrame = prices_and_returns[
            prices_and_returns["gvkey"].isin(gvkeys)
        ].copy()

        # Calculate the weights per stock
        prices_and_returns_subset["Weight"] = _calculate_weight_in_portfolio(firm_subset=prices_and_returns_subset, config=config)

        # Multiply the returns by the weights and sum them to get the portfolio return
        portfolio_returns[row["industry_name"]] = (
            (prices_and_returns_subset["Weight"] * prices_and_returns_subset["return"])
            .groupby("date")
            .sum()
        )

    # Format the portfolio returns dataframe
    portfolio_returns_formated: pd.DataFrame = portfolio_returns.rename_axis(
        "date"
    ).dropna(how="all")

    if config.LOG_INFO:
        config.logger.info("Successfully calculated the portfolio returns")

    return portfolio_returns_formated


def create_portfolios(
    firm_descr: pd.DataFrame,
    sic_descr: pd.DataFrame,
    ff_industry_portfolios: pd.DataFrame,
    firms_to_keep: pd.DataFrame,
    config: CONFIGURATION
) -> pd.DataFrame:
    """
    Function to orchestrate the creation and formatting of industry and marketcap portfolios.
    
    Parameters
    ----------
    firm_descr : pd.DataFrame
        DataFrame containing firm information including SIC codes.
    sic_descr : pd.DataFrame
        DataFrame containing SIC code descriptions.
    ff_industry_portfolios : pd.DataFrame
        DataFrame containing Fama-French industry classifications for each SIC-code.
    firms_to_keep: pd.DataFrame
        DataFrame containing the firms that passed the market cap cutoff.
    config : CONFIGURATION
        Configuration of the project.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the final portfolios with their industry classification and market cap based sub-portfolios."""
    
    # Assign each firm to an industry
    firm_industry_assignment: pd.DataFrame = assign_industry_to_firms(
        firm_descr=firm_descr,
        sic_descr=sic_descr,
        ff_industry_portfolios=ff_industry_portfolios,
        config=config,
    )

    # Add the information about firms and their stock prices
    industry_assignment_with_info: pd.DataFrame = (
        intersect_portfolios_price_companyinfo(
            firm_industry_assignment=firm_industry_assignment,
            firms_to_keep=firms_to_keep,
            firm_descr=firm_descr,
        )
    )

    # Get the constitution of the portfolios
    industry_portfolio_constitution: pd.DataFrame = get_portfolio_constitution(
        portfolio_df=industry_assignment_with_info
    )

    # Add the sub-portfolios by market cap
    industry_marketcap_portfolios: pd.DataFrame = compute_marketcap_portfolios(
        main_portfolios=industry_portfolio_constitution, config=config
    )

    # Aggregate the portfolios
    industry_marketcap_portfolios_agg: pd.DataFrame = aggregate_portfolio(
        portfolio=industry_marketcap_portfolios, config=config
    )

    # Drop non-significant portfolios
    industry_marketcap_portfolios_filtered: pd.DataFrame = drop_small_portfolios(
        portfolio_df=industry_marketcap_portfolios_agg, config=config
    )

    return industry_marketcap_portfolios_filtered

# Save the results
def save_portfolio_returns_constitution(
    portfolio_returns: pd.DataFrame,
    portfolio_constitution: pd.DataFrame,
    config: CONFIGURATION,
) -> None:
    """
    Function to save the portfolio returns and constitution details to CSV files.

    Parameters
    ----------
    portfolio_returns : pd.DataFrame
        DataFrame containing the returns of the portfolios.
    portfolio_constitution : pd.DataFrame
        DataFrame containing the constitution details of the portfolios.
    config : CONFIGURATION
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
    portfolio_returns.to_csv(config.paths.portfolios_out(FILENAMES.Portfolio_returns))
    portfolio_constitution.to_csv(
        config.paths.portfolios_out(FILENAMES.Portfolio_construction_details)
    )

    if config.LOG_INFO:
        config.logger.info(
            "Successfully saved the portfolio returns and constitution details"
        )

    return


# Main pipeline function
def create_portfolios_and_returns(
    config: CONFIGURATION,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to orchestrate the creation of portfolios based on SIC codes and market capitalization.
    This function acts as a pipeline and calls all other helper functions defined above.

    Parameters
    ----------
    config : CONFIGURATION
        Configuration of the project

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Two pandas dataframes
        One for the returns of the different portfolios
        One for their constitution details
    """
    # Dowload the data
    data: DATAFRAME_CONTAINER = download_processed_data(config)
    if not isinstance(data.monthly_inflation, pd.DataFrame):
        raise ValueError("Inflation data should be a DataFrame.")

    if config.LOG_INFO:
        config.logger.info("Starting to create portfolios....\n" + "-" * 80)

    # Compute the market cap
    stock_market_info_marketcap: pd.DataFrame = get_market_cap(
        stock_prices=data.stock_market_info, inflation=data.monthly_inflation,config=config
    )

    # Cutoff the firms based on their market cap
    firms_to_keep: pd.DataFrame = apply_marketcap_cutoff(
        stock_prices=stock_market_info_marketcap,
        inflation=data.monthly_inflation,
        config=config,
    )

    # Create the portfolios
    portfolios: pd.DataFrame = create_portfolios(
        firm_descr=data.firm_info,
        sic_descr=data.sic_info,
        ff_industry_portfolios=data.ff_industry_portfolios,
        firms_to_keep=firms_to_keep,
        config=config
    )

    # Calculate the returns of the portfolios from their constituents
    industry_marketcap_portfolio_returns: pd.DataFrame = calculate_portfolio_returns(
        portfolios_df=portfolios,
        prices_df=stock_market_info_marketcap,
        config=config,
    )

    if config.LOG_INFO:
        config.logger.info("Successfully created portfolios")

    return industry_marketcap_portfolio_returns, portfolios


def create_save_portfolios(config: CONFIGURATION) -> None:
    """
    Function to run the entire portfolio creation and return calculation pipeline.

    Parameters
    ----------
    config : CONFIGURATION
        Configuration of the project.

    Returns
    -------
    None
        This function runs the pipeline and saves the results to CSV files.
    """
    # Create the portfolios and their returns
    portfolio_returns, portfolio_constitution = create_portfolios_and_returns(config)

    # Save the results
    save_portfolio_returns_constitution(
        portfolio_returns=portfolio_returns,
        portfolio_constitution=portfolio_constitution,
        config=config,
    )


if __name__ == "__main__":
    create_save_portfolios(CONFIG)
