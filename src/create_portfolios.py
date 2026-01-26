import pandas as pd
import numpy as np
from typing import Tuple, List, Any, Literal
from configs import CONFIG, CONFIGURATION, FILENAMES


# Import the data
def get_stockprices_firminfo_siccodes(
    config: CONFIGURATION,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function to read the stock prices, firm info, and SIC code descriptions from the processed data directory.

    Parameters
    ----------
    config : CONFIGURATION
        Configuration of the model
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A tuple containing three DataFrames: stock prices, firm info, and SIC code descriptions.
    """

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

    return stock_prices, firm_info, sic_codes


# Compute and cutoff MarketCap
def compute_market_cap(
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


def apply_cutoff_latest_marketcap(
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

    # Get the latest date and corresponding entries
    latest_date: pd.Timestamp = stock_prices.index.max()
    latest_prices: pd.DataFrame = (
        stock_prices.loc[latest_date].reset_index().drop_duplicates(subset=["gvkey"])
    )

    return latest_prices[latest_prices["market_cap"] >= marketcap_cutoff]


# Create Industry Portfolios from SIC-Codes
def format_sic_codes(
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


def format_firms_sic(
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


def compute_industry_portfolios_sic(
    firm_descr: pd.DataFrame, sic_descr: pd.DataFrame, config: CONFIGURATION
) -> pd.DataFrame:
    """
    Function to group the firms by industry.
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
        Dataframe with firms grouped by the specified sic code level.
    """

    # Unpack the config:
    sic_level: Literal[1,2,3,4] = config.SIC_LEVEL

    # Format firms and sic codes according to the level
    firms: pd.DataFrame = format_firms_sic(firm_descr, sic_level)
    descr: pd.DataFrame = format_sic_codes(sic_descr, sic_level)

    # Merge the two dataframes
    merged = firms.merge(descr, on="sic_level", how="left")

    return merged[["gvkey", "sic_level", "sicdescription"]]


# Portfolio formatting
def intersect_portfolios_price_companyinfo(
    portfolios_df: pd.DataFrame, price_df: pd.DataFrame, firm_info_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Function to intersect the portfolios dataframe with the firm information dataframe.

    Parameters
    ----------
    portfolios_df : pd.DataFrame
        DataFrame containing portfolio information.
    prices_df: pd.DataFrame
        Dataframe containing the prices info
    firm_info_df : pd.DataFrame
        DataFrame containing firm information.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing only the firms that are present in both the portfolios and firm information dataframes.
    """
    final_portfolio: pd.DataFrame = portfolios_df.merge(
        price_df, on="gvkey", how="inner"
    ).merge(firm_info_df[["gvkey", "companyname"]], on="gvkey", how="left")
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
    ).set_index(["sicdescription", "gvkey"])[
        ["companyname", "market_cap", "close", "sharesoutstanding", "sic_level", "date"]
    ]

    group_totals: pd.Series = final_portfolio_constitution.groupby(level=0)[
        "market_cap"
    ].sum()
    group_key: pd.Series = final_portfolio_constitution.index.get_level_values(0).map(
        group_totals
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
            raise ValueError("Either percentile or number of firms for portfolios needs to be provided.")
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


def aggregate_sicportfolio(
    portfolio: pd.DataFrame,
) -> pd.DataFrame:
    """
    Function to aggregate the portfolio by SIC description and sort them.
    Drops all portfolios with less than the minimum number of firms.

    Parameters
    ----------
    portfolio : pd.DataFrame
        DataFrame containing the portfolio information.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the aggregated and sorted SIC portfolios.
    """
    sic_portfolios: pd.DataFrame = (
        portfolio.groupby("sicdescription")
        .agg(
            sic_level=("sic_level", "first"),
            num_firms=("gvkey", "nunique"),
            total_market_cap=("market_cap", "sum"),
            gvkeys=("gvkey", lambda x: list(x)),
            marketcap_id=("MarketCapID", "first"),
        )
        .reset_index()
    )

    return sic_portfolios


def drop_nonsignicant_portfolios(
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
    min_firms: int = config.CUTOFF_FIRMS_PER_PORTFOLIO

    return portfolio_df[portfolio_df["num_firms"] >= min_firms]


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

    # Unpack the config
    how: Literal["MarketCap", "Equal"] = config.PORTFOLIO_AGGREGATION_METHOD

    # Get the stock returns and format the portfolios
    prices_and_returns: pd.DataFrame = get_returns_stocks(prices_df)
    portfolios_df = portfolios_df.reset_index()

    # Initialize the dataframe to store portfolio prices
    col_names: List[str] = portfolios_df["sicdescription"].drop_duplicates().tolist()
    portfolio_returns = pd.DataFrame(columns=col_names)

    for _, row in portfolios_df.iterrows():

        # Get the prices and returns for the gvkeys in the portfolio
        gvkeys: List[Any] = row["gvkeys"]
        prices_and_returns_subset: pd.DataFrame = prices_and_returns[
            prices_and_returns["gvkey"].isin(gvkeys)
        ].copy()

        # Calculate the weights per stock
        if how == "MarketCap":
            # Compute the lagged Market Cap
            prices_and_returns_subset["Lagged_MarketCap"] = (
                prices_and_returns_subset.groupby("gvkey")["market_cap"].shift(1)
            )

            # Compute the weight as the MarketCap weight in the portfolio
            prices_and_returns_subset["Weight"] = prices_and_returns_subset[
                "Lagged_MarketCap"
            ] / prices_and_returns_subset.groupby("date")["Lagged_MarketCap"].transform(
                "sum"
            )

        elif how == "Equal":
            # Get the number of firms in the portfolio on each date
            num_firms: pd.Series = prices_and_returns_subset.groupby("date")[
                "gvkey"
            ].transform("nunique")
            prices_and_returns_subset["Weight"] = 1.0 / num_firms

        else:
            raise ValueError("Invalid aggregation method.")

        # Multiply the returns by the weights and sum them to get the portfolio return
        portfolio_returns[row["sicdescription"]] = (
            (prices_and_returns_subset["Weight"] * prices_and_returns_subset["return"])
            .groupby("date")
            .sum()
        )

    # Format the portfolio returns dataframe
    portfolio_returns_formated: pd.DataFrame = portfolio_returns.rename_axis(
        "date"
    ).dropna(how="all")

    return portfolio_returns_formated


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

    # Save the results
    portfolio_returns.to_csv(config.paths.portfolios_out(FILENAMES.Portfolio_returns))
    portfolio_constitution.to_csv(
        config.paths.portfolios_out(FILENAMES.Portfolio_construction_details)
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
    stock_prices, firm_info, sic_codes = get_stockprices_firminfo_siccodes(config)

    # Compute the market cap
    stock_prices["market_cap"] = compute_market_cap(
        stock_prices, "close", "sharesoutstanding"
    )

    # Apply market cap cutoff to filter the firms
    prices_cutoff: pd.DataFrame = apply_cutoff_latest_marketcap(stock_prices, config)

    # Get the main portfolios by industry using the SIC codes
    indutry_portfolios: pd.DataFrame = compute_industry_portfolios_sic(
        firm_descr=firm_info, sic_descr=sic_codes, config=config
    )

    # Add the information about firms and their stock prices
    industry_portfolios_with_info: pd.DataFrame = (
        intersect_portfolios_price_companyinfo(
            portfolios_df=indutry_portfolios,
            price_df=prices_cutoff,
            firm_info_df=firm_info,
        )
    )

    # Get the constitution of the portfolios
    industry_portfolio_constitution: pd.DataFrame = get_portfolio_constitution(
        industry_portfolios_with_info
    )

    # Add the sub-portfolios by market cap
    industry_marketcap_portfolios: pd.DataFrame = compute_marketcap_portfolios(
        main_portfolios=industry_portfolio_constitution, config=config
    )

    # Aggregate the portfolios
    industry_marketcap_portfolios_agg: pd.DataFrame = aggregate_sicportfolio(
        portfolio=industry_marketcap_portfolios
    )

    # Drop non-significant portfolios
    industry_marketcap_portfolios_filtered: pd.DataFrame = drop_nonsignicant_portfolios(
        portfolio_df=industry_marketcap_portfolios_agg, config=config
    )

    # Calculate the returns of the portfolios from their constituents
    industry_marketcap_portfolio_returns: pd.DataFrame = calculate_portfolio_returns(
        portfolios_df=industry_marketcap_portfolios_filtered,
        prices_df=stock_prices,
        config=config,
    )

    return industry_marketcap_portfolio_returns, industry_marketcap_portfolios_filtered


def run_pipeline(config: CONFIGURATION) -> None:
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
    run_pipeline(CONFIG)
