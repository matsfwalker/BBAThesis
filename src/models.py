import pandas as pd
import datetime as dt
import statsmodels.api as sm
from typing import Dict, Tuple, List, Union, Optional, Sequence
from configs import CONFIG, CONFIGURATION, FILENAMES


def download_processed_data(
    config: CONFIGURATION,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function to download the processed data and return it as a Dataframe

    Parameters
    ----------
    config: CONFIGURATION
        Configuration of the project

    Returns
    -------
    Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
        Returns a tuple containing the processed data for:
        - ff_factors_monthly
        - ff_factors_yearly
        - portfolio_returns_monthly
    """
    ff_factors_monthly: pd.DataFrame = pd.read_csv(
        config.paths.processed_read(FILENAMES.FF5_factors_monthly),
        parse_dates=["date"],
        index_col="date",
    )

    ff_factors_yearly: pd.DataFrame = pd.read_csv(
        config.paths.processed_read(FILENAMES.FF5_factors_yearly),
        parse_dates=["date"],
        index_col="date",
    )

    portfolio_returns_monthly: pd.DataFrame = pd.read_csv(
        config.paths.portfolios_read(FILENAMES.Portfolio_returns),
        parse_dates=["date"],
        index_col="date",
    )

    return ff_factors_monthly, ff_factors_yearly, portfolio_returns_monthly


def extract_factor_loadings(
    factors: pd.DataFrame, returns: Union[pd.Series, pd.DataFrame], rf_label: str = "RF"
) -> pd.DataFrame:
    """
    Function to perform a linear regression for a factor model (e.g., Fama-French 5-Factor Model).
    Returns the factor loadings (coefficients) and other statistics as a pandas Dataframe.
    This works for an arbitrary number of factors.
    Automaticaly aligns the factors and returns dataframes on their datetime index.

    Parameters
    ----------
    factors : pd.DataFrame
        DataFrame containing the independent variables (factors).
        Expects a column of the risk-free rate (canonically called RF).
        Expects a datetime index.
        Expects absolute factors as a number (not percentages).
    returns : Union[pd.Series, pd.DataFrame]∫
        Series containing the dependent variable (asset returns).
        Expects a datetime index.
        Expects absolute returns as a number.
        Works for a single stock and multiple stocks (DataFrame).
    rf_label : str = "RF"
        Optional argument to specify the label of the column in X representing the risk-free rate.
        Default is "RF"

    Returns
    -------
    pd.DataFrame
        DataFrame containing the factor loadings and regression statistics.
    """

    # Helper function for linear regression
    def lin_reg(X: pd.DataFrame, y: pd.Series) -> pd.Series:
        # Fit model
        model = sm.OLS(y, X, missing="drop").fit()

        values = []
        index = []

        for coeff in X.columns:
            name = "Alpha" if coeff == "const" else coeff

            index.extend(
                [
                    ("Beta", name),
                    ("Stdev", name),
                    ("Tstat", name),
                    ("Pvalue", name),
                ]
            )

            values.extend(
                [
                    model.params[coeff],
                    model.bse[coeff],
                    model.tvalues[coeff],
                    model.pvalues[coeff],
                ]
            )

        # Add Model statistics
        index.extend([("Rsquared", "Rsquared"), ("Rsquared", "Adj Rsquared")])
        values.extend([model.rsquared, model.rsquared_adj])

        return pd.Series(
            values,
            index=pd.MultiIndex.from_tuples(index, names=["Statistic", "Factor"]),
        ).sort_index(level=["Statistic", "Factor"])

    # Align the data on the index
    factors_aligned, returns_aligned = factors.align(returns, join="inner", axis=0)

    # Format the dependent variable by adding a constant to calculate alpha and removing RF from factors
    X = sm.add_constant(factors_aligned.drop(columns=[rf_label], inplace=False))

    # Check if we are working with multiple stocks
    if isinstance(returns_aligned, pd.DataFrame):
        results = {}

        # Extract and store the results for each stock
        for ticker in returns_aligned.columns:
            if all(returns_aligned[ticker].isna()):
                continue
            excess_return = returns_aligned[ticker] - factors_aligned[rf_label]
            results[ticker] = lin_reg(X, excess_return)

        return pd.DataFrame(results)

    # Case when working with one stock
    else:
        excess_return = returns_aligned - factors_aligned[rf_label]
        return lin_reg(X, excess_return).to_frame(name="Asset")


def factor_model_return_predictor(
    factor_values: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    rf_label: str = "RF",
    beta_label: str = "Beta",
) -> pd.DataFrame:
    """
    Function to predict the returns of assets using a factor model.
    Works for one or multiple assets.
    Works for an arbitrary number of factors.

    Parameters
    ----------
    factor_values : pd.DataFrame
        DataFrame containing the factor values for prediction.
        Expects a datetime index.
        Expects absolute factor values as numbers.
        Expects to have one column of the risk-free rate (canonically called "RF")
    factor_loadings : pd.DataFrame
        DataFrame containing the factor loadings for each asset.
        Expects the columns to be the asset ticker.
        Expects a MultiIndex with levels ["Statistic", "Factor"].
    rf_label : str = "RF"
        Optional argument to specify the label of the column in X representing the risk-free rate.
        Default is "RF"
    betas_label: str = "Beta"
        Optional argument to specify the label of the level in the MultiIndex representing the factor loadings.
        Default is "Beta"
    Returns
    -------
    pd.DataFrame
        DataFrame containing the predicted returns for each asset.
        The index is the datetime index of the factor values.
    """

    # Retrieve the different factors
    factors: List[str] = [
        factor for factor in factor_values.columns if factor != rf_label
    ]

    out: dict[str, pd.Series] = {}

    # Iterate over all tickers
    for ticker in factor_loadings.columns:
        loadings = factor_loadings[ticker].loc[beta_label]

        pred_excess = loadings["Alpha"]
        for factor in factors:
            pred_excess = pred_excess + factor_values[factor] * loadings[factor]

        out[ticker] = pred_excess + factor_values[rf_label]

    return pd.DataFrame(out, index=factor_values.index)


def compare_pred_actual(
    pred_return_monthly: pd.DataFrame, portfolio_returns_monthly: pd.DataFrame
) -> pd.DataFrame:
    """
    Function to compare the predicted and actual return and add the residual

    Parameters
    ----------
    pred_return_monthly : pd.DataFrame
        The predicted returns per month
    portfolio_returns_monthly : pd.DataFrame
        The actual returns per month

    Returns
    -------
    pd.DataFrame
        Dataframe containing the predicted, residual and actual return
    """
    # Merge the prediction and actual returns for comparison
    comparison_monthly_returns: pd.DataFrame = pd.concat(
        {
            "Pred_returns": pred_return_monthly,
            "Actual_returns": portfolio_returns_monthly,
            "Residual_returns": pred_return_monthly - portfolio_returns_monthly,
        },
        axis=1,
    )

    # Swap the levels to have tickers at top level for convenience
    comparison_monthly_returns = comparison_monthly_returns.swaplevel(
        0, 1, axis=1
    ).sort_index(axis=1)

    return comparison_monthly_returns


def gibbons_ross_shanken_test(
    model_parameters: pd.DataFrame, alpha_columns_name: str = "Alpha"
) -> float:
    """
    Function to compute the Gibbons-Ross-Shanken test stat for the test of pricing models.
    Computes and returns the sum of all alphas.
    The Null Hypothesis of the model is that all alphas are equal to zero.

    Parameters
    ----------
    model_parameters : pd.DataFrame
        DataFrame containing the model parameters for different assets.
        Expects a row for the alphas.
        Expects the columns to be the different assets of the tradeable unviverse.
    alpha_columns_name : str = "Alpha"
        Optional argument to specify the label of the row representing the alphas.
        Default is "Alpha"

    Returns
    -------
    float
        The Gibbons-Ross-Shanken test statistic.
    """
    return 0.0


def t_test_significance(
    model_parameters: pd.DataFrame, config: CONFIGURATION, t_stat_index: str = "Tstat"
) -> pd.DataFrame:
    """
    Function to test the significance of model parameters using t-statistics.
    Null Hypothesis: The parameter is equal to 0 and does not influence the model.
    Alternative Hypothesis: The parameter is different from 0 and influences the model.
    The function returns a dataframe of the analysis results.

    Parameters
    ----------
    model_parameters : pd.DataFrame
        Parameters of the model
        Expected to have one row containing the Tstat per parameter.
        Expects the Tstat to be in a seperate multi-index
    config : CONFIGURATION
        Configurations of the model
    t_stat_index : str = "Tstat"
        Name of the index in the multi-index representing the t-statistics.
        Default is "Tstat"

    Returns
    -------
    pd.DataFrame
        Returns the model_parameters dataframe with a new column for each parameter analyzed:
            - "<parameter_name>_is_significant": bool indicating if the parameter is significant (True) or not (False)
        Also reshapes the
    """
    # Get only the tstats
    t_stats: pd.DataFrame = model_parameters.loc[[t_stat_index]]

    # Unpack the configuration
    if config.T_TEST_FACTORS != "all":
        t_stats = t_stats[config.T_TEST_FACTORS]
    significance_level_tstat: float = config.T_TEST_SIGNIFICANCE_LEVEL

    significance: pd.DataFrame = t_stats.abs() > significance_level_tstat

    # Build new MultiIndex row
    new_index = pd.MultiIndex.from_tuples(
        [("is_significant", factor) for factor in t_stats.index]
    )

    # Create DataFrame and append
    sig_df = pd.DataFrame(significance.values, index=new_index, columns=t_stats.columns)
    model_parameters = pd.concat([model_parameters, sig_df])

    return model_parameters


def date_range_to_str(start_date: dt.datetime, end_date: dt.datetime) -> str:
    """
    Function to convert a date range to a string representation.

    Parameters
    ----------
    start_date : dt.datetime
        Start date of the range.
    end_date : dt.datetime
        End date of the range.

    returns
    -------
    str
        String representation of the date range.
    """

    return f"{start_date.strftime('%m/%Y')}:{end_date.strftime('%m/%Y')}"


def date_ranges_break_dates(
    all_dates: List[dt.datetime],
    break_dates: Sequence[dt.datetime],
    include_end_date: bool = True,
    include_start_date: bool = True,
    include_whole_period: bool = True,
) -> Dict[str, Tuple[dt.datetime, dt.datetime]]:
    """
    Function to generate date ranges from a list of break dates.
    Parameters
    ----------
    all_dates : List[dt.datetime]
        List of all dates available.
    break_dates : Sequence[dt.datetime]
        Sequence of break dates to generate ranges.
    include_end_date : bool = True
        Whether to include the end date in the range.
        Default is True.
    include_start_date : bool = True
        Whether to include the start date in the range.
        Default is True.
    include_whole_period : bool = True
        Whether to include the whole period from the minimum to maximum date.
        Default is True.

    Returns
    -------
    Dict[str, Tuple[dt.datetime, dt.datetime]]
        Dictionary relating the title of the timeframe to the start and end dates.
    """
    date_ranges: Dict[str, Tuple[dt.datetime, dt.datetime]] = {}

    sorted_break_dates: List[dt.datetime] = sorted(break_dates)
    sorted_dates: List[dt.datetime] = sorted(all_dates)

    if include_whole_period:
        date_ranges["Entire Period"] = (sorted_dates[0], sorted_dates[-1])

    if include_start_date:
        date_ranges[date_range_to_str(sorted_dates[0], sorted_break_dates[0])] = (
            sorted_dates[0],
            sorted_break_dates[0],
        )

    for i in range(len(sorted_break_dates) - 1):
        start_date: dt.datetime = sorted_break_dates[i]
        end_date: dt.datetime = sorted_break_dates[i + 1]
        date_ranges[date_range_to_str(start_date, end_date)] = (start_date, end_date)

    if include_end_date:
        date_ranges[date_range_to_str(sorted_break_dates[-1], sorted_dates[-1])] = (
            sorted_break_dates[-1],
            sorted_dates[-1],
        )

    return date_ranges


def date_ranges_windows(
    all_dates: List[dt.datetime],
    window_size_months: int,
    include_whole_period: bool = True,
) -> Dict[str, Tuple[dt.datetime, dt.datetime]]:
    """
    Function to generate date ranges using a rolling window approach.
    Parameters
    ----------
    all_dates : List[dt.datetime]
        List of all dates available.
    window_size_months : int
        Size of the sliding window in months
    include_whole_period : bool = True
        Whether to include the whole period from the minimum to maximum date.
        Default is True.

    Returns
    -------
    Dict[str, Tuple[dt.datetime, dt.datetime]]
        Dictionary relating the title of the timeframe to the start and end dates.
    """

    date_ranges: Dict[str, Tuple[dt.datetime, dt.datetime]] = {}

    sorted_dates: List[dt.datetime] = sorted(all_dates)
    start_date: dt.datetime = sorted_dates[0]
    end_date: dt.datetime = sorted_dates[-1]

    if include_whole_period:
        date_ranges["Entire Period"] = (sorted_dates[0], sorted_dates[-1])

    curr_date: dt.datetime = start_date
    last_start_date: Optional[dt.datetime] = None
    while curr_date < end_date:
        window_end_date: dt.datetime = curr_date + pd.DateOffset(
            months=window_size_months
        )
        date_ranges[date_range_to_str(curr_date, window_end_date)] = (
            curr_date,
            window_end_date,
        )
        last_start_date = curr_date
        curr_date = window_end_date

    if last_start_date != end_date and last_start_date is not None:
        date_ranges[date_range_to_str(last_start_date, end_date)] = (
            last_start_date,
            end_date,
        )

    return date_ranges


def factor_loadings_over_time(
    factors: pd.DataFrame,
    returns: Union[pd.Series, pd.DataFrame],
    config: CONFIGURATION,
    rf_label: str = "RF",
) -> pd.DataFrame:
    """
    Function to compute and compare the factor loadings for a factor model (e.g. Fama-French 5-Factor Model).
    Factor loadings for different time periods are computed and then saved in a DataFrame for comparison.
    Function to perform a linear regression for a factor model (e.g., Fama-French 5-Factor Model).
    Returns the factor loadings (coefficients) and other statistics for each time period.
    This works for an arbitrary number of factors.
    Automaticaly aligns the factors and returns dataframes on their datetime index.

    Parameters
    ----------
    factors : pd.DataFrame
        DataFrame containing the independent variables (factors).
        Expects a column of the risk-free rate (canonically called RF).
        Expects a datetime index.
        Expects absolute factors as a number (not percentages).
    returns : Union[pd.Series, pd.DataFrame]∫
        Series containing the dependent variable (asset returns).
        Expects a datetime index.
        Expects absolute returns as a number.
        Works for a single stock and multiple stocks (DataFrame).
    config : Configurations
        Configurations of the model
    rf_label : str = "RF"
        Optional argument to specify the label of the column in X representing the risk-free rate.
        Default is "RF"
    Returns
    -------
    pd.DataFrame
        DataFrame containing the factor loadings and regression statistics.
    """

    # Unpack the config
    break_dates: Optional[Sequence[dt.datetime]] = config.BREAK_DATE_PERIODS
    date_window_months: Optional[int] = config.PERIOD_WINDOW_LENGTH_MONTHS
    include_start_date: bool = config.INCLUDE_START_DATE_PERIOD
    include_end_date: bool = config.INCLUDE_END_DATE_PERIOD
    include_whole_period: bool = config.INCLUDE_WHOLE_PERIOD_MODEL

    assert (break_dates is not None) or (
        date_window_months is not None
    ), "Either break_dates or date_window_months must be specified."

    # Align the data on the index
    factors_aligned, returns_aligned = factors.align(returns, join="inner", axis=0)

    date_ranges: Dict[str, Tuple[dt.datetime, dt.datetime]] = {}

    # Determine the date ranges
    if break_dates is not None:
        date_ranges = date_ranges_break_dates(
            all_dates=factors_aligned.index.tolist(),
            break_dates=break_dates,
            include_end_date=include_end_date,
            include_start_date=include_start_date,
            include_whole_period=include_whole_period,
        )
    elif date_window_months is not None:
        date_ranges = date_ranges_windows(
            all_dates=factors_aligned.index.tolist(),
            window_size_months=date_window_months,
            include_whole_period=include_whole_period,
        )
    else:
        raise ValueError(
            "Either config.PERIOD_WINDOW_LENGTH_MONTHS or  config.BREAK_DATE_PERIODS must be defined"
        )

    # Run the regression for each time period
    regression_results: Dict[str, pd.DataFrame] = {
        label: extract_factor_loadings(
            factors=factors_aligned.loc[start_date:end_date],
            returns=returns_aligned.loc[start_date:end_date],
            rf_label=rf_label,
        )
        for label, (start_date, end_date) in date_ranges.items()
    }

    # Combine the results into a new multi-index df
    outcome: pd.DataFrame = pd.concat(
        regression_results, axis=1, names=["date Range", "Ticker"]
    )

    # Inverse the level of the column multi-index
    outcome = outcome.swaplevel("date Range", "Ticker", axis=1)
    return outcome


def build_model(
    config: CONFIGURATION,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrator function to build the model.
    Calls the different subfunctions that handle the model building in different steps

    Parameters
    ----------
    config : CONFIGURATION
        Configuration of the project

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Returns the data of the model:
        - factor_loadings_monthly
        - comparison_monthly_returns
        - ff_factors_over_time"""
    # Download the data
    ff_factors_monthly, ff_factors_yearly, portfolio_returns_monthly = (
        download_processed_data(config)
    )

    # Extract the factors
    factor_loadings_monthly = extract_factor_loadings(
        factors=ff_factors_monthly, returns=portfolio_returns_monthly
    )

    # Predict the returns according to the model
    pred_return_monthly: pd.DataFrame = factor_model_return_predictor(
        factor_values=ff_factors_monthly, factor_loadings=factor_loadings_monthly
    )

    # Compare predicted and actual
    comparison_monthly_returns: pd.DataFrame = compare_pred_actual(
        pred_return_monthly, portfolio_returns_monthly
    )

    # Test the significance using the t-test
    factor_loadings_monthly = t_test_significance(
        factor_loadings_monthly, config=CONFIG
    )

    # Calculate the factors in different periods
    ff_factors_over_time: pd.DataFrame = factor_loadings_over_time(
        factors=ff_factors_monthly, returns=portfolio_returns_monthly, config=CONFIG
    )

    return factor_loadings_monthly, comparison_monthly_returns, ff_factors_over_time


def save_model(
    factor_loadings_monthly: pd.DataFrame,
    comparison_monthly_returns: pd.DataFrame,
    ff_factors_over_time: pd.DataFrame,
    config: CONFIGURATION,
) -> None:
    """
    Function to save the info of the model in different places

    Parameters
    ----------
    factor_loadings_monthly : pd.DataFrame
        Dataframe containing the monthly factor loadings
    comparison_monthly_returns : pd.DataFrame
        Dataframe with the predicted, actual and residual returns of the portfolios
    ff_factors_over_time : pd.DataFrame
        Dataframe with the factor loadings for different periods
    config : CONFIGURATION
        Configuration of the project

    Returns
    -------
    None"""

    comparison_monthly_returns.to_csv(
        config.paths.results_out(FILENAMES.Comp_pred_actual_portfolio)
    )

    factor_loadings_monthly.to_csv(
        config.paths.results_out(FILENAMES.Factor_loadings_monthly)
    )

    ff_factors_over_time.to_csv(
        config.paths.results_out(FILENAMES.Factor_loadings_differentperiods)
    )


def build_save_model(config: CONFIGURATION) -> None:
    """
    Main orchestrator function to build and then save the model

    Parameters
    ----------
    config : CONFIGURATION
        COnfiguration of the project

    Returns
    -------
    None"""

    factor_loadings_monthly, comparison_monthly_returns, ff_factors_over_time = (
        build_model(config)
    )

    save_model(
        factor_loadings_monthly,
        comparison_monthly_returns,
        ff_factors_over_time,
        config,
    )


if __name__ == "__main__":
    build_save_model(CONFIG)
