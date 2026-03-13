import datetime as dt
from typing import Any, Dict, List, Tuple, Union, cast
import pandas as pd
import statsmodels.api as sm
from utils import construct_date_ranges

from configs import PROJ_CONFIG, CONFIGURATION_CLASS, FILENAMES_CLASS


def download_processed_data(
    config: CONFIGURATION_CLASS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Function to download the processed data and return it as a Dataframe

    Parameters
    ----------
    config: CONFIGURATION_CLASS
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
        config.paths.processed_read(FILENAMES_CLASS.FF5_factors_monthly),
        parse_dates=["date"],
        index_col="date",
    )

    ff_factors_yearly: pd.DataFrame = pd.read_csv(
        config.paths.processed_read(FILENAMES_CLASS.FF5_factors_yearly),
        parse_dates=["date"],
        index_col="date",
    )

    portfolio_info_monthly: pd.DataFrame = pd.read_csv(
        config.paths.portfolios_read(FILENAMES_CLASS.Portfolio_info),
        parse_dates=["date"],
        index_col=["date"],
    )

    # Bring the portfolio info to the format of a dataframe with the date as index and the columns as the different industries
    portfolio_returns_monthly: pd.DataFrame = portfolio_info_monthly.pivot(
        columns="industry_name", values="return"
    )

    if config.LOG_INFO:
        config.logger.info("Downloaded processed data successfully")

    return ff_factors_monthly, ff_factors_yearly, portfolio_returns_monthly


def extract_factor_loadings(
    factors: pd.DataFrame,
    config: CONFIGURATION_CLASS,
    returns: Union[pd.Series, pd.DataFrame],
    rf_label: str = "RF",
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
    config: CONFIGURATION_CLASS
        Configurations of the model
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


        if config.LOG_INFO:
            config.logger.info(
                "Extracted factor loadings successfully for multiple stocks"
            )
        return pd.DataFrame(results)

    # Case when working with one stock
    else:
        if config.LOG_INFO:
            config.logger.info("Extracted factor loadings successfully for one stock")

        excess_return = returns_aligned - factors_aligned[rf_label]
        return lin_reg(X, excess_return).to_frame(name="Asset")


def factor_model_return_predictor(
    factor_values: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    config: CONFIGURATION_CLASS,
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
    config: CONFIGURATION_CLASS
        Configurations of the model
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

    if config.LOG_INFO:
        config.logger.info("Predicted returns successfully using the factor model")
    return pd.DataFrame(out, index=factor_values.index)


def compare_pred_actual(
    pred_return_monthly: pd.DataFrame,
    portfolio_returns_monthly: pd.DataFrame,
    config: CONFIGURATION_CLASS,
) -> pd.DataFrame:
    """
    Function to compare the predicted and actual return and add the residual

    Parameters
    ----------
    pred_return_monthly : pd.DataFrame
        The predicted returns per month
    portfolio_returns_monthly : pd.DataFrame
        The actual returns per month
    config: CONFIGURATION_CLASS
        Configurations of the model

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
            "Residual_returns": portfolio_returns_monthly - pred_return_monthly,
        },
        axis=1,
    )

    # Swap the levels to have tickers at top level for convenience
    comparison_monthly_returns = comparison_monthly_returns.swaplevel(
        0, 1, axis=1
    ).sort_index(axis=1)

    if config.LOG_INFO:
        config.logger.info("Compared predicted and actual returns successfully")

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
    model_parameters: pd.DataFrame,
    config: CONFIGURATION_CLASS,
    t_stat_index: str = "Tstat",
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
    config : CONFIGURATION_CLASS
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

    if config.LOG_INFO:
        config.logger.info("Tested the significance of model parameters successfully")

    return model_parameters


def factor_loadings_over_time(
    factors: pd.DataFrame,
    returns: Union[pd.Series, pd.DataFrame],
    config: CONFIGURATION_CLASS,
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

    if config.logger:
        config.logger.info(
            "Starting Analysis for subperiods...\n" + "-"*80
        )

    # Align the data on the index
    factors_aligned, returns_aligned = factors.align(returns, join="inner", axis=0)

    date_ranges: Dict[str, Tuple[dt.datetime, dt.datetime]] = construct_date_ranges(
        factors_aligned, config
    )

    # Run the regression for each time period
    regression_results: Dict[str, pd.DataFrame] = {
        label: extract_factor_loadings(
            factors=factors_aligned.loc[
                cast(Any, start_date) : cast(Any, end_date)
            ],  # Cast for typechecker
            returns=returns_aligned.loc[
                cast(Any, start_date) : cast(Any, end_date)
            ],  # Cast for typechecker
            rf_label=rf_label,
            config=config,
        )
        for label, (start_date, end_date) in date_ranges.items()
    }

    # Combine the results into a new multi-index df
    outcome: pd.DataFrame = pd.concat(
        regression_results, axis=1, names=["date Range", "Ticker"]
    )

    # Inverse the level of the column multi-index
    outcome = outcome.swaplevel("date Range", "Ticker", axis=1)

    if config.LOG_INFO:
        config.logger.info(
            "Computed factor loadings for different time periods successfully"
        )

    return outcome


def build_model(
    config: CONFIGURATION_CLASS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Orchestrator function to build the model.
    Calls the different subfunctions that handle the model building in different steps

    Parameters
    ----------
    config : CONFIGURATION_CLASS
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

    if config.logger:
        config.logger.info(
            "Starting analysis of the entire period...\n" + "-"*80
        )
    
    # Extract the factors
    factor_loadings_monthly = extract_factor_loadings(
        factors=ff_factors_monthly, returns=portfolio_returns_monthly, config=config
    )

    # Predict the returns according to the model
    pred_return_monthly: pd.DataFrame = factor_model_return_predictor(
        factor_values=ff_factors_monthly,
        factor_loadings=factor_loadings_monthly,
        config=config,
    )

    # Compare predicted and actual
    comparison_monthly_returns: pd.DataFrame = compare_pred_actual(
        pred_return_monthly, portfolio_returns_monthly, config=config
    )

    # Test the significance using the t-test
    factor_loadings_monthly = t_test_significance(
        factor_loadings_monthly, config=config
    )

    # Calculate the factors in different periods
    ff_factors_over_time: pd.DataFrame = factor_loadings_over_time(
        factors=ff_factors_monthly,
        returns=portfolio_returns_monthly,
        config=PROJ_CONFIG,
    )

    return factor_loadings_monthly, comparison_monthly_returns, ff_factors_over_time


def save_model(
    factor_loadings_monthly: pd.DataFrame,
    comparison_monthly_returns: pd.DataFrame,
    ff_factors_over_time: pd.DataFrame,
    config: CONFIGURATION_CLASS,
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
    config : CONFIGURATION_CLASS
        Configuration of the project

    Returns
    -------
    None"""

    config.paths.results_save(
        df=comparison_monthly_returns, stem=FILENAMES_CLASS.Comp_pred_actual_portfolio
    )

    config.paths.results_save(
        df=factor_loadings_monthly, stem=FILENAMES_CLASS.Factor_loadings_monthly
    )

    config.paths.results_save(
        df=ff_factors_over_time, stem=FILENAMES_CLASS.Factor_loadings_differentperiods
    )

    if config.LOG_INFO:
        config.logger.info("Saved model results successfully")


def build_save_model(config: CONFIGURATION_CLASS) -> None:
    """
    Main orchestrator function to build and then save the model

    Parameters
    ----------
    config : CONFIGURATION_CLASS
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
    build_save_model(PROJ_CONFIG)
