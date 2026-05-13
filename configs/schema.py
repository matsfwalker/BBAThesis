import datetime as dt
import os
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union, Protocol

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

type ALLOWED_TYPE = Literal["raw", "processed", "portfolios", "results"]


# Basic Configuration parent class
class BasePathConfig:
    # General info for all files
    suffix: str = ".csv"

    @property
    def meta_info(self) -> str:
        return dt.datetime.today().strftime("%Y-%m-%d")

    def get_directory(self, type_: ALLOWED_TYPE) -> Path:
        raise NotImplementedError

    def get_latest(
        self,
        stem: str,
        type_: ALLOWED_TYPE,
        suffix: Optional[str] = None,
        date_pattern: str = r"\d{4}-\d{2}-\d{2}",
    ) -> Path:
        r"""
        Returns the file matching f"{directory}/{stem}_{type_}_YYYY-MM-DD{suffix}".

        Parameters
        ----------
        stem : str
            Stem/name of the file
        type_ : ALLOWED_TYPE
            Type of the file
        date_pattern : str = r"\d{4}-\d{2}-\d{2}"
            Pattern under which the file has been saved
            Default is YYYY-MM-DD
        suffix : Optional[str] = None
            Suffix/filetype of the file (include .)
            Default is None, using self.suffix
        Returns
        -------
        Path
            Path to the file
        """
        if suffix is None:
            suffix = self.suffix

        directory: Path = self.get_directory(type_)

        rx = re.compile(
            rf"^{re.escape(stem)}_{re.escape(type_)}_(?P<date>{date_pattern}){re.escape(suffix)}$"
        )
        matches = []
        for p in directory.iterdir():
            m = rx.match(p.name)
            if m:
                matches.append((m.group("date"), p))
        if not matches:
            raise FileNotFoundError(
                f"No files found for pattern {stem}_{type_}_<date>{suffix} in {directory}"
            )
        # ISO date sorts lexicographically correctly
        return max(matches, key=lambda x: x[0])[1]

    def get_file(
        self, stem: str, type_: ALLOWED_TYPE, date: str, suffix: Optional[str] = None
    ) -> Path:
        """
        Returns the most recent file matching f"{stem}_{type_}_{date}{suffix}"".

        Parameters
        ----------
        stem : str
            Stem/name of the file
        type_: ALLOWED_TYPE
            type of file
        date: str
            Date from which to get the file
        suffix : Optional[str] = None
            Suffix/filetype of the file (include .)
            Default is None, using self.suffix
        Returns
        -------
        Path
            Path to the file
        """

        if suffix is None:
            suffix = self.suffix

        directory: Path = self.get_directory(type_)

        return directory / f"{stem}_{type_}_{date}{suffix}"

    def create_filename_with_date(
        self, stem: str, type_: ALLOWED_TYPE, suffix: Optional[str] = None
    ) -> Path:
        """
        Function to create a path for a new file with a date identifier.

        Parameters
        ----------
        stem : str
            Stem/name of the file
        type_ : ALLOWED_TYPE
            Type of the file
        suffix: Optional[str] = None
            Suffix/filetype of the file (include .)
            Defaults to self.suffix

        Returns
        -------
        Path:
            Path to where the file should be saved
        """

        if suffix is None:
            suffix = self.suffix
        directory: Path = self.get_directory(type_)

        return directory / f"{stem}_{type_}_{self.meta_info}{suffix}"

    def resolve_path(
        self, stem: str, type_: ALLOWED_TYPE, date: Optional[dt.datetime]
    ) -> Path:
        if date is None:
            return self.get_latest(stem=stem, type_=type_)
        else:
            return self.get_file(stem=stem, type_=type_, date=date.strftime("%Y-%m-%d"))

    @staticmethod
    def save_df(
        df: Union[pd.DataFrame, pd.Series],
        path: Path,
        *args,
        save_description: bool = True,
        **kwargs,
    ) -> None:
        """
        Method to save a CSV file of the pd.DataFrame under the specified address.
        By default, it also saves the description of each dataframe in another file
        called dir/description_{filename}.csv

        Parameters
        ----------
        df: Union[pd.DataFrame, pd.Series]
            Dataframe or Series to be saved
        path: Path
            Path of where to save the df
        save_description: bool = True
            Whether to also save the description.
            Default is True

        Returns
        -------
        None"""

        df.to_csv(path, *args, **kwargs)

        if save_description:
            descr: Union[pd.DataFrame, pd.Series] = df.describe()
            descr.to_csv(path_or_buf=path.parent / f"description_{path.stem}.csv")


# Paths for the analysis (only access to model and portfolio data and results)
@dataclass(frozen=True, slots=True)
class PATH_ANALYSIS_CLASS(BasePathConfig):
    PORTFOLIO_DATA_DIR: Path
    # Result directories in the RESULTS_DIR
    RESULT_DATA_DIR: Path
    RESULT_IMAGES_DIR: Path
    RESULT_TABLES_DIR: Path
    LOGGING_DIR: Path

    def get_directory(self, type_: ALLOWED_TYPE) -> Path:
        match type_:
            case "results":
                return self.RESULT_DATA_DIR
            case "portfolios":
                return self.PORTFOLIO_DATA_DIR
            case "logs":
                return self.LOGGING_DIR
            case _:
                raise ValueError(f"Type {type_} is not allowed")

    def portfolios_save(
        self, df: Union[pd.DataFrame, pd.Series], stem: str, *args, **kwargs
    ) -> None:
        path: Path = self.create_filename_with_date(
            stem=stem,
            type_="portfolios",
            suffix=self.suffix,
        )
        self.save_df(df, path, *args, **kwargs)

    def portfolios_read(self, stem: str, date: Optional[dt.datetime] = None) -> Path:
        return self.resolve_path(stem=stem, type_="portfolios", date=date)

    def results_save(
        self, df: Union[pd.DataFrame, pd.Series], stem: str, *args, **kwargs
    ) -> None:
        path: Path = self.create_filename_with_date(
            stem=stem,
            type_="results",
            suffix=self.suffix,
        )
        self.save_df(df, path, *args, **kwargs)

    def results_read(self, stem: str, date: Optional[dt.datetime] = None) -> Path:
        return self.resolve_path(stem=stem, type_="results", date=date)


# Paths for the entire program (all access)
@dataclass(frozen=True, slots=True)
class PATH_CONFIG_CLASS(BasePathConfig):
    # SQL directory
    SQL_DIR: Path

    # Data directories in the DATA_DIR
    RAW_DATA_DIR: Path
    PROCESSED_DATA_DIR: Path
    PORTFOLIO_DATA_DIR: Path

    # Result directories in the RESULTS_DIR
    RESULT_DATA_DIR: Path
    RESULT_IMAGES_DIR: Path

    # Logging directory
    LOGGING_DIR: Path

    def get_directory(self, type_: ALLOWED_TYPE) -> Path:
        match type_:
            case "raw":
                return self.RAW_DATA_DIR
            case "processed":
                return self.PROCESSED_DATA_DIR
            case "results":
                return self.RESULT_DATA_DIR
            case "portfolios":
                return self.PORTFOLIO_DATA_DIR
            case "logs":
                return self.LOGGING_DIR
            case _:
                raise ValueError(f"Type {type_} is not allowed")

    def raw_save(
        self, df: Union[pd.DataFrame, pd.Series], stem: str, *args, **kwargs
    ) -> None:
        path: Path = self.create_filename_with_date(
            stem=stem,
            type_="raw",
            suffix=self.suffix,
        )
        self.save_df(df, path, *args, **kwargs)

    def raw_read(self, stem: str, date: Optional[dt.datetime] = None) -> Path:
        return self.resolve_path(stem=stem, type_="raw", date=date)

    def processed_save(
        self, df: Union[pd.DataFrame, pd.Series], stem: str, *args, **kwargs
    ) -> None:
        path: Path = self.create_filename_with_date(
            stem=stem,
            type_="processed",
            suffix=self.suffix,
        )

        self.save_df(df, path, *args, **kwargs)

    def processed_read(self, stem: str, date: Optional[dt.datetime] = None) -> Path:
        return self.resolve_path(stem=stem, type_="processed", date=date)

    def portfolios_save(
        self, df: Union[pd.DataFrame, pd.Series], stem: str, *args, **kwargs
    ) -> None:
        path: Path = self.create_filename_with_date(
            stem=stem,
            type_="portfolios",
            suffix=self.suffix,
        )
        self.save_df(df, path, *args, **kwargs)

    def portfolios_read(self, stem: str, date: Optional[dt.datetime] = None) -> Path:
        return self.resolve_path(stem=stem, type_="portfolios", date=date)

    def results_save(
        self, df: Union[pd.DataFrame, pd.Series], stem: str, *args, **kwargs
    ) -> None:
        path: Path = self.create_filename_with_date(
            stem=stem,
            type_="results",
            suffix=self.suffix,
        )
        self.save_df(df, path, *args, **kwargs)

    def results_read(self, stem: str, date: Optional[dt.datetime] = None) -> Path:
        return self.resolve_path(stem=stem, type_="results", date=date)

    def sql_query(self, stem: str) -> Path:
        return self.SQL_DIR / f"{stem}.sql"

    def read_raw_txt(self, stem: str) -> Path:
        return self.get_latest(stem=stem, type_="raw", suffix=".txt")


# Configurations of the entire project
@dataclass(frozen=True, slots=True)
class CONFIGURATION_CLASS:
    """Configuration dataclass to group all configurations."""

    # Paths
    paths: PATH_CONFIG_CLASS

    # Logging
    LOG_INFO: bool
    logger: logging.Logger

    # Constants
    FACTORS_LIB: str
    FACTORS_DATA_SOURCE: str
    INFLATION_LIB: str
    INFLATION_SOURCE: str

    # Data downloading configs
    START_DATE_ANALYSIS: dt.datetime
    END_DATE_ANALYSIS: dt.datetime

    # Data-cleaning configs
    THRESHOLD_MISSING_SHARESOUTSTANDING: float
    MIN_STOCK_PRICE: float
    EXCHANGES_TO_KEEP: List[str]
    MAX_MONTHLY_RETURN: Optional[float]

    # Portfolio creation configs
    CUTOFF_FIRMS_PER_PORTFOLIO: int
    MIN_OCCURANCES_PORTFOLIOS_ENTIRE: int
    MIN_OCCURANCES_PORTFOLIOS_SUB: int
    MIN_MARKETCAP_FIRM: float
    DISCOUNT_MARKETCAP_FIRM_INFLATION: bool
    INDUSTRY_CLASSIFICATION_METHOD: Literal["Sic_level", "Fama-French_portfolios"]
    FAMA_FRENCH_INDUSTRY_PORTFOLIOS: Optional[
        Literal[
            "Siccodes5",
            "Siccodes17",
            "Siccodes30",
            "Siccodes38",
            "Siccodes48",
            "Siccodes49",
        ]
    ]
    SIC_LEVEL: Optional[Literal[1, 2, 3, 4]]
    PORTFOLIO_AGGREGATION_METHOD: Literal["MarketCap", "Equal"]

    # Model configurations
    BREAK_DATE_PERIODS: Optional[Sequence[dt.datetime]]
    INCLUDE_END_DATE_PERIOD: bool
    INCLUDE_START_DATE_PERIOD: bool
    PERIOD_WINDOW_LENGTH_MONTHS: Optional[int]
    INCLUDE_WHOLE_PERIOD_MODEL: bool

    CREATE_MARKETCAP_PORTFOLIOS: bool
    MARKETCAP_PORTFOLIO_PERCENTILE: Optional[List[float]]
    MARKETCAP_PORTFOLIO_EXCHANGE: Optional[Union["str", Literal["all"]]]
    MARKETCAP_PORTFOLIO_NUMBER_FIRMS: Optional[int]

    # Statistical configurations
    T_TEST_FACTORS: Union[List[str], Literal["all"]]
    T_TEST_SIGNIFICANCE_LEVEL: float
    P_THRESHOLD: float

    def __post_init__(self):
        # Make sure the data is well structured
        policies: List[Policy] = [
            IndustryClassificationPolicy(),
            MarketCapPortfolioPolicy(),
            DatePolicy(),
            CutoffPoliciy(),
            StatsPolicy(),
        ]
        for policy in policies:
            policy.validate(self)


    def get_wrds_data(self) -> Dict[str, str]:
        load_dotenv(PROJECT_ROOT / "configs/.env")
        username: Optional[str] = os.getenv("WRDS_USERNAME")
        password: Optional[str] = os.getenv("WRDS_PASSWORD")

        if username is None:
            raise ValueError("Username for WRDS is None")

        if password is None:
            raise ValueError("Password for WRDS is None")

        return {"username": username, "password": password}


# Policies for Configuration dataclass validation
class Policy(Protocol): # Class for defining policies to the static type checker
    def validate(self, config: CONFIGURATION_CLASS) -> None:
        ...

class IndustryClassificationPolicy():
    def validate(self, config: CONFIGURATION_CLASS) -> None:
        self.isvalid_classification_method(config)

        if config.INDUSTRY_CLASSIFICATION_METHOD == "Sic_level":
            self.isvalid_sic_level(config)
        elif config.INDUSTRY_CLASSIFICATION_METHOD == "Fama-French_portfolios":
            self.isvalid_famafrench_configuration(config)
        else:
            raise ValueError(f"Encountered invalid INDUSTRY_CLASSIFICATION_METHOD: {config.INDUSTRY_CLASSIFICATION_METHOD}")
    
    def isvalid_classification_method(self, config: CONFIGURATION_CLASS) -> None:
        if config.INDUSTRY_CLASSIFICATION_METHOD not in {
            "Sic_level",
            "Fama-French_portfolios",
        }:
            raise ValueError(
                "INDUSTRY_CLASSIFICATION_METHOD must be 'Sic_level' or 'Fama-French_portfolios'"
            )

    def isvalid_sic_level(self, config: CONFIGURATION_CLASS) -> None:
        if config.SIC_LEVEL is None:
            raise ValueError(
                "SIC_LEVEL must be provided when INDUSTRY_CLASSIFICATION_METHOD is 'Sic_level'"
            )
        elif config.SIC_LEVEL not in {1, 2, 3, 4}:
            raise ValueError("SIC_LEVEL must be one of {1, 2, 3, 4}")
        
    def isvalid_famafrench_configuration(self, config: CONFIGURATION_CLASS) -> None:
            if config.FAMA_FRENCH_INDUSTRY_PORTFOLIOS is None:
                raise ValueError(
                    "FAMA_FRENCH_INDUSTRY_PORTFOLIOS must be provided when INDUSTRY_CLASSIFICATION_METHOD is 'Fama-French_portfolios'"
                )
            else:
                available_portfolios = {
                    "Siccodes5",
                    "Siccodes17",
                    "Siccodes30",
                    "Siccodes38",
                    "Siccodes48",
                    "Siccodes49",
                }
                if config.FAMA_FRENCH_INDUSTRY_PORTFOLIOS not in available_portfolios:
                    raise ValueError(
                        f"FAMA_FRENCH_INDUSTRY_PORTFOLIOS must be one of {available_portfolios}"
                    )

class MarketCapPortfolioPolicy():
    def validate(self, config: CONFIGURATION_CLASS) -> None:
        if not config.CREATE_MARKETCAP_PORTFOLIOS:
            return
        
        else:
            if config.MARKETCAP_PORTFOLIO_NUMBER_FIRMS is not None:
                self.isvalid_marketcap_portfolio_number_firms(config)
            elif config.MARKETCAP_PORTFOLIO_PERCENTILE is None:
                raise ValueError(
                    "Either MARKETCAP_PORTFOLIO_NUMBER_FIRMS or MARKETCAP_PORTFOLIO_PERCENTILE can be provided, but not both"
                )

    
    def isvalid_marketcap_portfolio_number_firms(self, config: CONFIGURATION_CLASS) -> None:

        if config.MARKETCAP_PORTFOLIO_EXCHANGE is None:
            raise ValueError(
                "If the market cap percentile is defined, then the exchanges to be used must also be defined."
            )
        elif (config.MARKETCAP_PORTFOLIO_EXCHANGE not in config.EXCHANGES_TO_KEEP) and (
            config.MARKETCAP_PORTFOLIO_EXCHANGE != "all"
        ):
            raise ValueError(
                f"{config.MARKETCAP_PORTFOLIO_EXCHANGE} is an invalid exchange. Must be either 'all' or in {config.EXCHANGES_TO_KEEP}."
            )

class DatePolicy():
    def validate(self, config: CONFIGURATION_CLASS) -> None:
        if config.END_DATE_ANALYSIS < config.START_DATE_ANALYSIS:
            raise ValueError("END_DATE_ANALYSIS must be after START_DATE_ANALYSIS")

        if not (config.BREAK_DATE_PERIODS is None) ^ (
            config.PERIOD_WINDOW_LENGTH_MONTHS is None
        ):
            raise ValueError(
                "Either BREAK_DATE_PERIODS or PERIOD_WINDOW_LENGTH_MONTHS must be provided, but not both"
            )

class CutoffPoliciy():
    def validate(self, config: CONFIGURATION_CLASS) -> None:
        if config.CUTOFF_FIRMS_PER_PORTFOLIO < 0:
            raise ValueError("CUTOFF_FIRMS_PER_PORTFOLIO must be positive")

        if config.MIN_MARKETCAP_FIRM < 0:
            raise ValueError("MIN_MARKETCAP_FIRM must be non-negative")

        if config.PORTFOLIO_AGGREGATION_METHOD not in {"MarketCap", "Equal"}:
            raise ValueError(
                "PORTFOLIO_AGGREGATION_METHOD must be 'MarketCap' or 'Equal'"
            )

        if config.MIN_OCCURANCES_PORTFOLIOS_SUB < 6:
            raise ValueError(
                f"MIN_OCCURANCES_PORTFOLIOS_SUB({config.MIN_OCCURANCES_PORTFOLIOS_SUB}) is too low. Needed at least 6 (degrees of freedom + 1)"
            )
        
        if (
            config.THRESHOLD_MISSING_SHARESOUTSTANDING < 0
            or config.THRESHOLD_MISSING_SHARESOUTSTANDING > 1
        ):
            raise ValueError(
                "THRESHOLD_MISSING_SHARESOUTSTANDING must be between 0 and 1"
            )

class StatsPolicy():
    def validate(self, config: CONFIGURATION_CLASS) -> None:
        if config.T_TEST_SIGNIFICANCE_LEVEL <= 0:
            raise ValueError("T_TEST_SIGNIFICANCE_LEVEL must be positive")

        if config.P_THRESHOLD <= 0 or config.P_THRESHOLD >= 1:
            raise ValueError("P_THRESHOLD must be between 0 and 1")

# Plotting configurations
@dataclass(frozen=True, slots=True)
class PLOTTING_CONFIGURATIONS_CLASS:
    TIMESPANS_TO_PLOT: List[Dict[str, Union[str, pd.Timestamp, dt.datetime]]]


@dataclass(slots=True)
class DATAFRAME_CONTAINER:
    monthly_stock_info: pd.DataFrame
    firm_info: pd.DataFrame
    sic_info: pd.DataFrame
    monthly_inflation: Union[pd.Series, pd.DataFrame]
    ff_industry_portfolios: pd.DataFrame
    monthly_fama_french: Optional[pd.DataFrame] = None
    yearly_fama_french: Optional[pd.DataFrame] = None
