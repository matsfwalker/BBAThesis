import datetime as dt
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union

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
            self, 
            stem: str, 
            type_: ALLOWED_TYPE, 
            date: str,
            suffix: Optional[str] = None
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
        self,
        stem: str,
        type_: ALLOWED_TYPE,
        suffix: Optional[str] = None
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

    def resolve_path(self, stem: str, type_: ALLOWED_TYPE, date: Optional[dt.datetime]) -> Path:
        if date is None:
            return self.get_latest(stem=stem, type_=type_)
        else:
            return self.get_file(
                stem=stem,
                type_=type_,
                date=date.strftime("%Y-%m-%d")
            )


# Paths for the analysis (only access to model and portfolio datas)
@dataclass(frozen=True, slots=True)
class PATH_ANALYSIS(BasePathConfig):

    PORTFOLIO_DATA_DIR: Path
    # Result directories in the RESULTS_DIR
    RESULT_DATA_DIR: Path
    RESULT_IMAGES_DIR: Path

    def get_directory(self, type_: ALLOWED_TYPE) -> Path:
        match type_:
            case "results":
                return self.RESULT_DATA_DIR
            case "portfolios":
                return self.PORTFOLIO_DATA_DIR
            case _:
                raise ValueError(f"Type {type_} is not allowed")

    def portfolios_out(self, stem: str) -> Path:
        return self.create_filename_with_date(
            stem=stem,
            type_="portfolios",
            suffix=self.suffix,
        )

    def portfolios_read(self, stem: str, date: Optional[dt.datetime] = None) -> Path:
        return self.resolve_path(stem=stem, type_="portfolios", date=date)

    def results_out(self, stem: str) -> Path:
        return self.create_filename_with_date(
            stem=stem,
            type_="results",
            suffix=self.suffix,
        )

    def results_read(self, stem: str, date: Optional[dt.datetime] = None) -> Path:
        return self.resolve_path(stem=stem, type_="results", date=date)


# Paths for the entire program (all access)
@dataclass(frozen=True, slots=True)
class PATH_CONFIG(BasePathConfig):
    # SQL directory
    SQL_DIR: Path

    # Data directories in the DATA_DIR
    RAW_DATA_DIR: Path
    PROCESSED_DATA_DIR: Path
    PORTFOLIO_DATA_DIR: Path

    # Result directories in the RESULTS_DIR
    RESULT_DATA_DIR: Path
    RESULT_IMAGES_DIR: Path

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
            case _:
                raise ValueError(f"Type {type_} is not allowed")

    def raw_out(self, stem: str) -> Path:
        return self.create_filename_with_date(
            stem=stem,
            type_="raw",
            suffix=self.suffix,
        )

    def raw_read(self, stem: str, date: Optional[dt.datetime] = None) -> Path:
        return self.resolve_path(stem=stem, type_="raw", date=date)

    def processed_out(self, stem: str) -> Path:
        return self.create_filename_with_date(
            stem=stem,
            type_="processed",
            suffix=self.suffix,
        )

    def processed_read(self, stem: str, date: Optional[dt.datetime] = None) -> Path:
        return self.resolve_path(stem=stem, type_="processed", date=date)

    def portfolios_out(self, stem: str) -> Path:
        return self.create_filename_with_date(
            stem=stem,
            type_="portfolios",
            suffix=self.suffix,
        )

    def portfolios_read(self, stem: str, date: Optional[dt.datetime] = None) -> Path:
        return self.resolve_path(stem=stem, type_="portfolios", date=date)

    def results_out(self, stem: str) -> Path:
        return self.create_filename_with_date(
            stem=stem,
            type_="results",
            suffix=self.suffix,
        )

    def results_read(self, stem: str, date: Optional[dt.datetime] = None) -> Path:
        return self.resolve_path(stem=stem, type_="results", date=date)

    def sql_query(self, stem: str) -> Path:
        return self.SQL_DIR / f"{stem}.sql"


# Configurations of the entire project
@dataclass(frozen=True, slots=True)
class CONFIGURATION:
    """Configuration dataclass to group all configurations."""

    # Paths
    paths: PATH_CONFIG

    # Constants
    FACTORS_LIB: str
    FACTORS_DATA_SOURCE: str

    # Data downloading configs
    START_DATE_FACTORS_DOWNLOAD: dt.datetime
    END_DATE_FACTORS_DOWNLOAD: dt.datetime

    # Data-cleaning configs
    THRESHOLD_MISSING_SHARESOUTSTANDING: float

    # Portfolio creation configs
    CUTOFF_FIRMS_PER_PORTFOLIO: int
    MIN_MARKETCAP_FIRM: float
    SIC_LEVEL: Literal[1, 2, 3, 4]
    PORTFOLIO_AGGREGATION_METHOD: Literal["MarketCap", "Equal"]

    # Model configurations
    BREAK_DATE_PERIODS: Optional[Sequence[dt.datetime]]
    INCLUDE_END_DATE_PERIOD: bool
    INCLUDE_START_DATE_PERIOD: bool
    PERIOD_WINDOW_LENGTH_MONTHS: Optional[int]
    INCLUDE_WHOLE_PERIOD_MODEL: bool

    MARKETCAP_PORTFOLIO_PERCENTILE: Optional[float]
    MARKETCAP_PORTFOLIO_NUMBER_FIRMS: Optional[int]

    # Statistical configurations
    T_TEST_FACTORS: Union[List[str], Literal["all"]]
    T_TEST_SIGNIFICANCE_LEVEL: float

    def __post_init__(self):
        # Make sure the data is well structured
        if self.END_DATE_FACTORS_DOWNLOAD < self.START_DATE_FACTORS_DOWNLOAD:
            raise ValueError(
                "END_DATE_FACTORS_DOWNLOAD must be after START_DATE_FACTORS_DOWNLOAD"
            )

        if self.CUTOFF_FIRMS_PER_PORTFOLIO < 0:
            raise ValueError("CUTOFF_FIRMS_PER_PORTFOLIO must be positive")

        if self.MIN_MARKETCAP_FIRM < 0:
            raise ValueError("MIN_MARKETCAP_FIRM must be non-negative")

        if self.SIC_LEVEL not in {1, 2, 3, 4}:
            raise ValueError("SIC_LEVEL must be one of {1, 2, 3, 4}")

        if self.PORTFOLIO_AGGREGATION_METHOD not in {"MarketCap", "Equal"}:
            raise ValueError(
                "PORTFOLIO_AGGREGATION_METHOD must be 'MarketCap' or 'Equal'"
            )

        if not (self.BREAK_DATE_PERIODS is None) ^ (
            self.PERIOD_WINDOW_LENGTH_MONTHS is None
        ):
            raise ValueError(
                "Either BREAK_DATE_PERIODS or PERIOD_WINDOW_LENGTH_MONTHS must be provided, but not both"
            )

        if not (self.MARKETCAP_PORTFOLIO_NUMBER_FIRMS is None) ^ (
            self.MARKETCAP_PORTFOLIO_PERCENTILE is None
        ):
            raise ValueError(
                "Either MARKETCAP_PORTFOLIO_NUMBER_FIRMS or MARKETCAP_PORTFOLIO_PERCENTILE can be provided, but not both"
            )

        if self.T_TEST_SIGNIFICANCE_LEVEL <= 0:
            raise ValueError("T_TEST_SIGNIFICANCE_LEVEL must be positive")

        if (
            self.THRESHOLD_MISSING_SHARESOUTSTANDING < 0
            or self.THRESHOLD_MISSING_SHARESOUTSTANDING > 1
        ):
            raise ValueError(
                "THRESHOLD_MISSING_SHARESOUTSTANDING must be between 0 and 1"
            )

    def get_wrds_data(self) -> Dict[str, str]:
        load_dotenv(PROJECT_ROOT / "configs/.env")
        username: Optional[str] = os.getenv("WRDS_USERNAME")
        password: Optional[str] = os.getenv("WRDS_PASSWORD")

        if username is None:
            raise ValueError("Username for WRDS is None")

        if password is None:
            raise ValueError("Password for WRDS is None")

        return {"username": username, "password": password}


# Plotting configurations
@dataclass(frozen=True, slots=True)
class PLOTTING_CONFIGURATIONS:
    TIMESPANS_TO_PLOT: List[Dict[str, Union[str, pd.Timestamp, dt.datetime]]]
