import datetime as dt
from dataclasses import dataclass
from typing import Literal, Optional, Iterable, Union, List, Dict
import pandas as pd
from pathlib import Path
import re

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

type ALLOWED_TYPE = Literal["raw", "processed", "portfolios", "results"]


# Basic Configuration parent class
class BASIC_CONFIG_CLASS:
    # General info for all files
    suffix: str = ".csv"
    meta_info: str = dt.datetime.today().strftime("%Y-%m-%d")

    def get_directory(
        self,
        type_: ALLOWED_TYPE
    ) -> Path:
        raise NotImplementedError

    def get_latest(
        self,
        stem: str,
        type_: ALLOWED_TYPE,
        suffix: str,
        date_pattern: str = r"\d{4}-\d{2}-\d{2}",
    ) -> Path:
        """
        Returns the most recent file matching f"{directory}/{stem}_YYYY-MM-DD{suffix}".

        Parameters
        ----------
        stem : str
            Stem/name of the file
        type_ : ALLOWED_TYPE
            Type of the file
        suffix : str
            Suffix/filetype of the file (include .)
        date_pattern : str = r"\d{4}-\d{2}-\d{2}"
            Pattern under which the file has been saved
            Default is YYYY-MM-DD

        Returns
        -------
        Path
            Path to the file
        """

        directory: Path = self.get_directory(type_)

        rx = re.compile(
            rf"^{re.escape(stem)}_(?P<date>{date_pattern}){re.escape(suffix)}$"
        )
        matches = []
        for p in directory.iterdir():
            m = rx.match(p.name)
            if m:
                matches.append((m.group("date"), p))
        if not matches:
            raise FileNotFoundError(
                f"No files found for pattern {stem}_<date>{suffix} in {directory}"
            )
        # ISO date sorts lexicographically correctly
        return max(matches, key=lambda x: x[0])[1]

    def get_file(self, stem: str, type_: ALLOWED_TYPE, suffix: str, date: str) -> Path:
        """
        Returns the most recent file matching f"{directory}/{stem}_YYYY-MM-DD{suffix}".

        Parameters
        ----------
        stem : str
            Stem/name of the file
        type: ALLOWED_TYPE
            type of file
        suffix : str
            Suffix/filetype of the file (include .)
        date: str
            Date from which to get the file

        Returns
        -------
        Path
            Path to the file
        """
        directory: Path = self.get_directory(type_)

        return directory / f"{stem}_{type_}_{date}{suffix}"

    def create_filename_with_date(
        self, stem: str, type_: ALLOWED_TYPE, suffix: str, date_pattern: str = "%Y-%m-%d"
    ) -> Path:
        """
        Function to create a path for a new file with a date identifier.

        Parameters
        ----------
        stem : str
            Stem/name of the file
        type_ : ALLOWED_TYPE
            Type of the file
        suffix : str
            Suffix/filetype of the file (include .)
        date_pattern : str = "%Y-%m-%d"
            Date patter under which the file should be saved
            Default is YYYY-MM-DD

        Returns
        -------
        Path:
            Path to where the file should be saved
        """

        directory: Path = self.get_directory(type_)

        return directory / f"{stem}_{self.meta_info}{suffix}"

    def read(self, stem: str, type_: ALLOWED_TYPE, date: Optional[dt.datetime]) -> Path:
        if date is None:
            return self.get_latest(stem=stem, suffix=self.suffix, type_=type_)
        else:
            return self.get_file(
                stem=f"{stem}_raw",
                type_=type_,
                date=date.strftime("%Y-%m-%d"),
                suffix=self.suffix,
            )


# Paths for the analysis (only access to model and portfolio datas)
@dataclass(frozen=True, slots=True)
class PATH_ANALYSIS(BASIC_CONFIG_CLASS):

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
        return self.read(stem=stem, type_="portfolios", date=date)

    def results_out(self, stem: str) -> Path:
        return self.create_filename_with_date(
            stem=stem,
            type_="results",
            suffix=self.suffix,
        )

    def results_read(self, stem: str, date: Optional[dt.datetime] = None) -> Path:
        return self.read(stem=stem, type_="results", date=date)


# Paths for the entire program (all access)
@dataclass(frozen=True, slots=True)
class PATH_CONFIG(BASIC_CONFIG_CLASS):
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
        return self.read(stem=stem, type_="raw", date=date)

    def processed_out(self, stem: str) -> Path:
        return self.create_filename_with_date(
            stem=stem,
            type_="processed",
            suffix=self.suffix,
        )

    def processed_read(self, stem: str, date: Optional[dt.datetime] = None) -> Path:
        return self.read(stem=stem, type_="processed", date=date)

    def portfolios_out(self, stem: str) -> Path:
        return self.create_filename_with_date(
            stem=stem,
            type_="portfolios",
            suffix=self.suffix,
        )

    def portfolios_read(self, stem: str, date: Optional[dt.datetime] = None) -> Path:
        return self.read(stem=stem, type_="portfolios", date=date)

    def results_out(self, stem: str) -> Path:
        return self.create_filename_with_date(
            stem=stem,
            type_="results",
            suffix=self.suffix,
        )

    def results_read(self, stem: str, date: Optional[dt.datetime] = None) -> Path:
        return self.read(stem=stem, type_="results", date=date)

    def sql_query(self, stem: str) -> Path:
        return self.SQL_DIR / f"{stem}.sql"


# Configurations of the entire project
@dataclass(frozen=True, slots=True)
class CONFIGURATION:
    """Configuration dataclass to group all configurations."""

    # Paths
    paths: PATH_CONFIG

    # WRDS login
    WRDS_USERNAME: str
    WRDS_PASSWORD: str

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
    BREAK_DATE_PERIODS: Optional[Iterable[dt.datetime]]
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


# Plotting configurations
@dataclass(frozen=True, slots=True)
class PLOTTING_CONFIGURATIONS:
    TIMESPANS_TO_PLOT: List[Dict[str, Union[str, pd.Timestamp]]]
