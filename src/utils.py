import datetime as dt
import pandas as pd
import numpy as np
from typing import List, Sequence, Dict, Tuple, Optional, Iterator
from configs import CONFIGURATION_CLASS


def _date_range_to_str(start_date: dt.datetime, end_date: dt.datetime) -> str:
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


def _date_ranges_break_dates(
    all_dates: List[pd.Timestamp],
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
        date_ranges[_date_range_to_str(sorted_dates[0], sorted_break_dates[0])] = (
            sorted_dates[0],
            sorted_break_dates[0],
        )

    for i in range(len(sorted_break_dates) - 1):
        start_date: dt.datetime = sorted_break_dates[i]
        end_date: dt.datetime = sorted_break_dates[i + 1]
        date_ranges[_date_range_to_str(start_date, end_date)] = (start_date, end_date)

    if include_end_date:
        date_ranges[_date_range_to_str(sorted_break_dates[-1], sorted_dates[-1])] = (
            sorted_break_dates[-1],
            sorted_dates[-1],
        )

    return date_ranges


def _date_ranges_windows(
    all_dates: List[pd.Timestamp],
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
    # last_start_date: Optional[dt.datetime] = None

    while curr_date < end_date:
        window_end_date = min(
            curr_date + pd.DateOffset(months=window_size_months), end_date
        )
        date_ranges[_date_range_to_str(curr_date, window_end_date)] = (
            curr_date,
            window_end_date,
        )
        if window_end_date == end_date:
            break
        curr_date = window_end_date

    return date_ranges


def construct_date_ranges(
    df: pd.DataFrame,
    config: CONFIGURATION_CLASS,
) -> Dict[str, Tuple[dt.datetime, dt.datetime]]:
    """
    Function to create the date ranges depending on the config.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data with a datetime index to create the date ranges from.
    config : CONFIGURATION_CLASS
        Configuration of the model containing the parameters to create the date ranges.

    Returns
    -------
    Dict[str, Tuple[dt.datetime, dt.datetime]]
        Dictionary relating the title of the timeframe to the start and end dates.
    """
    # Deal with different formats
    if pd.api.types.is_numeric_dtype(df.index) and "date" in df.columns:
        df = df.set_index("date", drop=True)

    # Convert index to datetime
    converted = pd.to_datetime(df.index, errors="coerce")
    valid_ratio: float = float(np.mean(converted.notna().to_numpy()))
    if valid_ratio > 0.9:  # 90% convertible
        df.index = converted
    else:
        raise ValueError(
            f"Index of df should be of type pd.Timestamp, but is of type {type(df.index)}"
        )

    # Unpack the config
    break_dates: Optional[Sequence[dt.datetime]] = config.BREAK_DATE_PERIODS
    date_window_months: Optional[int] = config.PERIOD_WINDOW_LENGTH_MONTHS
    include_start_date: bool = config.INCLUDE_START_DATE_PERIOD
    include_end_date: bool = config.INCLUDE_END_DATE_PERIOD
    include_whole_period: bool = config.INCLUDE_WHOLE_PERIOD_MODEL

    assert (break_dates is not None) or (
        date_window_months is not None
    ), "Either break_dates or date_window_months must be specified."

    date_ranges: Dict[str, Tuple[dt.datetime, dt.datetime]] = {}

    # Determine the date ranges
    if break_dates is not None:
        date_ranges = _date_ranges_break_dates(
            all_dates=df.index.tolist(),
            break_dates=break_dates,
            include_end_date=include_end_date,
            include_start_date=include_start_date,
            include_whole_period=include_whole_period,
        )
    elif date_window_months is not None:
        date_ranges = _date_ranges_windows(
            all_dates=df.index.tolist(),
            window_size_months=date_window_months,
            include_whole_period=include_whole_period,
        )
    else:
        raise ValueError(
            "Either config.PERIOD_WINDOW_LENGTH_MONTHS or  config.BREAK_DATE_PERIODS must be defined"
        )

    return date_ranges


def chunkify_dates(
    start_date: dt.datetime, end_date: dt.datetime
) -> Iterator[Tuple[str, str]]:
    """
    Function to create an iterator of the dates between start_date and end_date in chunks of 1 year.
    This reduces the time to run one query.

    Parameters
    ----------
    start_date : dt.datetime
        Start date of the period to chunkify
    end_date : dt.datetime
        End date of the period to chunkify

    Returns
    -------
    Iterator[Tuple[str, str]]
        Iterator of tuples containing the start and end date of each chunk as strings
    """
    current = start_date

    while current <= end_date:
        next_year = dt.datetime(current.year + 1, 1, 1)
        chunk_end = min(next_year - dt.timedelta(days=1), end_date)

        yield (
            current.strftime("%Y-%m-%d"),
            chunk_end.strftime("%Y-%m-%d"),
        )

        current = chunk_end + dt.timedelta(days=1)
