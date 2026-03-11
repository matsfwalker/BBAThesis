# src/configs/__init__.py
from .configurations import CONFIG, PLOTTING_CONFIG, PLOTTING_CONFIGURATIONS
from .paths import ANALYSIS_PATHS, FILENAMES_CLASS, FILENAMES_ANALYSIS_CLASS
from .schema import CONFIGURATION, DATAFRAME_CONTAINER

__all__ = [
    "CONFIGURATION",  # For type hinting
    "CONFIG",  # Configuration of the project
    "ANALYSIS_PATHS",  # Paths for the analysis
    "FILENAMES_ANALYSIS_CLASS",  # Enum for the filenames for the analysis
    "FILENAMES_CLASS",  # Enum for all filenames
    "PLOTTING_CONFIG",  # Configurations for plotting
    "PLOTTING_CONFIGURATIONS",  # Class of PLOTTOMG_CONFIG
    "DATAFRAME_CONTAINER",  # Container for the dataframes for downloading and processing data
]
