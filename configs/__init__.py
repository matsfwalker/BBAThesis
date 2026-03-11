# src/configs/__init__.py
from .configurations import PROJ_CONFIG, PLOTTING_CONFIG, PLOTTING_CONFIGURATIONS_CLASS
from .paths import ANALYSIS_PATHS, FILENAMES_CLASS, FILENAMES_ANALYSIS_CLASS
from .schema import CONFIGURATION_CLASS, DATAFRAME_CONTAINER

__all__ = [
    "CONFIGURATION_CLASS",  # For type hinting
    "PROJ_CONFIG",  # Configuration of the project
    "ANALYSIS_PATHS",  # Paths for the analysis
    "FILENAMES_ANALYSIS_CLASS",  # Enum for the filenames for the analysis
    "FILENAMES_CLASS",  # Enum for all filenames
    "PLOTTING_CONFIG",  # Configurations for plotting
    "PLOTTING_CONFIGURATIONS_CLASS",  # Class of PLOTTING_CONFIG
    "DATAFRAME_CONTAINER",  # Container for the dataframes for downloading and processing data
]
