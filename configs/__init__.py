# src/configs/__init__.py
from .configurations import CONFIG, PLOTTING_CONFIG
from .paths import ANALYSIS_PATHS, FILENAMES, FILENAMES_ANALYSIS
from .schema import CONFIGURATION

__all__ = [
    "CONFIGURATION",  # For type hinting
    "CONFIG",  # Configuration of the project
    "ANALYSIS_PATHS",  # Paths for the analysis
    "FILENAMES_ANALYSIS",  # Enum for the filenames for the analysis
    "FILENAMES",  # Enum for all filenames
    "PLOTTING_CONFIG",  # Configurations for plotting
]
