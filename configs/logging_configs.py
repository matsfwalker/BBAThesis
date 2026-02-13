import logging

def setup_logging(log_file: str)->logging.Logger:
    """
    Function to initialise the logging configuration for the application.
    
    Parameters
    ----------
    log_file : str
        The path to the log file where logs will be written.
    
    Returns
    -------
    logging.Logger
        The configured logger instance.
    """