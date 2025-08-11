import logging

def start_logging(
    level=logging.INFO,
    fmt="%(asctime)s|%(filename)s|%(lineno)d|%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_to_file: str | None = None
):
    """
    Initializes logging with a standard format.
    
    Simply run 
    ``
    import logging
    cosmotracer.utils.start_logging()
    logger = logging.getLogger(__name__)
    ``
    
    Parameters:
    -----------
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG)
    fmt : str
        Log message format
    datefmt : str
        Format for timestamps
    log_to_file : str or None
        If set, logs will be written to the specified file instead of stdout
    """
    handlers = []
    if log_to_file:
        handlers.append(logging.FileHandler(log_to_file))
    else:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=handlers
    )