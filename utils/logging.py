import logging
import sys
from pathlib import Path


def configure_logging(log_debug=False, logfile_path: Path = None):
    """Set up standard logging to stdout and to a log file"""
    root = logging.getLogger()

    logging_level = logging.DEBUG if log_debug else logging.INFO

    root.setLevel(logging_level)

    # Stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging_level)
    formatter_stdout = logging.Formatter(
        '%(asctime)s %(pathname)s, line %(lineno)d - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter_stdout)
    root.addHandler(handler)

    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)

    if logfile_path is not None:
        # Logfile
        logfile = logging.FileHandler(logfile_path, mode='w')
        logfile.setLevel(logging.DEBUG)
        formatter_logfile = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logfile.setFormatter(formatter_logfile)
        root.addHandler(logfile)
