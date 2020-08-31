import logging
import sys
from pathlib import Path
import os

def configure_logging(args):
    """Set up standard logging to stdout and to a log file"""
    root = logging.getLogger()
    logging_level = logging.INFO
    root.handlers.clear()  ## to avoid multiple prints on subsequent runs
    root.setLevel(logging_level)

    # log_file = os.path.join('./checkpoints/', args.name + "/logger.info")
    log_file = os.path.join(args.checkpoint, "/logger.info")  # polyaxon

    # Stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging_level)
    formatter_stdout =logging.Formatter("%(message)s")
        # logging.Formatter(
        # '%(asctime)s %(pathname)s, line %(lineno)d - %(levelname)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter_stdout)
    root.addHandler(handler)

    # mpl_logger = logging.getLogger('matplotlib')
    # mpl_logger.setLevel(logging.WARNING)

    # if logfile_path is not None:
    # Logfile
    logfile = logging.FileHandler(log_file, mode='a')
    logfile.setLevel(logging.INFO)
    formatter_logfile = logging.Formatter('%(message)s')
    logfile.setFormatter(formatter_logfile)
    root.addHandler(logfile)
