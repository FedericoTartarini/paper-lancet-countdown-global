"""
Logging configuration for the project.
"""

import logging
from pathlib import Path


def setup_logging(project_root: Path, log_filename: str = "daily_summaries.log"):
    """
    Set up logging for the application.

    Args:
        project_root: The root directory of the project.
        log_filename: Name of the log file.

    Returns:
        logger: Configured logger instance.
    """
    # Create a logs directory
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / log_filename

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger(__name__)
    return logger
