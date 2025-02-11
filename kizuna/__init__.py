from kizuna.modeling.model import KModel
from kizuna.pipeline import KPipeline

__version__ = '0.1.0'

import sys

from loguru import logger

# Remove default handler
logger.remove()

# Add custom handler with clean format including module and line number
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <cyan>{module:>16}:{line}</cyan> | <level>{level: >8}</level> | <level>{message}</level>",
    colorize=True,
    level="INFO" # "DEBUG" to enable logger.debug("message") and up prints 
                 # "ERROR" to enable only logger.error("message") prints
                 # etc
)

# Disable before release or as needed
logger.disable("kizuna")

__all__ = ["KModel", "KPipeline"]
