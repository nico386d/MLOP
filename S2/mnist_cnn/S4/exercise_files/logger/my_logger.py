import sys
from loguru import logger

# Remove default logger
logger.remove()

# 1) Terminal: only WARNING and higher
logger.add(
    sys.stdout,
    level="WARNING"
)

# 2) File: save everything (DEBUG and above), rotate at 100 MB
logger.add(
    "my_log.log",
    level="DEBUG",
    rotation="100 MB"
)

# Test messages
logger.debug("Used for debugging your code.")
logger.info("Informative messages from your code.")
logger.warning("Everything works but there is something to be aware of.")
logger.error("There's been a mistake with the process.")
logger.critical("There is something terribly wrong and process may terminate.")
