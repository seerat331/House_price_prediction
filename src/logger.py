import logging
from pathlib import Path
from src.config import BASE_DIR

LOG_DIR=BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE=LOG_DIR /"project.log"

logger=logging.getLogger("HousePricePrediction")
logger.setLevel(logging.INFO)



logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter=logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    file_handler=logging.FileHandler(
    LOG_FILE,
    encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
