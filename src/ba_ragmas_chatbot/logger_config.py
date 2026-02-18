import logging
import sys  # <--- NEU

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)


def get_logger(name):
    return logging.getLogger(name)


def shutdown():
    logging.shutdown()
