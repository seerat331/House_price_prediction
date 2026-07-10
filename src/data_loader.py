import pandas as pd
from src.logger import logger
from src.exception import DatasetNotFoundError

class DataLoader:
    def __init__(self, file_path):
        self.file_path=file_path

    def load_data(self):
        try:
            df=pd.read_csv(self.file_path)

            logger.info(f"Dataset loaded successfully from {self.file_path}")

            print("Dataset loaded Successfully")

            return df
        except FileNotFoundError:
            logger.error(f"Dataset not found {self.file_path}")

            raise DatasetNotFoundError(
                f"Dataset not found at {self.file_path}"
            )
        except Exception as e:
            logger.exception(
                "Unexpected error while loading dataset."
            )
            raise DatasetNotFoundError(str(e))
