import pandas as pd

from src.logger import logger 
from sklearn.preprocessing import LabelEncoder
from src.exception import DataProcessingError

class DataPreprocessor:
    def __init__(self, df):
        self.df=df

    def dataset_info(self):
        try:
            print("\n"+"="*60)
            print("DATASET INFORMATION")
            print("="*60)

            self.df.info()

            logger.info("Dataset information displayed successfully.")

        except Exception as e:
            logger.exception("Failed to display successfully.")

            raise DataProcessingError(
                f"Dataset information Failed: {e}"
            )
    
    def check_missing_values(self):
        try:
            print("\n"+"="*60)
            print("MISSING VALUES")
            print("="*60)

            print(self.df.isnull().sum())
            logger.info("Missing values checked successfully")

        except Exception as e:
            logger.exception(
                "Failed to check missing values."
            )
            raise DataProcessingError(
                f"Missing value check failed: {e}"
            )
        
    def handle_missing_values(self):
        try:
            numerical_columns=self.df.select_dtypes(
                include=["int64", "float64"]
            ).columns

            categorical_columns=self.df.select_dtypes(
                include=["object"]
            ).columns

            for column in numerical_columns:
                self.df[column]=self.df[column].fillna(
                    self.df[column].median()
                )
            
            for column in categorical_columns:
                self.df[column]=self.df[column].fillna(
                    self.df[column].mode()[0]
                )
            logger.info("Missing values handled successfully.")
            return self.df
        except Exception as e:
            logger.exception("Failed to handle missing values.")

            raise DataProcessingError(
                f"Missing value handling failed: {e}"
            )
    def check_duplicates(self):

        try:
            duplicates=self.df.duplicated().sum()

            print("\n"+"="*60)
            print("DUPLICATE ROWS")
            print("="*60)
            print(f"Duplicate Row:{duplicates}")

            logger.info("Duplicate rows checked successfully")

        except Exception as e:
            logger.exception("Failed to check the duplicate rows")

            raise DataProcessingError(
                f"Duplicate check failed: {e}"
            )
    def remove_duplicates(self):

        try:
            before=len(self.df)
            self.df.drop_duplicates(inplace=True)
            after=len(self.df)
            print(f"Removed {before - after} duplicate rows")
            logger.info("Duplicate rows removed successfully")
            return self.df
        except Exception as e:
            logger.exception("Failed to remove duplicate rows")
            raise DataProcessingError(
                f"Duplicate removal failed: {e}"
            )
        
    def encode_categories(self):
        try:
            encoder=LabelEncoder()
            categorical_columns=self.df.select_dtypes(
                include=["object"]
            ).columns
            for column in categorical_columns:
                self.df[column]=encoder.fit_transform(
                    self.df[column]
                )

            logger.info(
                "Categorical columns encoded successfully."
            )

            return self.df
        
        except Exception as e:
            logger.exception(
                "Encoding categorial columns failed"
            )
            raise DataProcessingError(
                f"Encoding failed: {e}"
            )
    def save_processed_data(self, save_path):
        try:
            self.df.to_csv(save_path, index=False)
            logger.info(
                f"Processed dataset saved successfully at {save_path}"
            )
            logger.info(f"\nProcessed dataset saved to:\n{save_path}")

        except Exception as e:
            logger.exception(
                "Failed to save the processed dataset."
            )
            raise DataProcessingError(
                f"Saving processed dataset failed: {e}"
            )
        

        