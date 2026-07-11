from src.logger import logger
from src.exception import DataProcessingError


class FeatureEngineering:
    def __init__(self, df):
        self.df=df
    def create_total_rooms(self):
        try:
            self.df["TotalRooms"]=self.df["Bedrooms"]+self.df["Bathrooms"]
            logger.info(
                "TotalRooms feature created successfully."
            )
            return self.df
        except Exception as e:
            logger.exception(
                "Failed to create TotalRooms feature."
            )
            raise DataProcessingError(
                f"TotalRooms creation failed: {e}"
            )
        
    def create_sqft_per_bedroom(self):
        try:
            self.df["SqFtPerBedroom"]=self.df["SqFt"]/self.df["Bedrooms"].replace(0, 1)
            logger.info(
                "SqFtPerBedroom feature created successfully."
            )
            return self.df
        except Exception as e:
            logger.exception(
                "Failed to create SqFtPerBedroom feature."
            )
            raise DataProcessingError(
                f"SqFtPerbedroom creation failed: {e}"
            )
    
    def create_Bathroom_Per_Bedroom(self):
        try:
            self.df["BathroomPerBedroom"]=self.df["Bathrooms"]/self.df["Bedrooms"].replace(0, 1)
            logger.info(
                "BathroomPerBedroom feature created successfuuly."
            )
            return self.df
        except Exception as e:
            logger.exception(
                "Failed to create BathroomPerBedroom feature."
            )
            raise DataProcessingError(
                f"BathroomPerBedroom creation failed: {e}"
            )
    
    def save_feature_engineered_data(self, save_path):
        try:
            self.df.to_csv(save_path, index=False)
            logger.info(
                f"Feature engineered dataset saved successfully at {save_path}"
            )
            print(f"\nFeature engineered dataset savedto:\n{save_path}")

        except Exception as e:
            logger.exception(
                "Failed to save feature engineered dataset."
            )
            raise DataProcessingError(
                f"Saving feature engineered dataset failed: {e}"
            )
        
        

        