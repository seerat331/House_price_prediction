import joblib
import pandas as pd

from src.logger import logger
from src.exception import PredictionError
from src.feature_engineering import FeatureEngineering
from src.config import MODEL_PATH
class PredictionPipeline:
    def __init__(self):
        self.model=None

    def load_model(self):
        try:
        
            print("\n"+"="*60)
            print("LOADING TRAINED MODEL")
            print("="*60)
        
            self.model=joblib.load(MODEL_PATH)
        
            print("Model loaded successfully.")
        
            logger.info(
            "Best model loaded successfully."
            )
        except Exception as e:
            logger.exception(
                "Failed to load the model."
            )

            raise PredictionError(
                f"Model loading failed: {e}"
            )
        
    def create_input_dataframe(
            self,
            home,
            offers,
            sqft,
            bedrooms,
            bathrooms,
            brick,
            neighborhood

    ):
        try:
            brick_mapping={
                "No":0,
                "Yes":1
            }

            neighborhood_mapping={
                "East":0,
                "North":1,
                "West":2
            }

            brick=brick_mapping[brick]
            neighborhood=neighborhood_mapping[neighborhood]

            input_data=pd.DataFrame({
                "Home":[home],
                "SqFt":[sqft],
                "Bedrooms":[bedrooms],
                "Bathrooms":[bathrooms],
                "Offers":[offers],
                "Brick":[brick],
                "Neighborhood":[neighborhood]


            })

            print("\n" + "="*60)
            print("INPUT DATAFRAME CREATED")
            print("="*60)

            logger.info(
                "Input dataframe created successfully."
            )

            return input_data

            
        except Exception as e:
            logger.exception(
                "Failed to create input dataframe."
            )

            raise PredictionError(
                f"Input dataframe creation failed: {e}"
            )
        
    def apply_feature_engineering(self, input_data):
        try:
            feature_engineer=FeatureEngineering(input_data)
            feature_engineer.create_sqft_per_bedroom()
            feature_engineer.create_Bathroom_Per_Bedroom()
            feature_engineer.create_total_rooms()

            logger.info(
                "Feature Engineering applied successfully."
            )

            print("\n" + "="*60)
            print("FEATURE ENGINEERING APPLIED")
            print("="*60)

            return feature_engineer.df
        
        except Exception as e:
            logger.exception(
                "Failed to apply feature engineering"
            )

            raise PredictionError(
                f"Applying Feature Enginnering failed at {e}"
            )
        

    def predict(self, input_data):
        try:
            prediction=self.model.predict(input_data)
            print(f"Predicted House Price: {prediction[0]:.2f}")
            
            logger.info(
                "House price predicted successfully."
            )
            return prediction[0]
        
        except Exception as e:
            logger.exception(
                "Failed to predict house price."
            )

            raise PredictionError(
                f"Predicting house price failed {e}"
            )
        
    def predict_price(
            self,
            home,
            offers,
            sqft,
            bedrooms,
            bathrooms,
            brick,
            neighborhood
        ):
        try:
            self.load_model()
            input_data=self.create_input_dataframe(
                home,
                offers,
                sqft,
                bedrooms,
                bathrooms,
                brick,
                neighborhood
            )

            input_data=self.apply_feature_engineering(input_data)

            prediction=self.predict(input_data)

            logger.info("Prediction is done successfully. ")

            return prediction
        
        except Exception as e:
            logger.exception(
                "Failed to predict data."
            )

            raise PredictionError(
                f"Failed to predict data {e}"
            )

