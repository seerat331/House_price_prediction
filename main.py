from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineering

from src.config import (
    RAW_DATA_PATH, 
    PROCESSED_DATA_PATH, 
    FEATURE_ENGINEERED_DATA_PATH
)
def main():

# DataLoader

    loader = DataLoader(RAW_DATA_PATH)
    df = loader.load_data()

# Preprocessing Section

    preprocessor = DataPreprocessor(df)
    preprocessor.dataset_info()
    preprocessor.check_missing_values()
    preprocessor.handle_missing_values()
    preprocessor.check_duplicates()
    preprocessor.remove_duplicates()
    preprocessor.encode_categories()

# Save Processed data

    preprocessor.save_processed_data(PROCESSED_DATA_PATH)
    print(df.columns)

# Feature Engineering Section

    feature_engineer=FeatureEngineering(preprocessor.df)
    feature_engineer.create_sqft_per_bedroom()
    feature_engineer.create_Bathroom_Per_Bedroom()
    feature_engineer.create_total_rooms()

# Save Feature engineer Dataset

    feature_engineer.save_feature_engineered_data(
        FEATURE_ENGINEERED_DATA_PATH
    )
    print("\nProject pipeline completed successfully!")

if __name__=="__main__":
    main()





