from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineering
from src.model_training import ModelTrainer
from src.logger import logger
from src.config import MODEL_PATH
from src.config import (
    RAW_DATA_PATH, 
    PROCESSED_DATA_PATH, 
    FEATURE_ENGINEERED_DATA_PATH,
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

# Feature Engineering Section

    feature_engineer=FeatureEngineering(preprocessor.df)
    feature_engineer.create_sqft_per_bedroom()
    feature_engineer.create_Bathroom_Per_Bedroom()
    feature_engineer.create_total_rooms()

# Model training Section
    trainer=ModelTrainer(feature_engineer.df)
    trainer.split_features_target()
    trainer.train_test_split_data()

    trainer.train_linear_regression()
    trainer.evaluate_linear_regression()

    trainer.train_decision_tree()
    trainer.evaluate_decision_tree()

    trainer.train_random_forest()
    trainer.evaluate_random_forest()

    trainer.train_k_neighbors()
    trainer.evaluate_k_neighbors()

    trainer.compare_models()

    trainer.save_best_model(MODEL_PATH)

# Save Feature engineer Dataset

    feature_engineer.save_feature_engineered_data(
        FEATURE_ENGINEERED_DATA_PATH
    )
    print("\nProject pipeline completed successfully!")

if __name__=="__main__":
    main()





