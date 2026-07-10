from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

# Load dataset
loader = DataLoader(RAW_DATA_PATH)
df = loader.load_data()

# Create preprocessing object
preprocessor = DataPreprocessor(df)

# Dataset information
preprocessor.dataset_info()

# Missing values
preprocessor.check_missing_values()
preprocessor.handle_missing_values()

# Duplicates
preprocessor.check_duplicates()
preprocessor.remove_duplicates()

# Encoding
preprocessor.encode_categories()

# Save processed dataset
preprocessor.save_processed_data(PROCESSED_DATA_PATH)