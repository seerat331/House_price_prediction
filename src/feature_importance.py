import matplotlib.pyplot as plt
from src.logger import logger
from src.exception import ModelTrainingError

class FeatureImportance:
    def __init__(self, model, feature_names):
        self.model=model
        self.feature_names=feature_names

    def plot(self, save_path):
        try:
            importance=self.model.feature_importances_
            plt.figure(figsize=(10, 4))
            plt.barh(
                self.feature_names,
                importance
            )
            plt.title("Feature Importance")
            plt.xlabel("features")
            plt.ylabel("Importance")

            plt.xticks(rotation=45)

            plt.tight_layout()

            plt.savefig(save_path)

            plt.close()

            print("\n"+"="*60)
            print("FEATURE IMPORTANCE PLOT SAVED")
            print("="*60)
            print(f"Saved at: {save_path}")
            
            logger.info(
                f"Feature Importance plot saved at {save_path}"
            )

        except Exception as e:
            logger.exception("Failed to create feature importance plot.")

            raise ModelTrainingError(
                f"Feature importance plotting failed: {e}"
            )