import matplotlib.pyplot as plt
from src.logger import logger
from src.exception import ModelTrainingError

class FeatureImportance:
    def __init__(self, model, feature_names):
        self.model=model
        self.feature_names=feature_names

    def plot(self, save_path):
        try:
            importances=self.model.feature_importances_

            sorted_indices=importances.argsort()

            sorted_features=[
                self.feature_names[i]
                for i in sorted_indices
            ]

            sorted_importances=importances[sorted_indices]

            plt.figure(figsize=(10, 4))
            bars=plt.barh(
                sorted_features,
                sorted_importances,
                color="skyblue",
                edgecolor="black"
                )
            plt.title("Feature Importance",
                      fontsize=16,
                      fontweight="bold"
                      )
            plt.xlabel("features",
                       fontsize=12)
            plt.ylabel("Importance",
                       fontsize=12)
            
            plt.grid(
                axis="x",
                linestyle="--",
                alpha=0.5
            )

            for bar in bars:
                width=bar.get_width()

                plt.text(
                    width+0.002,
                    bar.get_y()+bar.get_height()/2,
                    f"[width:.3f]",
                    va="center",
                    fontsize=10
                )
            plt.tight_layout()

            plt.savefig(save_path,
                        dpi=300,
                        bbox_inches="tight")

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