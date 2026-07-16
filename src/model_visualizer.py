import matplotlib.pyplot as plt

from src.logger import logger
from src.exception import ModelTrainingError

class ModelVisualizer:
    def __init__(self, y_test, y_pred):
        self.y_test=y_test
        self.y_pred=y_pred

    def actual_vs_prediction(self, save_path):
        try:
            plt.figure(figsize=(8,6))

            plt.scatter(
                self.y_test,
                self.y_pred,
                color="dodgerblue",
                edgecolor="black",
                alpha=0.8
            )

            plt.plot(
                [self.y_test.min(), self.y_test.max()],
                [self.y_test.min(), self.y_test.max()],
                color="red",
                linewidth=2,
                linestyle="--"
            )

            plt.title(
                "Actual vs Predicted Home Prices",
                fontsize=16,
                fontweight="bold"
            )

            plt.xlabel("Actual Price")
            plt.ylabel("Predicted Price")

            plt.grid(
                linestyle="--",
                alpha=0.5
            )

            plt.tight_layout()

            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            print("\n"+"="*60)
            print("ACTUAL VS PREDICTED PLOT SAVED")
            print("="*60)
            print(f"Saved at: {save_path}")

            logger.info(
                f"Actual vs Predicted plot saved at {save_path}"
            )

        except Exception as e:
            logger.exception(
                "Failed to create Actual vs Predicted plot."
            )

            raise ModelTrainingError(
                f"Actual vs Predicted plot failed: {e}"
            )