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
        
    def residual_plot(self, save_path):
        try:
            residuals=self.y_test =self.y_pred

            plt.figure(figsize=(8, 6))

            plt.scatter(
                self.y_pred,
                residuals,
                color="mediumseagreen",
                edgecolor="black",
                alpha=0.8
            )

            plt.axhline(
                y=0,
                color="red",
                linestyle="--",
                linewidth=2
            )

            plt.title(
                "Residual Plot",
                fontsize=16,
                fontweight="bold"
            )

            plt.xlabel("Predicted Price")
            plt.ylabel("Residuals")

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
            print("RESIDUAL PLOT SAVED")
            print("="*60)
            print(f"Saved at: {save_path}")

            logger.info(
                f"Residual plot saved at {save_path}"
            )

        except Exception as e:
            logger.exception(
                "Failed to create residual plot."
            )

            raise ModelTrainingError(
                f"Residual plot failed: {e}"
            )
        
    def model_comparison(self, results, save_path):
        try:
            model_names=list(results.keys())
            r2_scores=[
                results[model]["R2 Score"]
                for model in model_names
            ]

            plt.figure(figsize=(8,6))

            bars=plt.bar(
                model_names,
                r2_scores,
                color="cornflowerblue",
                edgecolor="black"
            )

            plt.title(
                "Model Comparison (R2 Score)",
                fontsize=16,
                fontweight="bold"
            )

            plt.xlabel("Models")
            plt.ylabel("R2 Score")

            plt.ylim(0,1)

            plt.grid(
                axis="y",
                linestyle="--",
                alpha=0.5
            )

            for bar in bars:
                height=bar.get_height()

                plt.text(
                    bar.get_x()+bar.get_width()/2,
                    height+0.01,
                    f"{height:.3f}",
                    ha="center",
                    fontsize=10
                )

            plt.tight_layout()

            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            print("\n"+"="*60)
            print("MODEL COMPARISON PLOT SAVED")
            print("="*60)

            logger.info(
                f"Model comparison plot saved at {save_path}"
            )

        except Exception as e:
            logger.exception(
                "Failed to create model comparison plot."
            )

            raise ModelTrainingError(
                f"Model comparison plotting failed: {e}"
            )