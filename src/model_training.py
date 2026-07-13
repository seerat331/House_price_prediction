import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from src.logger import logger
from src.exception import ModelTrainingError
from src.config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
) 

# Creating Class
class ModelTrainer:
    def __init__(self, df):
        self.df=df

        self.X=None
        self.y=None

        self.X_train=None
        self.X_test=None

        self.y_train=None
        self.y_test=None

        self.model=None
        self.results={}

        self.best_model=None
        self.best_model_name=None

        self.models={}

# Spliting Features
    def split_features_target(self):
        try :
            self.X=self.df.drop(columns=[TARGET_COLUMN])
            self.y=self.df[TARGET_COLUMN]
            print("\n"+"="*60)
            print("FEATURES AND TARGET CREATED")
            print("\n"+"="*60)
            
            print(f"Features shape : {self.X.shape}")
            print(f"Target Shape   : {self.y.shape}")
            logger.info("Features and target created successfully.")

        except Exception as e:
            logger.exception(
                "Failed to split features and target."
            )

            raise ModelTrainingError(
                f"Feature or Target split failed: {e}"
            )

# Using train Test split
    def train_test_split_data(self):
        try:
            
            (self.X_train,
            self.X_test,
            self.y_train,
            self.y_test)=train_test_split(self.X, 
                                        self.y,  
                                        test_size=TEST_SIZE,
                                        random_state=RANDOM_STATE )
            
            print("\n"+"="*60)
            print("TRAIN TEST SPLIT")
            print("\n"+"="*60)

            print(f"X_train Shape : {self.X_train.shape}")
            print(f"X_test Shape  : {self.X_test.shape}")
            print(f"y_train Shape : {self.y_train.shape}")
            print(f"y_test Shape  : {self.y_test.shape}")

            logger.info(
                "Train-test split completed successfully."
            )

        except Exception as e:
            logger.exception(
                "Train-test split failed."
            )

            raise ModelTrainingError(
                f"Train-test split failed {e}"
            )
    
# Train Linear Regression Model
    def train_linear_regression(self):
        try:
            self.model=LinearRegression()
            self.model.fit(
                self.X_train,
                self.y_train
            )

            self.models["Linear Regression"]=self.model

            print("\n"+"="*60)
            print("LINEAR REGRESSION MODEL TRAINED.")
            print("="*60)

            logger.info(
                "Linear Regression model trained successfully."
            )

        except Exception as e:
            logger.exception(
                "Failed to train Linear Regression."
            )

            raise ModelTrainingError(
                f"Linear Regression training failed: {e}"
            )
        
# Evaluate the Linear Regression Model
    def evaluate_linear_regression(self):
        try:
            y_pred=self.model.predict(self.X_test)
            mae=mean_absolute_error(
                self.y_test,
                y_pred 
            )
            mse=mean_squared_error(
                self.y_test,
                y_pred
            )
            rmse=np.sqrt(mse)

            r2=r2_score(
                self.y_test,
                y_pred)
            
            self.results["Linear Regression"]={
                "MAE":mae,
                "MSE":mse,
                "RMSE":rmse,
                "R2 Score":r2
            }

            print("\n" + "=" * 60)
            print("LINEAR REGRESSION RESULTS")
            print("=" * 60)

            for metric, value in self.results["Linear Regression"].items():
                print(f"{metric}: {value:.4f}")
            logger.info("Linear Regression evaluated successfully.")

        except Exception as e:
            logger.exception(
                "Failed to evaluate Linear Regression model."
            )

            raise ModelTrainingError(
                f"Linear Regression evaluation failed: {e}"
            )
        
        print("\n" + "=" * 60)
        print("MODEL EVALUATION COMPLETED")
        print("=" * 60)
        
# Train Decision Tree Model
    def train_decision_tree(self):
        try:
            self.model=DecisionTreeRegressor(
                random_state=RANDOM_STATE
            )
            self.model.fit(
                self.X_train,
                self.y_train
                
            )

            self.models["Decision Tree"]=self.model

            print("\n"+"="*60)
            print("DECISION TREE REGRESSION MODEL TRAINED.")
            print("="*60)
            
            logger.info(
                "Decision tree regression model trained successfully."
            )

        except Exception as e:
            logger.exception(
            "Failed to train the Decision Regressor model"
            )

            raise ModelTrainingError(
                f"Failed to train Decision Regressor Model: {e}"
            )
    
# Evaluate Decision Tree Model
    def evaluate_decision_tree(self):
        try:
            y_pred=self.model.predict(self.X_test)
            mae=mean_absolute_error(
                self.y_test,
                y_pred 
            )
            mse=mean_squared_error(
                self.y_test,
                y_pred
            )
            rmse=np.sqrt(mse)

            r2=r2_score(
                self.y_test,
                y_pred)
            
            self.results["Decision Tree"]={
                "MAE":mae,
                "MSE":mse,
                "RMSE":rmse,
                "R2 Score":r2
            }

            print("\n" + "=" * 60)
            print("DECISION TREE RESULTS")
            print("=" * 60)

            for metric, value in self.results["Decision Tree"].items():
                 print(f"{metric}: {value:.4f}")
            logger.info("Decision Tree evaluated successfully.")

        except Exception as e:
            logger.exception(
                "Failed to evaluate Decision Tree Regression model."
            )

            raise ModelTrainingError(
                f"Decision Tree Regression evaluation failed: {e}"
            )
        
        print("\n" + "=" * 60)
        print("MODEL EVALUATION COMPLETED")
        print("=" * 60)
        
# Train Random Forest Model
    def train_random_forest(self):
        try:
            self.model=RandomForestRegressor(
                random_state=RANDOM_STATE
            )
            self.model.fit(
                self.X_train,
                self.y_train
            )

            self.models["Random Forest"]=self.model

            print("\n"+"="*60)
            print("RANDOM FOREST REGRESSION MODEL TRAINED.")
            print("="*60)
            
            logger.info(
                "Random Forest regression model trained successfully."
            )

        except Exception as e:
            logger.exception(
            "Failed to train the Random Forest Regressor model"
            )

            raise ModelTrainingError(
                f"Failed to train Random Forest Regressor Model: {e}"
            )
        
# Evaluate Random Forest Model
    def evaluate_random_forest(self):
        try:
            y_pred=self.model.predict(self.X_test)
            mae=mean_absolute_error(
                self.y_test,
                y_pred 
            )
            mse=mean_squared_error(
                self.y_test,
                y_pred
            )
            rmse=np.sqrt(mse)

            r2=r2_score(
                self.y_test,
                y_pred)
            
            self.results["Random Forest"]={
                "MAE":mae,
                "MSE":mse,
                "RMSE":rmse,
                "R2 Score":r2
            }

            print("\n" + "=" * 60)
            print("RANDOM FOREST REGRESSION RESULTS")
            print("=" * 60)

            for metric, value in self.results["Random Forest"].items():
                print(f"{metric}: {value:.4f}")
            logger.info("Random Forest Regressor evaluated successfully.")

        except Exception as e:
            logger.exception(
                "Failed to evaluate Random Forest Regression model."
            )

            raise ModelTrainingError(
                f"Random Forest Regression evaluation failed: {e}"
            )
        
        print("\n" + "=" * 60)
        print("MODEL EVALUATION COMPLETED")
        print("=" * 60)
        
# Train Kneighbors Regressor Model
    def train_k_neighbors(self):
        try:
            self.model=KNeighborsRegressor(
                n_neighbors=5
            )
            self.model.fit(
                self.X_train,
                self.y_train
            )

            self.models["KNN"]=self.model

            print("\n"+"="*60)
            print("KNEIGHBORSREGRESSOR MODEL TRAINED.")
            print("="*60)
            
            logger.info(
                "KNN model trained successfully."
            )

        except Exception as e:
            logger.exception(
            "Failed to train the KNN model"
            )

            raise ModelTrainingError(
                f"Failed to train KNN Model: {e}"
            )
        
# Evaluate Kneighbors Regressor Model
    def evaluate_k_neighbors(self):
        try:
            y_pred=self.model.predict(self.X_test)
            mae=mean_absolute_error(
                self.y_test,
                y_pred 
            )
            mse=mean_squared_error(
                self.y_test,
                y_pred
            )
            rmse=np.sqrt(mse)

            r2=r2_score(
                self.y_test,
                y_pred)
            
            self.results["KNN"]={
                "MAE":mae,
                "MSE":mse,
               "RMSE":rmse,
                "R2 Score":r2
            }

            print("\n" + "=" * 60)
            print("KNN REGRESSION RESULTS")
            print("=" * 60)

            for metric, value in self.results["KNN"].items():
                print(f"{metric}: {value:.4f}")

            logger.info("KNN Regression evaluated successfully.")

        except Exception as e:
            logger.exception(
                "Failed to evaluate KNN model."
            )

            raise ModelTrainingError(
                f"KNN evaluation failed: {e}"
            )
        
        print("\n" + "=" * 60)
        print("MODEL EVALUATION COMPLETED")
        print("=" * 60)

# Compare Models
    def compare_models(self):
        try:
            print("\n"+"="*60)
            print("MODEL COMPARISON")
            print("="*60)

            best_r2=float("-inf")
            best_model_name=None

            for model_name, metrics in self.results.items():
                print(f"\nModel: {model_name}")

                for metric, value in metrics.items():
                    print(f"{metric}: {value:.4f}")

                if metrics["R2 Score"]>best_r2:
                    best_r2=metrics["R2 Score"]
                    best_model_name=model_name

            self.best_model_name=best_model_name
            self.best_model=self.models[best_model_name]

            print("\n"+"="*60)
            print(f"Best Model : {self.best_model_name}")
            print(f"Best R2 Score    : {best_r2:.4f}")
            print("="*60)

            logger.info(
                f"Best model selected: {self.best_model_name}"
            )

        except Exception as e:
            logger.exception(
                "Failed to compare model."
            )

            raise ModelTrainingError(
                f"Failed to compare model: {e}"
            )

# Save Best Model 
    def save_best_model(self, save_path):
        try:
            print("\n"+"="*60)
            print("SAVE BEST MODEL")
            print("="*60)

            joblib.dump(
                self.best_model,
                save_path
            )

            print("\n"+"="*60)
            print("BEST MODEL SAVED SUCCESSFULLY")
            print("="*60)
            print(f"Model Name : {self.best_model_name}")
            print(f"Saved To   : {save_path}")

            logger.info(
                f"Best model saved successfully at {save_path}"
            )

        except Exception as e:
            logger.exception(
                "Failed to save the best model."
            )
            raise ModelTrainingError(
                f"Failed to save best model: {e}"
            )
        
