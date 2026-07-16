# House Price Prediction using Machine Learning

A complete end-to-end Machine Learning project that predicts house prices based on various property features. This project demonstrates the entire ML workflow, including data preprocessing, feature engineering, model training, evaluation, visualization, and prediction.

---

## Project Overview

The objective of this project is to build a regression model capable of predicting house prices accurately using historical housing data.

The project follows a modular software engineering approach, making the code clean, reusable, and easy to maintain.

---

## Project Structure

```
House Price Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── logs/
│
├── models/
│
├── outputs/
│   ├── figures/
│   ├── prediction/
│   └── reports/
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── feature_importance.py
│   ├── model_visualizer.py
│   ├── prediction_pipeline.py
│   ├── logger.py
│   └── exception.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Features

- Load raw dataset
- Data preprocessing
  - Missing value handling
  - Duplicate removal
  - Categorical encoding
- Feature engineering
  - Total Rooms
  - Square Feet per Bedroom
  - Bathroom per Bedroom
- Train multiple regression models
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - K-Nearest Neighbors Regressor
- Model evaluation using:
  - MAE
  - MSE
  - RMSE
  - R² Score
- Automatic best model selection
- Save trained model
- Prediction pipeline
- Feature importance visualization
- Actual vs Predicted plot
- Residual plot
- Model comparison plot
- Logging
- Custom exception handling

---

## Dataset

The dataset contains housing information such as:

- Home ID
- Price
- Square Feet
- Bedrooms
- Bathrooms
- Offers
- Brick
- Neighborhood

Target Variable:

- **Price**

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

---

## Machine Learning Workflow

1. Load Dataset
2. Data Cleaning
3. Feature Engineering
4. Train-Test Split
5. Model Training
6. Model Evaluation
7. Best Model Selection
8. Save Model
9. Generate Visualizations
10. Predict House Price

---

## Evaluation Metrics

The following metrics are used to compare model performance:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Visualizations

The project automatically generates:

- Feature Importance
- Actual vs Predicted Values
- Residual Plot
- Model Comparison Chart

These visualizations are saved inside:

```
outputs/figures/
```

---

## ▶️ How to Run

### Clone Repository

```bash
git clone https://github.com/yourusername/house-price-prediction.git
```

### Navigate to Project

```bash
cd house-price-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Project

```bash
python main.py
```

---

## Sample Prediction

```python
prediction_pipeline.predict_price(
    home=101,
    sqft=2500,
    bedrooms=4,
    bathrooms=3,
    offers=2,
    brick="Yes",
    neighborhood="East"
)
```

Example Output:

```
Predicted House Price: 182635.68
```

---

## Logging

Project execution logs are automatically stored inside:

```
logs/project.log
```

---

## Future Improvements

- Hyperparameter tuning using GridSearchCV
- Cross-validation
- Model deployment using Flask or FastAPI
- Interactive Streamlit web application
- Docker support
- CI/CD pipeline
- Cloud deployment

---

## Author

**Seerat Zahra**

- Python Developer
- Machine Learning Engineer
- AI Enthusiast

---

##  If you found this project useful, consider giving it a star.