import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calculate_metrics(y_true, y_pred):
    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # SMAPE
    smape = 100 * np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))
    
    return rmse, mae, smape


# Load the dataset
file_path = '../Data Sources/Normalized_Albania_Information.csv'
data = pd.read_csv(file_path)

# Selecting relevant columns for analysis
features = ['Exchange Rates', 'Population', 'GDP deflator', 'CPI']
target = 'Real GDP'

data = data.dropna(subset=[target])

# Retain Year column for plotting
X = data[['Year'] + features]  # Include Year for better x-axis labels
y = data['Real GDP']

# Preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),  # Replace missing values with the mean
    ("scaler", StandardScaler())  # Standardize features to z-scores
])

# Apply preprocessing to features (excluding Year)
X.loc[:, features] = preprocessing_pipeline.fit_transform(X[features])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extract Year column for plotting
years_test = X_test['Year']

# Drop Year column for training
X_train = X_train.drop(columns=['Year'])
X_test = X_test.drop(columns=['Year'])

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'kernel': ['rbf'],  # Radial Basis Function for capturing non-linear relationships
    'C': [0.1, 1.0, 10],  # Regularization parameter
    'epsilon': [0.01, 0.1, 0.2]  # Tolerance margin for error
}
svr = SVR()
grid_search = GridSearchCV(svr, param_grid, scoring='r2', cv=5)
grid_search.fit(X_train, y_train)

# Best model
svm_regressor = grid_search.best_estimator_

# Predictions and evaluation
train_pred = svm_regressor.predict(X_train)
test_pred = svm_regressor.predict(X_test)
mse = mean_squared_error(y_test, test_pred)
r2 = r2_score(y_test, test_pred)

print(f"Best Model Parameters: {grid_search.best_params_}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Calculate metrics for SVM
svm_rmse, svm_mae, svm_smape = calculate_metrics(y_test, test_pred)

print("\nSVM Model Metrics:")
print(f"RMSE: {svm_rmse:.2f}")
print(f"MAE: {svm_mae:.2f}")
print(f"SMAPE: {svm_smape:.2f}%")
print(f"R² Score: {r2:.2f}")


# Visualization
plt.figure(figsize=(12, 7))

# Plot true values for train and test sets using Year as X-axis
plt.plot(data['Year'][:len(y_train)], y_train, label="True Train Data", color="blue", linestyle="--")
plt.plot(data['Year'][len(y_train):], y_test, label="True Test Data", color="cyan", linestyle="--")

# Plot predictions for train and test sets using Year as X-axis
plt.scatter(data['Year'][:len(y_train)], train_pred, label="Train Predictions", color="green", s=15, alpha=0.7)
plt.scatter(data['Year'][len(y_train):], test_pred, label="Test Predictions", color="red", s=40, marker="x")

# Add MSE and R² in the title
plt.title(f"Real GDP Prediction: SVM Model (RBF Kernel)\nMSE: {mse:.2f}, R²: {r2:.2f}", fontsize=14)

# Updated labels
plt.xlabel("Year", fontsize=12)
plt.ylabel("Real GDP", fontsize=12)
plt.legend(fontsize=10)

# Grid
plt.grid(visible=True, linestyle="--", alpha=0.7)

plt.show()
