import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

# Define features and target
features = ['Exchange Rates', 'Population', 'GDP deflator', 'CPI']
target = 'Real GDP'

data = data.dropna(subset=[target])

# Impute missing values for features
imputer = SimpleImputer(strategy='mean')
data[features] = imputer.fit_transform(data[features])

# Retain Year in the dataset for plotting
X = data[['Year'] + features]  # Include Year for better x-axis labels
y = data['Real GDP']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extract Year column for plotting
years_test = X_test['Year']

# Drop Year from training and test features
X_train = X_train.drop(columns=['Year'])
X_test = X_test.drop(columns=['Year'])

# Define base models
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
gbm_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)

# Train base models
rf_model.fit(X_train, y_train)
gbm_model.fit(X_train, y_train)

# Predict with base models on the training set
rf_preds_train = rf_model.predict(X_train).reshape(-1, 1)
gbm_preds_train = gbm_model.predict(X_train).reshape(-1, 1)

# Stack predictions
stacked_train_preds = np.hstack([rf_preds_train, gbm_preds_train])

# Train final model on stacked predictions
final_model = RandomForestRegressor(n_estimators=50, random_state=42)
final_model.fit(stacked_train_preds, y_train)

# Predict with base models on the test set
rf_preds_test = rf_model.predict(X_test).reshape(-1, 1)
gbm_preds_test = gbm_model.predict(X_test).reshape(-1, 1)
stacked_test_preds = np.hstack([rf_preds_test, gbm_preds_test])

# Make final predictions
y_pred = final_model.predict(stacked_test_preds)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
# Calculate metrics for Hybrid Model
hybrid_rmse, hybrid_mae, hybrid_smape = calculate_metrics(y_test, y_pred)

print("\nHybrid Model Metrics:")
print(f"RMSE: {hybrid_rmse:.2f}")
print(f"MAE: {hybrid_mae:.2f}")
print(f"SMAPE: {hybrid_smape:.2f}%")
print(f"R² Score: {r2:.2f}")

# Visualization
plt.figure(figsize=(12, 7))

# Scatter plot for true values
plt.scatter(years_test, y_test, label="True Real GDP", color="blue", alpha=0.7, s=50)

# Scatter plot for predicted values
plt.scatter(years_test, y_pred, label="Predicted Real GDP", color="red", alpha=0.7, s=50, marker="x")

# Add error bars to show deviations
errors = abs(y_test - y_pred)
plt.errorbar(years_test, y_pred, yerr=errors, fmt='o', color='red', alpha=0.5, label="Error")

# Title with metrics
plt.title(f"Real GDP Prediction: Hybrid Model (Random Forest + XGBoost)\nMSE: {mse:.2f}, R²: {r2:.2f}", fontsize=14)

# Labels and legend
plt.xlabel("Year", fontsize=12)
plt.ylabel("Real GDP", fontsize=12)
plt.legend(fontsize=10)

# Grid
plt.grid(visible=True, linestyle="--", alpha=0.7)

# Show plot
plt.show()
