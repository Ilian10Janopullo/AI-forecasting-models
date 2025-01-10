import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import GRU, Dense, Dropout #type: ignore
from tensorflow.keras.losses import Huber #type: ignore
from tensorflow.keras.regularizers import l2 #type: ignore
from tensorflow.keras.callbacks import EarlyStopping #type: ignore
from tensorflow.keras.activations import swish #type: ignore
import matplotlib.pyplot as plt
from DataPreparation import normalize, getMax, getMin
import os.path

# Load the dataset
if not os.path.exists("/Data Sources/Normalized_Albania_Information.csv"):
    normalize()

file_path = 'Data Sources/Normalized_Albania_Information.csv'
data = pd.read_csv(file_path)

# Data Preparation
# Handle missing values (e.g., interpolate numerical columns)
data = data.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

# Feature selection
features_to_use = ['Real GDP', 'GDP Growth Rate', 'Population', 'Population Growth Rate',
                   'Real GDP per capita', 'Real GDP per capita growth rate',
                   'GDP deflator', 'GDP deflator growth rates', 'CPI', 'CPI Growth Rate']
data = data[['Year'] + features_to_use]

# Manual scaling functions
real_gdp_min = getMin()
real_gdp_max = getMax()

def manual_inverse_scale(scaled_values, original_min, original_max):
    return scaled_values * (original_max - original_min) + original_min

# Create sequences for the GRU model
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, 1:])  # Use all features except 'Year'
        y.append(data[i + time_steps, 1])    # Predict 'Real GDP'
    return np.array(X), np.array(y)

# Combine Year and scaled features for sequence generation
combined_data = np.column_stack([data['Year'], data[features_to_use].values])

# Create sequences
time_steps = 10
X, y = create_sequences(combined_data, time_steps)

# Split into train and test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build the GRU Model
input_shape = (X_train.shape[1], X_train.shape[2])
model = Sequential([
    GRU(64, return_sequences=False, kernel_regularizer=l2(0.04), input_shape=input_shape),
    Dropout(0.323),
    Dense(32),
    Dense(16, activation=swish),
    Dense(8, activation='linear'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss=Huber(), metrics=['mape'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse scale predictions
train_predictions = manual_inverse_scale(train_predictions.flatten(), real_gdp_min, real_gdp_max)
test_predictions = manual_inverse_scale(test_predictions.flatten(), real_gdp_min, real_gdp_max)

y_train = manual_inverse_scale(y_train, real_gdp_min, real_gdp_max)
y_test = manual_inverse_scale(y_test, real_gdp_min, real_gdp_max)

# Compute metrics
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
train_mae = mean_absolute_error(y_train, train_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)

# SMAPE calculation
def calculate_smape(true, predicted):
    return np.mean(2 * np.abs(true - predicted) / (np.abs(true) + np.abs(predicted))) * 100

train_smape = calculate_smape(y_train, train_predictions)
test_smape = calculate_smape(y_test, test_predictions)

train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

# Print evaluation metrics
print(f"Train RMSE: {train_rmse:.2f}, Train MAE: {train_mae:.2f}, Train SMAPE: {train_smape:.2f}%, Train R²: {train_r2:.2f}")
print(f"Test RMSE: {test_rmse:.2f}, Test MAE: {test_mae:.2f}, Test SMAPE: {test_smape:.2f}%, Test R²: {test_r2:.2f}")

# Visualize the results
plt.figure(figsize=(14, 7))
plt.plot(data['Year'], manual_inverse_scale(data['Real GDP'], real_gdp_min, real_gdp_max), label='True Real GDP', color='blue')
plt.plot(data['Year'][time_steps:time_steps+len(train_predictions)], train_predictions, label='Train Predictions', color='green')
plt.plot(data['Year'][time_steps+len(train_predictions):time_steps+len(train_predictions)+len(test_predictions)], test_predictions, label='Test Predictions', color='red')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Real GDP')
plt.title('Real GDP Forecasting with GRU and Enhanced Features')
plt.show()

# Forecast future Real GDP
future_steps = 10
last_sequence = combined_data[-time_steps:, 1:]
future_predictions = []

for _ in range(future_steps):
    prediction = model.predict(last_sequence.reshape(1, time_steps, -1))
    future_predictions.append(prediction[0, 0])
    last_sequence = np.append(last_sequence[1:], [[prediction[0, 0]] + [0] * (len(features_to_use) - 1)], axis=0)

# Inverse scale future predictions
future_predictions = manual_inverse_scale(np.array(future_predictions), real_gdp_min, real_gdp_max)

# Generate future years
future_years = np.arange(data['Year'].iloc[-1] + 1, data['Year'].iloc[-1] + 1 + future_steps)

# Fit a trend using polynomial regression
trend_coeffs = np.polyfit(data['Year'], manual_inverse_scale(data['Real GDP'], real_gdp_min, real_gdp_max), deg=2)
trend_func = np.poly1d(trend_coeffs)

# Predict the trend for future years
future_trend = trend_func(future_years)

# Blend the GRU predictions with the trend
blended_forecast = 0.7 * future_trend + 0.3 * future_predictions

# Display forecasted data
forecasted_data = pd.DataFrame({'Year': future_years, 'Blended Forecasted Real GDP': blended_forecast})
print(forecasted_data)

# Visualize blended forecast
plt.figure(figsize=(10, 6))
plt.plot(forecasted_data['Year'], forecasted_data['Blended Forecasted Real GDP'], marker='o', label='Blended Predicted Real GDP')
plt.xlabel('Year')
plt.ylabel('Real GDP')
plt.title('Blended Predicted Real GDP (2035–2044)')
plt.legend()
plt.grid(True)
plt.show()