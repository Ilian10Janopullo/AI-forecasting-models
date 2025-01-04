import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Step 0: Load the dataset
file_path = 'Data Sources/Albania Information.csv'
data = pd.read_csv(file_path)

# Data Preparation
data = data.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

# Feature selection
features_to_use = ['Real GDP', 'GDP Growth Rate', 'Population', 'Population Growth Rate',
                   'Real GDP per capita', 'Real GDP per capita growth rate',
                   'GDP deflator', 'GDP deflator growth rates', 'CPI', 'CPI Growth Rate']
data = data[['Year'] + features_to_use]

# Handle very small values
data[features_to_use] = data[features_to_use].applymap(lambda x: max(x, 1e-2))

# Smooth features using a moving average
data[features_to_use] = data[features_to_use].rolling(window=3).mean().fillna(method='bfill')

# Step 1: Fit ARIMA Model
def fit_arima(data, order):
    arima_model = ARIMA(data, order=order)
    arima_fit = arima_model.fit()
    return arima_fit

# Define ARIMA parameters (p, d, q)
arima_order = (2, 1, 2)  # Adjust these values based on ACF/PACF analysis or testing
arima_fit = fit_arima(data['Real GDP'], arima_order)
arima_predictions = arima_fit.fittedvalues
residuals = data['Real GDP'] - arima_predictions

# Step 2: Prepare Residuals for GRU
# Convert residuals to sequences
time_steps = 10
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

residuals_smoothed = residuals.rolling(window=3).mean().fillna(method='bfill')
residuals_scaled = (residuals_smoothed - residuals_smoothed.mean()) / residuals_smoothed.std()
X, y = create_sequences(residuals_scaled.values, time_steps)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 3: Build and Train GRU Model
input_shape = (X_train.shape[1], 1)  # Reshape for GRU
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

gru_model = Sequential([
    GRU(16, return_sequences=False, input_shape=input_shape, kernel_regularizer='l2'),
    Dropout(0.4),
    Dense(8, activation='relu'),
    Dense(1, activation='linear')
])

gru_model.compile(optimizer='adam', loss='mse', metrics=['mape'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = gru_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16, callbacks=[early_stopping])

# Plot Training History
plt.figure(figsize=(14, 7))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Step 4: Combine Predictions
arima_test_predictions = arima_fit.forecast(steps=len(y_test))
gru_test_predictions = gru_model.predict(X_test).flatten()
final_predictions = arima_test_predictions + (gru_test_predictions * residuals.std() + residuals.mean())

# Evaluate
train_predictions = arima_predictions[:len(X_train)] + (gru_model.predict(X_train).flatten() * residuals.std() + residuals.mean())
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(data['Real GDP'][-len(final_predictions):], final_predictions))
train_mae = mean_absolute_error(y_train, train_predictions)
test_mae = mean_absolute_error(data['Real GDP'][-len(final_predictions):], final_predictions)
train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(data['Real GDP'][-len(final_predictions):], final_predictions)

print(f"Train RMSE: {train_rmse:.2f}, Train MAE: {train_mae:.2f}, Train R²: {train_r2:.2f}")
print(f"Test RMSE: {test_rmse:.2f}, Test MAE: {test_mae:.2f}, Test R²: {test_r2:.2f}")

# Visualize Final Predictions with Training Data
plt.figure(figsize=(14, 7))
plt.plot(data['Real GDP'], label='Actual', color='blue')
plt.plot(range(len(arima_predictions)), arima_predictions + residuals.mean(), label='ARIMA Predictions', color='green')
plt.plot(range(len(train_predictions)), train_predictions, label='Train Predictions (ARIMA + GRU)', color='orange')
plt.plot(range(len(data['Real GDP']) - len(final_predictions), len(data['Real GDP'])), final_predictions, label='Test Predictions (ARIMA + GRU)', color='red')
plt.legend()
plt.title('ARIMA-GRU Hybrid Model: Training and Testing Predictions')
plt.xlabel('Time')
plt.ylabel('Real GDP')
plt.show()
