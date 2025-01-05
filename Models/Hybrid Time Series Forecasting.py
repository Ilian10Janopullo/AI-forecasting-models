import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import GRU, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.losses import Huber # type: ignore
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

# Step 1: Fit ARIMA Model with Exogenous Variables
exog_features = ['GDP Growth Rate', 'Population', 'Population Growth Rate',
                 'Real GDP per capita', 'Real GDP per capita growth rate',
                 'GDP deflator', 'GDP deflator growth rates', 'CPI', 'CPI Growth Rate']
exog = data[exog_features]

# Fit ARIMA Model
def fit_arima(data, order, exog=None):
    arima_model = ARIMA(data, order=order, exog=exog)
    arima_fit = arima_model.fit()
    return arima_fit

arima_order = (2, 1, 3)
arima_fit = fit_arima(data['Real GDP'], arima_order, exog=exog)
arima_predictions = arima_fit.fittedvalues

# Compute residuals
residuals = data['Real GDP'] - arima_predictions
residuals_smoothed = residuals.rolling(window=20).mean().fillna(method='bfill')
residuals_scaled = (residuals_smoothed - residuals_smoothed.mean()) / residuals_smoothed.std()

# Step 2: Prepare Residuals for GRU
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 5
X, y = create_sequences(residuals_scaled.values, time_steps)

# Train-test split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Normalize data for GRU stability
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, time_steps)).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, time_steps)).reshape(X_test.shape)

# Step 3: Build and Train GRU Model
input_shape = (X_train.shape[1], 1)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

gru_model = Sequential([
    GRU(64, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.05)),
    Dropout(0.4),
    GRU(32, return_sequences=False, kernel_regularizer=l2(0.05)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

gru_model.compile(optimizer='adam', loss=Huber(), metrics=['mape'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

history = gru_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=16, callbacks=[early_stopping, reduce_lr])

# Step 4: Combine Predictions
def dynamic_combination(arima_preds, gru_preds, residuals):
    alpha = 1 - (np.abs(residuals).mean() / np.abs(arima_preds).mean())
    beta = 1 - alpha
    return alpha * arima_preds + beta * gru_preds

arima_test_predictions = arima_fit.forecast(steps=len(y_test), exog=exog.iloc[-len(y_test):])
gru_test_predictions = gru_model.predict(X_test).flatten()

final_predictions = dynamic_combination(arima_test_predictions, gru_test_predictions * residuals.std() + residuals.mean(), residuals)

# Evaluate
train_predictions_arima = arima_predictions[:len(X_train)]
train_predictions_gru = gru_model.predict(X_train).flatten() * residuals.std() + residuals.mean()
train_predictions = dynamic_combination(train_predictions_arima, train_predictions_gru, residuals[:len(X_train)])

valid_indices = range(len(train_predictions))
train_rmse = np.sqrt(mean_squared_error(y_train[valid_indices], train_predictions[valid_indices]))
train_mae = mean_absolute_error(y_train[valid_indices], train_predictions[valid_indices])
train_r2 = r2_score(y_train[valid_indices], train_predictions[valid_indices])

test_rmse = np.sqrt(mean_squared_error(data['Real GDP'][-len(final_predictions):], final_predictions))
test_mae = mean_absolute_error(data['Real GDP'][-len(final_predictions):], final_predictions)
test_r2 = r2_score(data['Real GDP'][-len(final_predictions):], final_predictions)

print(f"Train RMSE: {train_rmse:.2f}, Train MAE: {train_mae:.2f}, Train R²: {train_r2:.2f}")
print(f"Test RMSE: {test_rmse:.2f}, Test MAE: {test_mae:.2f}, Test R²: {test_r2:.2f}")

# Visualize Final Predictions with Training Data
plt.figure(figsize=(14, 7))
plt.plot(data['Year'], data['Real GDP'], label='Actual', color='blue')
plt.plot(data['Year'][:len(arima_predictions)], arima_predictions + residuals.mean(), label='ARIMA Predictions', color='green')
plt.plot(data['Year'][:len(train_predictions)], train_predictions, label='Train Predictions (ARIMA + GRU)', color='orange')
plt.plot(data['Year'][-len(final_predictions):], final_predictions, label='Test Predictions (ARIMA + GRU)', color='red')
plt.legend()
plt.title('Optimized ARIMA-GRU Hybrid Model with Exogenous Variables: Training and Testing Predictions')
plt.xlabel('Year')
plt.ylabel('Real GDP')
plt.show()
