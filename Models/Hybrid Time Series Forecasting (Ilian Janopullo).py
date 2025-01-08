import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import GRU, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.losses import Huber # type: ignore
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from DataPreparation import normalize, getMin, getMax
import os.path

# Step 0: Load the dataset
if(not os.path.exists("/Data Sources/Normalized_Albania_Information.csv")):
    normalize()

file_path = 'Data Sources/Normalized_Albania_Information.csv'
data = pd.read_csv(file_path)

real_gdp_min = getMin()
real_gdp_max = getMax()

# Feature selection
features_to_use = ['Real GDP', 'GDP Growth Rate', 'Population', 'Population Growth Rate',
                   'Real GDP per capita', 'Real GDP per capita growth rate',
                   'GDP deflator', 'GDP deflator growth rates', 'CPI', 'CPI Growth Rate']
data = data[['Year'] + features_to_use]

# Step 1: Fit ARIMA Model with Exogenous Variables
exog_features = ['GDP Growth Rate', 'Population', 'Population Growth Rate',
                 'Real GDP per capita', 'Real GDP per capita growth rate',
                 'GDP deflator', 'GDP deflator growth rates', 'CPI', 'CPI Growth Rate']
exog = data[exog_features]

exog.fillna(method='ffill', inplace=True)  # Forward fill
exog.fillna(method='bfill', inplace=True)  # Backward fill
exog.fillna(0, inplace=True)  # Replace NaNs with 0 (useful for percentages)

# Fit ARIMA Model
def fit_arima(data, order, exog=None):
    arima_model = ARIMA(data, order=order, exog=exog)
    arima_fit = arima_model.fit()
    return arima_fit

arima_order = (2, 1, 2)
arima_fit = fit_arima(data['Real GDP'], arima_order, exog=exog)
arima_predictions = arima_fit.fittedvalues

# Compute residuals
residuals = data['Real GDP'] - arima_predictions
residuals = residuals.iloc[1:]
residuals_scaled = (residuals - residuals.mean()) / residuals.std()

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
    GRU(32, return_sequences=True, input_shape=input_shape, kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    GRU(16, return_sequences=False, kernel_regularizer=l2(0.01)),
    Dense(8, activation='linear'),
    Dense(1, activation='linear')
])

gru_model.compile(optimizer='adam', loss=Huber(), metrics=['mape'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

history = gru_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=16, callbacks=[early_stopping, reduce_lr])

# Step 4: Combine Predictions using Meta-Model
arima_test_predictions = arima_fit.forecast(steps=len(y_test), exog=exog.iloc[-len(y_test):])
gru_test_predictions = gru_model.predict(X_test).flatten()

# Prepare meta-model inputs
meta_inputs = np.column_stack((arima_test_predictions, gru_test_predictions))
meta_model = LinearRegression()

# Train meta-model
meta_model.fit(meta_inputs, data['Real GDP'][-len(y_test):])

# Make final predictions
final_predictions = meta_model.predict(meta_inputs)

# Evaluate
train_predictions_arima = arima_predictions[:len(X_train)]
train_predictions_gru = gru_model.predict(X_train).flatten() * residuals.std() + residuals.mean()
meta_train_inputs = np.column_stack((train_predictions_arima, train_predictions_gru))
meta_train_predictions = meta_model.predict(meta_train_inputs)

train_rmse = np.sqrt(mean_squared_error(data['Real GDP'][:len(meta_train_predictions)], meta_train_predictions))
train_mae = mean_absolute_error(data['Real GDP'][:len(meta_train_predictions)], meta_train_predictions)
train_r2 = r2_score(data['Real GDP'][:len(meta_train_predictions)], meta_train_predictions)
train_smap = 100 * train_mae / np.mean(data['Real GDP'][:len(meta_train_predictions)])

test_rmse = np.sqrt(mean_squared_error(data['Real GDP'][-len(final_predictions):], final_predictions))
test_mae = mean_absolute_error(data['Real GDP'][-len(final_predictions):], final_predictions)
test_r2 = r2_score(data['Real GDP'][-len(final_predictions):], final_predictions)
test_smap = 100 * test_mae / np.mean(data['Real GDP'][-len(final_predictions):])

print(f"Train RMSE: {train_rmse:.2f}, Train MAE: {train_mae:.2f}, Train R²: {train_r2:.2f}, Train SMAPE: {train_smap:.2f}%")
print(f"Test RMSE: {test_rmse:.2f}, Test MAE: {test_mae:.2f}, Test R²: {test_r2:.2f}, Test SMAPE: {test_smap:.2f}%")

def manual_inverse_transform(scaled_values, original_min, original_max):
    return scaled_values * (original_max - original_min) + original_min

meta_train_predictions_original = manual_inverse_transform(arima_predictions[:len(X_train)], real_gdp_min, real_gdp_max)
final_predictions_original = manual_inverse_transform(final_predictions, real_gdp_min, real_gdp_max)

# Inverse transform Real GDP in the dataset
data['Real GDP'] = manual_inverse_transform(data['Real GDP'], real_gdp_min, real_gdp_max)

# Visualize Final Predictions with Training and Testing Information
plt.figure(figsize=(14, 7))
plt.plot(data['Year'], data['Real GDP'], label='Actual', color='blue')
plt.plot(data['Year'][:len(meta_train_predictions_original)], meta_train_predictions_original, label='Train Predictions', color='orange')
plt.plot(data['Year'][-len(final_predictions_original):], final_predictions_original, label='Test Predictions', color='red')
plt.legend()
plt.title('Optimized ARIMA-GRU Hybrid Model with Meta-Model: Training and Testing Predictions')
plt.xlabel('Year')
plt.ylabel('Real GDP')
plt.show()

# Step 5: Generate Future Forecasts
future_years = np.arange(data['Year'].iloc[-1] + 1, data['Year'].iloc[-1] + 11)
future_exog = np.tile(exog.iloc[-1:].values, (10, 1))  # Repeat last row of exog variables
arima_future_forecast = arima_fit.forecast(steps=10, exog=future_exog)

future_residuals = residuals_scaled[-time_steps:].values.reshape(1, time_steps, 1)
gru_future_forecast = []
for _ in range(10):
    future_prediction = gru_model.predict(future_residuals).flatten()[0]
    gru_future_forecast.append(future_prediction)
    future_residuals = np.append(future_residuals.flatten()[1:], [future_prediction]).reshape(1, time_steps, 1)

gru_future_forecast = np.array(gru_future_forecast)
future_meta_inputs = np.column_stack((arima_future_forecast, gru_future_forecast))
future_predictions = meta_model.predict(future_meta_inputs)

# Blended Forecast Calculation
future_trend = np.polyval(np.polyfit(data['Year'], data['Real GDP'], 2), future_years)
future_predictions_original = manual_inverse_transform(future_predictions, real_gdp_min, real_gdp_max)
combined_forecast = 0.7 * future_trend + 0.3 * future_predictions_original

# Display the Blended Forecast
forecast_next_10_years = np.column_stack((future_years, combined_forecast))
forecast_df = pd.DataFrame(forecast_next_10_years, columns=['Year', 'Blended Real GDP Forecast'])
print("\nBlended Forecast for the Next 10 Years:")
print(forecast_df)