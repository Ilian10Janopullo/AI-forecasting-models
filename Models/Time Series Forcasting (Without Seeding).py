import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import GRU, Dense, Dropout # type: ignore
from tensorflow.keras.losses import Huber # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Step 1: Data Preparation
file_path = 'Data Sources/RealGDP.csv' 
data = pd.read_csv(file_path, encoding='latin1', skiprows=11)

# Set the first row as header
data.columns = data.iloc[0]
data = data[1:]

# Filter the "Albania" data
global_gdp = data[data['Country'] == 'Albania']

# Drop the 'Country' column and format
global_gdp = global_gdp.drop(columns=['Country']).reset_index(drop=True)

# Transpose the data to have years as rows
global_gdp = global_gdp.T
global_gdp.columns = ['GDP']

# Extract numeric years and filter invalid rows
global_gdp = global_gdp.reset_index()  # Reset index to make years a column
global_gdp.columns = ['Year', 'GDP']  # Rename columns for clarity
global_gdp['Year'] = global_gdp['Year'].str.extract(r'(\d+)')  # Extract numeric parts from 'Year'
global_gdp = global_gdp.dropna(subset=['Year'])  # Drop rows where 'Year' is NaN
global_gdp['Year'] = global_gdp['Year'].astype(int)  # Convert 'Year' to integers

# Convert GDP to numeric and drop rows with invalid GDP values
global_gdp['GDP'] = pd.to_numeric(global_gdp['GDP'], errors='coerce')
global_gdp = global_gdp.dropna(subset=['GDP'])  # Drop rows with NaN GDP values

# Handle very small GDP values
global_gdp['GDP'] = global_gdp['GDP'].apply(lambda x: max(x, 1e-2))  # Replace very small values with a threshold

# Decompose the time series into components
decomposition = seasonal_decompose(global_gdp['GDP'], model='multiplicative', period=1)
global_gdp['Trend'] = decomposition.trend
global_gdp['Trend'] = global_gdp['Trend'].fillna(method='bfill').fillna(method='ffill')  # Fill missing trend values

# Smooth the data using a moving average
global_gdp['Trend'] = global_gdp['Trend'].rolling(window=3).mean().fillna(method='bfill')

# Apply log transformation to stabilize variance on the trend component
global_gdp['Trend'] = np.log1p(global_gdp['Trend'])

# Step 2: Calculate and Smooth Growth Rates
global_gdp['GrowthRate'] = global_gdp['GDP'].pct_change()  # Calculate year-over-year growth rate
global_gdp['GrowthRate'] = global_gdp['GrowthRate'].rolling(window=3).mean().fillna(method='bfill')  # Smooth growth rate
global_gdp = global_gdp.dropna()  # Remove NaN values after smoothing

# Combine trend and growth rate into features
features = global_gdp[['Trend', 'GrowthRate']].values

# Step 3: Preprocess Data
scaler = RobustScaler()  # Scale the features
scaled_features = scaler.fit_transform(features)

# Create sequences for trend and growth rate
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, :])  # Include all features
        y.append(data[i + time_steps, 0])  # Predict GDP Trend only
    return np.array(X), np.array(y)

time_steps = 10
X, y = create_sequences(scaled_features, time_steps)

# Step 4: Split into Train and Test Sets
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 5: Build the GRU Model
input_shape = (X_train.shape[1], X_train.shape[2])  # time_steps, num_features
model = Sequential([
    GRU(40, return_sequences=True, kernel_regularizer=l2(0.01), input_shape=input_shape),
    Dropout(0.2),
    GRU(40, return_sequences=False, kernel_regularizer=l2(0.01)),
    Dropout(0.29),
    Dense(50),
    Dense(1)  # Predict GDP Trend
])

# Compile the model with Huber loss and MAPE
model.compile(optimizer='adam', loss=Huber(), metrics=['mape'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Step 6: Evaluate the Model
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Reverse scaling
train_predictions = scaler.inverse_transform(np.concatenate([train_predictions, np.zeros_like(train_predictions)], axis=1))[:, 0]
test_predictions = scaler.inverse_transform(np.concatenate([test_predictions, np.zeros_like(test_predictions)], axis=1))[:, 0]
train_predictions = np.expm1(train_predictions)
test_predictions = np.expm1(test_predictions)
y_train = scaler.inverse_transform(np.concatenate([y_train.reshape(-1, 1), np.zeros_like(y_train.reshape(-1, 1))], axis=1))[:, 0]
y_test = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), np.zeros_like(y_test.reshape(-1, 1))], axis=1))[:, 0]
y_train = np.expm1(y_train)
y_test = np.expm1(y_test)

# Compute metrics
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
train_mae = mean_absolute_error(y_train, train_predictions)
test_mae = mean_absolute_error(y_test, test_predictions)

def calculate_smape(true, predicted):
    return np.mean(2 * np.abs(true - predicted) / (np.abs(true) + np.abs(predicted))) * 100

train_smape = calculate_smape(y_train, train_predictions)
test_smape = calculate_smape(y_test, test_predictions)

train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

# Print results
print(f"Train RMSE: {train_rmse:.2f}, Train MAE: {train_mae:.2f}, Train SMAPE: {train_smape:.2f}%, Train R²: {train_r2:.2f}")
print(f"Test RMSE: {test_rmse:.2f}, Test MAE: {test_mae:.2f}, Test SMAPE: {test_smape:.2f}%, Test R²: {test_r2:.2f}")

# Visualize the results
plt.figure(figsize=(14, 7))
plt.plot(global_gdp['Year'], np.expm1(global_gdp['Trend']), label='True GDP Trend', color='blue')
plt.plot(global_gdp['Year'][time_steps:time_steps+len(train_predictions)], train_predictions, label='Train Predictions', color='green')
plt.plot(global_gdp['Year'][time_steps+len(train_predictions):time_steps+len(train_predictions)+len(test_predictions)], test_predictions, label='Test Predictions', color='red')
plt.legend()
plt.xlabel('Year')
plt.ylabel('GDP Trend')
plt.title('GDP Trend Forecasting with GRU and Smoothed Growth Rate Feature')
plt.show()

# Step 7: Forecast Future GDP Trend
future_steps = 10
last_sequence = scaled_features[-time_steps:]
future_predictions = []

for _ in range(future_steps):
    prediction = model.predict(last_sequence.reshape(1, time_steps, -1))
    future_predictions.append(prediction[0, 0])
    last_sequence = np.append(last_sequence[1:], [[prediction[0, 0], 0]], axis=0)

# Add placeholder for second feature (GrowthRate) and inverse transform
future_predictions_with_placeholder = np.concatenate(
    [np.array(future_predictions).reshape(-1, 1), np.zeros_like(np.array(future_predictions).reshape(-1, 1))], axis=1
)
future_predictions = scaler.inverse_transform(future_predictions_with_placeholder)[:, 0]

# Reverse the log transformation
future_predictions = np.expm1(future_predictions)

# Generate future years
future_years = np.arange(global_gdp['Year'].iloc[-1] + 1, global_gdp['Year'].iloc[-1] + 1 + future_steps)

# Display forecasted GDP Trend
forecasted_data = pd.DataFrame({'Year': future_years, 'Forecasted GDP Trend': future_predictions.flatten()})
print(forecasted_data)
