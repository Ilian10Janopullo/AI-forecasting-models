import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Step 1: Data Preparation
file_path = 'Data Sources/RealGDP.csv'  # Ensure this path is correct
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

# Check for remaining NaN values
if global_gdp['Trend'].isnull().sum() > 0:
    raise ValueError("NaN values remain in the Trend column after preprocessing.")

# Step 2: Data Transformation for GRU
trend_values = global_gdp['Trend'].values.reshape(-1, 1)
scaler = RobustScaler()  # Use RobustScaler to reduce sensitivity to outliers
trend_scaled = scaler.fit_transform(trend_values)

# Create sequences
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 10
X, y = create_sequences(trend_scaled, time_steps)

# Check for NaN in sequences
if np.isnan(X).sum() > 0 or np.isnan(y).sum() > 0:
    raise ValueError("NaN values found in sequences. Check preprocessing steps.")

# Split into train and test sets
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 3: Build and Train the GRU Model with Regularization and Early Stopping
model = Sequential([
    GRU(300, return_sequences=True, kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1], 1)),
    Dropout(0.3),
    GRU(300, return_sequences=False, kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(100),
    Dense(1)
])

# Compile the model with Huber loss
model.compile(optimizer='adam', loss=Huber())

# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=150,  # Increased epochs
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

# Step 4: Evaluate the Model
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Reverse log transformation and scaling
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
train_predictions = np.expm1(train_predictions)
test_predictions = np.expm1(test_predictions)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
y_train = np.expm1(y_train)
y_test = np.expm1(y_test)

# Compute RMSE, MAE, SMAPE, and R²
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
plt.plot(global_gdp['Year'], np.expm1(trend_values), label='True GDP Trend', color='blue')
plt.plot(global_gdp['Year'][time_steps:time_steps+len(train_predictions)], train_predictions, label='Train Predictions', color='green')
plt.plot(global_gdp['Year'][time_steps+len(train_predictions):time_steps+len(train_predictions)+len(test_predictions)], test_predictions, label='Test Predictions', color='red')
plt.legend()
plt.xlabel('Year')
plt.ylabel('GDP Trend')
plt.title('GDP Trend Forecasting for Albania')
plt.show()

# Step 5: Forecast Future GDP Trend
future_steps = 10
last_sequence = trend_scaled[-time_steps:]
future_predictions = []

for _ in range(future_steps):
    prediction = model.predict(last_sequence.reshape(1, time_steps, 1))
    future_predictions.append(prediction[0, 0])
    last_sequence = np.append(last_sequence[1:], prediction, axis=0)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_predictions = np.expm1(future_predictions)
future_years = np.arange(global_gdp['Year'].iloc[-1] + 1, global_gdp['Year'].iloc[-1] + 1 + future_steps)

# Display forecasted GDP Trend
forecasted_data = pd.DataFrame({'Year': future_years, 'Forecasted GDP Trend': future_predictions.flatten()})
print(forecasted_data)
