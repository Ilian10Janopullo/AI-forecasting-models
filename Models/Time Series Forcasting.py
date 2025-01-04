import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import GRU, Dense, Dropout # type: ignore
from tensorflow.keras.losses import Huber # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.activations import swish # type: ignore
import matplotlib.pyplot as plt


# Load the new dataset
file_path = 'Data Sources/Albania Information.csv'
data = pd.read_csv(file_path)

# Data Preparation
# Handle missing values (e.g., interpolate numerical columns)
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

# Create sequences for the GRU model
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, 1:])  # Use all features except 'Year'
        y.append(data[i + time_steps, 1])    # Predict 'Real GDP'
    return np.array(X), np.array(y)

# Scale features
scaler = RobustScaler()
scaled_features = scaler.fit_transform(data[features_to_use])

# Combine Year and scaled features for sequence generation
combined_data = np.column_stack([data['Year'], scaled_features])

# Create sequences
time_steps = 10
X, y = create_sequences(combined_data, time_steps)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the GRU Model with one layer
input_shape = (X_train.shape[1], X_train.shape[2])
model = Sequential([
    GRU(64, return_sequences=False, kernel_regularizer=l2(0.04), input_shape=input_shape),
    Dropout(0.323),
    Dense(32),
    Dense(16, activation=swish),   # Swish activation in hidden layer
    Dense(16, activation='linear'),  # Linear for regression output
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

# Ensure the correct number of features is maintained for inverse scaling
placeholder_train = np.zeros((train_predictions.shape[0], scaled_features.shape[1] - 1))
placeholder_test = np.zeros((test_predictions.shape[0], scaled_features.shape[1] - 1))

train_predictions = scaler.inverse_transform(np.column_stack([train_predictions, placeholder_train]))[:, 0]
test_predictions = scaler.inverse_transform(np.column_stack([test_predictions, placeholder_test]))[:, 0]

placeholder_y_train = np.zeros((y_train.shape[0], scaled_features.shape[1] - 1))
placeholder_y_test = np.zeros((y_test.shape[0], scaled_features.shape[1] - 1))

y_train = scaler.inverse_transform(np.column_stack([y_train.reshape(-1, 1), placeholder_y_train]))[:, 0]
y_test = scaler.inverse_transform(np.column_stack([y_test.reshape(-1, 1), placeholder_y_test]))[:, 0]

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
plt.plot(data['Year'], data['Real GDP'], label='True Real GDP', color='blue')
plt.plot(data['Year'][time_steps:time_steps+len(train_predictions)], train_predictions, label='Train Predictions', color='green')
plt.plot(data['Year'][time_steps+len(train_predictions):time_steps+len(train_predictions)+len(test_predictions)], test_predictions, label='Test Predictions', color='red')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Real GDP')
plt.title('Real GDP Forecasting with GRU and Enhanced Features')
plt.show()

# Forecast future Real GDP
future_steps = 10
last_sequence = scaled_features[-time_steps:]
future_predictions = []

for _ in range(future_steps):
    prediction = model.predict(last_sequence.reshape(1, time_steps, -1))
    future_predictions.append(prediction[0, 0])
    last_sequence = np.append(last_sequence[1:], [[prediction[0, 0]] + [0] * (scaled_features.shape[1] - 1)], axis=0)

# Inverse transform future predictions
placeholder_future = np.zeros((len(future_predictions), scaled_features.shape[1] - 1))
future_predictions = scaler.inverse_transform(np.column_stack([future_predictions, placeholder_future]))[:, 0]

# Generate future years
future_years = np.arange(data['Year'].iloc[-1] + 1, data['Year'].iloc[-1] + 1 + future_steps)

# Display forecasted data
forecasted_data = pd.DataFrame({'Year': future_years, 'Forecasted Real GDP': future_predictions})
print(forecasted_data)
