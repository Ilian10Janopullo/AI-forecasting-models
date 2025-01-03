import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings

# Step 1: Load and Prepare Data
def load_and_preprocess_data(file_path, country):
    data = pd.read_csv(file_path, encoding='latin1', skiprows=11)
    data.columns = data.iloc[0]
    data = data[1:]
    
    country_data = data[data['Country'] == country]
    country_data = country_data.drop(columns=['Country']).reset_index(drop=True)
    country_data = country_data.T
    country_data.columns = ['GDP']
    
    country_data = country_data.reset_index()
    country_data.columns = ['Year', 'GDP']
    country_data['Year'] = country_data['Year'].str.extract(r'(\d+)')
    country_data = country_data.dropna(subset=['Year'])
    country_data['Year'] = country_data['Year'].astype(int)
    country_data['GDP'] = pd.to_numeric(country_data['GDP'], errors='coerce')
    country_data = country_data.dropna(subset=['GDP'])
    country_data['GDP'] = country_data['GDP'].clip(lower=1e-2)  # Handle small GDP values
    return country_data

file_path = 'Data Sources/RealGDP.csv'
country = 'Albania'
global_gdp = load_and_preprocess_data(file_path, country)

# Suppress warnings for ARIMA convergence issues
warnings.filterwarnings("ignore")

def grid_search_arima(data, p_values, d_values, q_values):
    
    #Perform grid search to find the best ARIMA parameters (p, d, q).
    
    best_score, best_order = float("inf"), None  # Initialize best score
    results = []  # To store all combinations and their scores
    
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    # Fit ARIMA model
                    model = ARIMA(data, order=(p, d, q))
                    result = model.fit()
                    
                    # Use AIC as the evaluation metric
                    aic = result.aic
                    results.append((p, d, q, aic))  # Store results
                    
                    # Update best score and parameters if current AIC is lower
                    if aic < best_score:
                        best_score, best_order = aic, (p, d, q)
                except Exception as e:
                    continue  # Skip invalid combinations
    
    return best_order, best_score

# Define grid search ranges
p_values = range(0, 5)  # AR terms
d_values = range(0, 2)  # Differencing terms
q_values = range(0, 5)  # MA terms

# Perform grid search on the GDP data
best_order, best_aic = grid_search_arima(global_gdp['GDP'], p_values, d_values, q_values)

# Step 2: Apply ARIMA for Trend Extraction
def apply_arima(data, order=best_order):
    arima_model = ARIMA(data['GDP'], order=order)
    arima_result = arima_model.fit()
    data['ARIMA_Trend'] = arima_result.fittedvalues
    data['Residuals'] = data['GDP'] - data['ARIMA_Trend']
    return data

global_gdp = apply_arima(global_gdp)

# Step 3: Scale Residuals and Handle Small Values
def scale_residuals(data):
    epsilon = 1e-6
    data['Residuals'] = np.sign(data['Residuals']) * np.log1p(np.abs(data['Residuals']) + epsilon)
    scaler = RobustScaler()
    scaled_residuals = scaler.fit_transform(data['Residuals'].values.reshape(-1, 1))
    return scaled_residuals, scaler

scaled_residuals, scaler = scale_residuals(global_gdp)

# Step 4: Create Sequences for GRU
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

time_steps = 10
X, y = create_sequences(scaled_residuals, time_steps)

# Split into train and test sets
train_size = int(len(X) * 0.7)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Step 5: Build and Train GRU Model
def build_gru_model(input_shape):
    model = Sequential([
        GRU(200, return_sequences=True, kernel_regularizer=l2(0.025), input_shape=input_shape),
        Dropout(0.3),
        GRU(150, return_sequences=False, kernel_regularizer=l2(0.025)),
        Dropout(0.3),
        Dense(100),
        Dense(1)  # Predict GDP Trend
    ])
    model.compile(optimizer='adam', loss=Huber(), metrics=['mape'])
    return model

model = build_gru_model((X_train.shape[1], 1))

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# Step 6: Evaluate Model
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Reverse scaling
def inverse_transform(data, scaler):
    data = np.expm1(data)  # Reverse log transformation
    return scaler.inverse_transform(data.reshape(-1, 1))[:, 0]

final_train_predictions = inverse_transform(train_predictions, scaler)
final_test_predictions = inverse_transform(test_predictions, scaler)

# Combine ARIMA trend with GRU residual predictions
train_predictions = final_train_predictions + global_gdp['ARIMA_Trend'].iloc[time_steps:time_steps+len(final_train_predictions)].values
test_predictions = final_test_predictions + global_gdp['ARIMA_Trend'].iloc[time_steps+len(final_train_predictions):].values

# Compute Metrics
def calculate_smape(true, predicted, epsilon=1e-5):
    return np.mean(2 * np.abs(true - predicted) / (np.abs(true) + np.abs(predicted) + epsilon)) * 100

train_rmse = np.sqrt(mean_squared_error(global_gdp['GDP'].iloc[time_steps:time_steps+len(final_train_predictions)], train_predictions))
test_rmse = np.sqrt(mean_squared_error(global_gdp['GDP'].iloc[time_steps+len(final_train_predictions):], test_predictions))
train_mae = mean_absolute_error(global_gdp['GDP'].iloc[time_steps:time_steps+len(final_train_predictions)], train_predictions)
test_mae = mean_absolute_error(global_gdp['GDP'].iloc[time_steps+len(final_train_predictions):], test_predictions)
train_smape = calculate_smape(global_gdp['GDP'].iloc[time_steps:time_steps+len(final_train_predictions)], train_predictions)
test_smape = calculate_smape(global_gdp['GDP'].iloc[time_steps+len(final_train_predictions):], test_predictions)
train_r2 = r2_score(global_gdp['GDP'].iloc[time_steps:time_steps+len(final_train_predictions)], train_predictions)
test_r2 = r2_score(global_gdp['GDP'].iloc[time_steps+len(final_train_predictions):], test_predictions)

print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")
print(f"Train MAE: {train_mae:.2f}")
print(f"Test MAE: {test_mae:.2f}")
print(f"Train SMAPE: {train_smape:.2f}%")
print(f"Test SMAPE: {test_smape:.2f}%")
print(f"Train R²: {train_r2:.2f}")
print(f"Test R²: {test_r2:.2f}")

# Step 7: Visualization
plt.figure(figsize=(14, 7))
plt.plot(global_gdp['Year'], global_gdp['GDP'], label='True GDP', color='blue')
plt.plot(global_gdp['Year'][time_steps:time_steps+len(train_predictions)], train_predictions, label='Train Predictions', color='green')
plt.plot(global_gdp['Year'][time_steps+len(train_predictions):], test_predictions, label='Test Predictions', color='red')
plt.legend()
plt.xlabel('Year')
plt.ylabel('GDP')
plt.title(f'Hybrid Model GDP Forecasting for {country}')
plt.show()
