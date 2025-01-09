import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os.path
from DataPreparation import normalize

if(not os.path.exists("../Data Sources/Normalized_Albania_Information.csv")):
    normalize()
# Load the dataset
file_path = '../Data Sources/Normalized_Albania_Information.csv'
data = pd.read_csv(file_path)

# Data Preparation
data = data.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

# Feature selection
features_to_use = ['Real GDP', 'GDP Growth Rate', 'Population', 'Population Growth Rate',
                   'Real GDP per capita', 'Real GDP per capita growth rate',
                   'GDP deflator', 'GDP deflator growth rates', 'CPI', 'CPI Growth Rate']
data = data[['Year'] + features_to_use]

# Train-test split
X = data[['Year'] + features_to_use[1:]]  # Use all features except 'Real GDP'
y = data['Real GDP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Predict Real GDP for 2035–2044
future_years = pd.DataFrame({'Year': np.arange(2035, 2045)})
future_features = future_years.copy()

# Include feature engineering or assumptions for future values
for feature in features_to_use[1:]:
    future_features[feature] = np.polyval(np.polyfit(data['Year'], data[feature], 1), future_years['Year'])

# Predict
future_years['Predicted Real GDP'] = model.predict(future_features)
print(future_years)
