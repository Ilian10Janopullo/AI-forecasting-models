import pandas as pd  
import numpy as np
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler

real_gdp_min = 0
real_gdp_max = 0

def normalize():
    # Load data
    file_path = 'Data Sources/Albania Information.csv'
    data = pd.read_csv(file_path)

    # Strip whitespace from column names (if any)
    data.columns = data.columns.str.strip()

    # Ensure 'Year' is treated as numeric for calculations
    data['Year'] = pd.to_numeric(data['Year'])

    # Features to interpolate
    features_to_interpolate = ['Real GDP', 'Population', 'Exchange Rates', 
                            'Real GDP per capita', 'GDP deflator', 'CPI']
    
    # Interpolate the required features
    years = data['Year']
    for feature in features_to_interpolate:
        if feature in data.columns:  # Ensure the feature exists
            data[feature] = extrapolate_feature(data, feature, years)
        else:
            print(f"Feature '{feature}' not found in the dataset.")

    for feature in features_to_interpolate:
        data = calculate_growth_rate(data, feature)

    # Set specific values for the 1970 row
    data.loc[data['Year'] == 1970, ['GDP Growth Rate', 'Population Growth Rate', 
                                    'Exchange Rate', 'Real GDP per capita growth rate', 
                                    'GDP deflator growth rate', 'CPI Growth Rate']] = 0

    # Set growth rates to 0 for all "Growth Rate" columns in the 1970 row
    growth_rate_columns = [col for col in data.columns if 'Growth Rate' in col]
    data.loc[data['Year'] == 1970, growth_rate_columns] = 0

    # Interpolate GDP deflator for 1970 based on nearby years (1971, 1972)
    if 1971 in data['Year'].values and 1972 in data['Year'].values:
        gdp_deflator_1971 = data.loc[data['Year'] == 1971, 'GDP deflator'].values[0]
        gdp_deflator_1972 = data.loc[data['Year'] == 1972, 'GDP deflator'].values[0]
        interpolated_gdp_deflator_1970 = (gdp_deflator_1971 + gdp_deflator_1972) / 2
        data.loc[data['Year'] == 1970, 'GDP deflator'] = interpolated_gdp_deflator_1970

    # For rows with negative CPI values after interpolation, set them to a small positive value (0.2)
    data['CPI'] = data['CPI'].apply(lambda x: max(x, 0.2) if isinstance(x, (int, float)) else x)

    # Drop the last four columns
    data = data.iloc[:, :-4]

    # Normalizing the data (except 'Year')
    scaler = MinMaxScaler()
    global real_gdp_min 
    real_gdp_min = data['Real GDP'].min()
    global real_gdp_max 
    real_gdp_max = data['Real GDP'].max()
    columns_to_normalize = data.columns.difference(['Year'])

    data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

    # Save normalized data to a new file
    normalized_file_path = 'Data Sources/Normalized_Albania_Information.csv'
    data.to_csv(normalized_file_path, index=False)
    

# Interpolation function with extrapolation for missing years
def extrapolate_feature(data, feature, years):
    column = data[feature]
    available_years = years[~column.isna()]
    available_values = column.dropna()
    
    if len(available_values) < 2:  # Not enough data to fit a trend
        return column  # Return as-is
    
    # Fit a linear trend for simplicity
    def linear_trend(x, a, b):
        return a * x + b

    # Fit the trend
    popt, _ = curve_fit(linear_trend, available_years, available_values)
    a, b = popt

    # Extrapolate for missing years
    missing_years = years[column.isna()]
    extrapolated_values = linear_trend(missing_years, a, b)

    # Fill NaN values with extrapolated values
    data.loc[column.isna(), feature] = extrapolated_values
    return data[feature]


# Calculate growth rates
def calculate_growth_rate(data, feature):
    growth_rate_feature = f"{feature} Growth Rate"
    
    # Handle the first year with growth rate calculation using the most recent year's data
    first_year = data.iloc[0]['Year']
    last_year = data.iloc[-1]['Year']
    
    # For the first year, set the growth rate to 0 and calculate using the last year's data
    data[growth_rate_feature] = (data[feature].diff() / data[feature].shift(1)) * 100
    # Fill the first year's growth rate by calculating from the most recent year's value
    data.loc[data['Year'] == first_year, growth_rate_feature] = 0
    
    # Calculate the first year growth rate using the most recent year's value
    first_year_value = data.loc[data['Year'] == first_year, feature].values[0]
    last_year_value = data.loc[data['Year'] == last_year, feature].values[0]
    first_year_growth_rate = ((last_year_value - first_year_value) / first_year_value) * 100
    data.loc[data['Year'] == first_year, growth_rate_feature] = first_year_growth_rate
    
    return data

def getMin():
    global real_gdp_min
    return real_gdp_min

def getMax():
    global real_gdp_max
    return real_gdp_max