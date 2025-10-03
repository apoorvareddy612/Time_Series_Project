import pandas as pd
import matplotlib.pyplot as plt
from toolbox import Cal_rolling_mean_var,autocorrelation
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
import numpy as np

df = pd.read_csv('/Users/apoorvareddy/Downloads/Academic/DATS6313/Project/energy_dataset.csv')
df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
df.set_index('time', inplace=True)
df = df.resample('h').sum()
df = df.drop(['generation fossil coal-derived gas','generation fossil oil shale', 
                            'generation fossil peat', 'generation geothermal', 
                            'generation hydro pumped storage aggregated', 'generation marine', 
                            'generation wind offshore', 'forecast wind offshore eday ahead',
                            'total load forecast', 'forecast solar day ahead',
                            'forecast wind onshore day ahead'], 
                            axis=1)
df_weather = pd.read_csv('/Users/apoorvareddy/Downloads/Academic/DATS6313/Project/weather_features.csv')
df_weather['time'] = pd.to_datetime(df_weather['dt_iso'], utc=True, infer_datetime_format=True)
df_weather = df_weather.drop(['dt_iso'], axis=1)
df_weather = df_weather.set_index('time')
df_weather = df_weather.reset_index().drop_duplicates(subset=['time', 'city_name'],keep='first').set_index('time')
df_weather = df_weather.drop(['weather_main', 'weather_id', 
                              'weather_description', 'weather_icon'], axis=1)
print(df_weather.shape)
print(df.shape)



# 1. Forecast Variable and Numerical Independent Variables
# For this example, let's assume we are forecasting temperature (T (degC))
forecast_variable = df['price actual']
independent_variables = df.drop(['price actual'], axis=1)

# # 2. Plot the time series and check for stationarity
# plt.figure(figsize=(10,6))
# plt.plot(forecast_variable, label='Price')
# plt.title('Electricity Price Time Series')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# plt.show()

# Check for stationarity using the Augmented Dickey-Fuller test
adf_result = adfuller(forecast_variable.dropna())
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')
print(f'Critical Values: {adf_result[4]}')
if adf_result[1] < 0.05:
    print('The time series is stationary.')
else:
    print('The time series is not stationary.')

# # # 3. Calculate and plot the rolling mean and variance
# # Cal_rolling_mean_var(forecast_variable, 'price actual')

# price_actual_acf = acf(df['price actual'].dropna(), nlags=50)

# # Prepare lags for positive and negative sides
# lags = np.arange(-len(price_actual_acf) + 1, len(price_actual_acf))

# # Create a full ACF array with positive and negative lags
# full_acf = np.concatenate((price_actual_acf[::-1][:-1], price_actual_acf))

# # Calculate significance threshold
# N = len(df['price actual'].dropna())
# confidence_interval = 1.96 / np.sqrt(N)

# # Plot ACF with positive and negative lags
# plt.figure(figsize=(10, 6))
# plt.stem(lags, full_acf, basefmt=" ",linefmt='b', markerfmt='ro')
# plt.axhline(0, color='black', lw=1)

# # Shade the area between confidence intervals
# plt.fill_between(lags, confidence_interval, -confidence_interval, color='red', alpha=0.2, label='Significance Level')

# # Labels and title
# plt.title('ACF of Price Actual (Positive and Negative Lags)')
# plt.xlabel('Lag')
# plt.ylabel('Autocorrelation')
# plt.legend()
# plt.show()