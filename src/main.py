#%%
#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from toolbox import *
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error 
from sklearn.decomposition import PCA
from scipy.stats import chi2
from scipy import signal
from scipy.stats import norm
from scipy.signal import dlsim
# %%
#Importing Data
data = pd.read_csv('./Data/final.csv')
data.head()

#%%
# Convert the time column to datetime, including timezone information
data['time'] = pd.to_datetime(data['time'], utc=True)  
data.set_index('time', inplace=True)


#%%
#Data Cleaning
#Selecting the features
selected_features = [
    'generation biomass', 
    'generation fossil brown coal/lignite', 
    'generation fossil gas', 
    'generation fossil hard coal', 
    'generation fossil oil',
    'generation hydro pumped storage consumption',
    'generation hydro run-of-river and poundage', 
    'generation hydro water reservoir',
    'generation nuclear', 
    'generation other', 
    'generation other renewable', 
    'generation solar',
    'generation waste', 
    'generation wind onshore', 
    'total load actual',  
    'price actual'
]
data = data[selected_features]
#Checking for missing values
missing_values = data.isnull().sum()
print('Missing Values:')
print(missing_values)

data['hour'] = data.index.hour  # Extract hour from datetime index

# Impute by taking mean or median by hour or day
data = data.groupby('hour').transform(lambda x: x.fillna(x.mean()))
print("Remaining NaNs:", data.isna().sum())
# %% 
#---------------plot of the dependent variable versus time-----------------

daily_price = data['price actual'].asfreq('D')

# Plotting the monthly electricity price
plt.figure(figsize=(14, 7))
plt.plot(daily_price.index, daily_price, color='blue', label='Actual Price (€/MWh)')
plt.title('Actual Electricity Price (Daily Frequency)')
plt.xlabel('Time')
plt.ylabel('Actual Price (€/MWh)')
plt.legend(['Actual Price'])
plt.grid(True)
plt.show()

# %%
# 2. ACF and PACF of the dependent variable
y = data['price actual']
#lag - 50
ACF_PACF_Plot(y,50)
# %%
# Calculate the correlation matrix for the selected features
corr_matrix = data.corr()

# Plot the correlation matrix using seaborn heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix for Selected Features')
plt.show()

# %%
#Splitting the dataset into train (80%) and test (20%) sets
X = data.drop(columns='price actual')
y = data['price actual']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print('Length of X_train and y_train:', len(X_train))
print('Length of X_test and y_test:', len(X_test))

# %%
#Checking Stationarity on Original Data
check_stationarity(y, 'Price Actual - Original Data')
# %%
#Applying Differencing to make the data stationary
y_diff = differencing(y,1,365)
y_diff = np.array(removeNA(y_diff))
y_diff = y_diff.astype(float)
# %%
#Checking Stationarity on Transformed Data
check_stationarity(y_diff, 'Price Actual - Transformed Data')
# %%

T,S,R = STL_analysis(y,365)
# %%
# Additive decomposition
additive_decomposition = T + S + R

# Multiplicative decomposition
multiplicative_decomposition = T * S * R

# Reconstructed Time Series
plt.subplot(4, 1, 4)
plt.plot(data['price actual'],additive_decomposition, label="Reconstructed", color="red")
plt.title("Reconstructed Time Series")
plt.xlabel("Time")
plt.ylabel("Y(t) = T(t) + S(t) + R(t)")
plt.grid(True)

plt.tight_layout()
plt.show()

# Plotting the updated components and the reconstructed time series for multiplicative decomposition
plt.figure(figsize=(12, 8))
plt.plot(data['price actual'], multiplicative_decomposition, label="Reconstructed", color="red")
plt.title("Reconstructed Time Series (Multiplicative)")
plt.xlabel("Time")
plt.ylabel("Y(t) = T(t) * S(t) * R(t)")
plt.grid(True)

plt.tight_layout()
plt.show()

#Pick the Additive Decomposition

# %%'
#Holt-Winters Method
modelES = ExponentialSmoothing(y_train, seasonal='add', seasonal_periods=365).fit()

forecastES = modelES.forecast(steps=len(y_test))

title = 'Winter-Holt forecasting'
plot_forecasting_models(y_train, y_test, forecastES, title)

# Calculating MSE for Winter-Holt method
_, _, mse = cal_error_MSE(y_test, forecastES)

rmse_winter = np.sqrt(mse)
mae_winter = mean_absolute_error(y_test, forecastES)
print('MSE for Winter-Holt method:', np.round(mse, 2))
print('RMSE for Winter-Holt method:', np.round(np.sqrt(mse), 2))
print("MAE for Holt-Winter method:", np.round(mae_winter, 2))
# %%
#Collinearity Check

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

condition_number = LA.cond(X_train_scaled)
print('condition_number:', condition_number)

X_svd = X_train.to_numpy()
H = np.matmul(X_svd.T, X_svd)

s, d, v = LA.svd(H)
print('SingularValues =', d)

#%%
#Backward Stepwise Regression
# Function to calculate AIC, BIC, and Adjusted R² for a given model
def calculate_metrics(X, y):
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    return model.aic, model.bic, model.rsquared_adj

# Function to perform backward stepwise regression iteratively
def backward_stepwise_regression(X, y):
    features = list(X.columns)
    metrics_list = []

    while len(features) >= 1:  # Stop when there's only one feature left
        # Fit the model with the current features
        aic, bic, adj_r2 = calculate_metrics(X[features], y)
        metrics_list.append((aic, bic, adj_r2, features.copy()))

        # Check Adjusted R² values to find the feature to remove
        # Here we will remove the first feature in the list for simplicity
        feature_to_remove = features[0]
        features.remove(feature_to_remove)

    # Create a DataFrame to display the results
    metrics_df = pd.DataFrame(metrics_list, columns=['AIC', 'BIC', 'Adjusted R²', 'Features'])
    return metrics_df

# Perform backward stepwise regression
result_backward_stepwise = backward_stepwise_regression(pd.DataFrame(X_train, columns=X.columns), y_train)

# Display the results
print(result_backward_stepwise)

#%%
#VIF Analysis

# Function to calculate VIF
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data

# Function to perform VIF-based feature elimination iteratively
def vif_selection_iteratively(X, y):
    features = list(X.columns)
    results = []

    while len(features) > 1:  # Stop when there's only one feature left
        # Create a new feature set excluding the current features
        X_with_const = sm.add_constant(X[features])
        model = sm.OLS(y, X_with_const).fit()

        # Calculate VIF for the current features
        vif_data = calculate_vif(X[features])

        # Store metrics
        results.append((model.aic, model.bic, model.rsquared_adj, features.copy(), vif_data))

        # Remove the first feature in the list for simplicity
        feature_to_remove = features[0]
        features.remove(feature_to_remove)

    # Create a DataFrame to display the results
    vif_metrics_df = pd.DataFrame(results, columns=['AIC', 'BIC', 'Adjusted R²', 'Remaining Features', 'VIF Data'])
    return vif_metrics_df

# Perform VIF-based selection
result_vif_selection = vif_selection_iteratively(pd.DataFrame(X_train, columns=X.columns), y_train)

# Display the results
for step in range(len(result_vif_selection)):
    print(f"\nStep {step + 1}:")
    print(f"AIC: {result_vif_selection.iloc[step]['AIC']}, BIC: {result_vif_selection.iloc[step]['BIC']}, Adjusted R²: {result_vif_selection.iloc[step]['Adjusted R²']}")
    print("Remaining Features:", result_vif_selection.iloc[step]['Remaining Features'])
    print("VIF Data:")
    print(result_vif_selection.iloc[step]['VIF Data'])


#%%
#PCA
pca = PCA(n_components=0.92)  # Retain 92% variance
X_train_pca = pca.fit_transform(X_train_scaled)
print("Reduced dimensions:", X_train_pca.shape[1])


#%%
#Average Forecasting
_, forecast_average = average_forecasting(y_train, y_test)
_, _, mse_average = cal_error_MSE(y_test, forecast_average)
rmse_average = np.sqrt(mse_average)
mae_average = mean_absolute_error(y_test, forecast_average)
print('MSE for Average forecasting:', np.round(mse_average, 2))
print('RMSE for Average forecasting:', np.round(rmse_average, 2))
print("MAE for Average forecasting:", np.round(mae_average, 2))

#%%
# Naive forecasting
_, forecast_Naive = Naive_forecasting(y_train, y_test)
_, _, mse_Naive = cal_error_MSE(y_test, forecast_Naive)
rmse_Naive = np.sqrt(mse_Naive)
mae_Naive = mean_absolute_error(y_test, forecast_Naive)
print('MSE for Naive forecasting:', np.round(mse_Naive, 2))
print('RMSE for Naive forecasting:', np.round(rmse_Naive, 2))
print("MAE for Naive forecasting:", np.round(mae_Naive, 2))

#%%
# Drift forecasting
_, forecast_Drift = drift_forecasting(y_train, y_test)
_, _, mse_Drift = cal_error_MSE(y_test, forecast_Drift)
rmse_Drift = np.sqrt(mse_Drift)
mae_Drift = mean_absolute_error(y_test, forecast_Drift)
print('MSE for Drift forecasting:', np.round(mse_Drift, 2))
print('RMSE for Drift forecasting:', np.round(rmse_Drift, 2))
print("MAE for Drift forecasting:", np.round(mae_Drift, 2))

#%%
# Simple Exponential Smoothing
L0 = y_train[0]
_, forecast_SES = ses_forecasting(y_train, y_test, L0, alpha=0.9)
_, _, mse_SES = cal_error_MSE(y_test, forecast_SES)
rmse_SES = np.sqrt(mse_SES)
mae_SES = mean_absolute_error(y_test, forecast_SES)
print('MSE for SES forecasting:', np.round(mse_SES, 2))
print('RMSE for SES forecasting:', np.round(rmse_SES, 2))
print("MAE for SES forecasting:", np.round(mae_SES, 2))

#%%
#Selecting the features with VIF < 60
new_selected_features = ['generation other renewable', 'generation solar', 'generation waste', 'generation wind onshore', 'total load actual']
X  = data[new_selected_features]
y = data['price actual']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=False)
#Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
#%%
#Multi Linear Regression 
# adding bias
X_train_scaled = sm.add_constant(X_train_scaled)

cols = X_train.columns
cols = np.insert(cols, 0, 'constant')
# Going forward with Backward Stepwise Regression to reduce features
X_train_scaled = pd.DataFrame(X_train_scaled, columns=cols, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
X_test_scaled = pd.concat([pd.Series(1, index=X_test.index, name='constant'), X_test_scaled], axis=1)
result = sm.OLS(y_train, X_train_scaled).fit()
y_pred = result.predict(X_train_scaled)
y_forecast = result.predict(X_test_scaled)
mse_ols = mean_squared_error(y_test, y_forecast)
rmse_ols = np.sqrt(mse_ols)
ols_residuals = result.resid
print(result.summary())
# plot_forecasting_models(y_train, y_test, y_forecast, 'OLS Model')
print(f'T-Test :{result.pvalues}')
print(f'F-Test :{result.f_pvalue}')
print('MSE for OLS model:', np.round(mse_ols, 2))
print('AIC for OLS model:', np.round(result.aic, 2))
print('BIC for OLS model:', np.round(result.bic, 2))
print('RMSE for OLS model:', np.round(rmse_ols, 2))
print('R² for OLS model:', np.round(result.rsquared, 2))
print('Adjusted R² for OLS model:', np.round(result.rsquared_adj, 2))
# print(f'ACF of residuals for OLS model:{cal_autocorr(ols_residuals, 50)}')
q_ols = cal_Q_value(ols_residuals, 'OLS Residuals', 50)
print('Q Value of OLS residuals:', np.round(q_ols, 2))
# 42426.55565814915
print('Mean of residuals for OLS:', np.mean(ols_residuals))
print('Variance of residuals for OLS:', np.var(ols_residuals))

#%%
residuals = y_train - y_pred
acf_values = cal_autocorr(residuals, 50)
# Calculate confidence intervals
N = len(y_train)
conf_interval = 1.96 / np.sqrt(N)

# Plot the ACF
plt.figure(figsize=(10, 6))
lags = range(-50, 51)
plt.stem(lags, acf_values)
plt.axhspan(-conf_interval, conf_interval, color='red', alpha=0.2)
plt.title('Autocorrelation Function of Residuals')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.grid(True)
plt.show()


#%%
# Plot the train, test, and predicted values
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_train)), y_train, label='Training Data')
plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, label='Test Data')
plt.plot(range(len(y_train), len(y_train) + len(y_forecast)), y_forecast, label='Forecasted Data')

# Add labels and title
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Train, Test, and Forecasted Values')
plt.legend()
plt.show()

#%%
#Autocorrelation Function
max_lags = 20
Ry = cal_autocorr(y_diff,max_lags)
# Calculate confidence intervals
N = len(y_diff)
conf_interval = 1.96 / np.sqrt(N)

# Plot the ACF
plt.figure(figsize=(10, 6))
lags = range(-max_lags, max_lags + 1)
plt.stem(lags, Ry)
plt.axhspan(-conf_interval, conf_interval, color='red', alpha=0.2)
plt.title('Autocorrelation Function of Price Actual')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.grid(True)
plt.show()

#%%
#GPAC
cal_gpac(Ry,7,7)

# na = 2, nb = 2
#%%
#LM
def lm_cal_e(y, na, theta, seed=6313):
    np.random.seed(seed)
    den = theta[:na]
    num = theta[na:]
    if len(den) > len(num):  # matching len of num and den
        for x in range(len(den) - len(num)):
            num = np.append(num, 0)
    elif len(num) > len(den):
        for x in range(len(num) - len(den)):
            den = np.append(den, 0)
    den = np.insert(den, 0, 1)
    num = np.insert(num, 0, 1)
    sys = (den, num, 1)
    _, e = signal.dlsim(sys, y)
    return e


def lm_step1(y, na, nb, delta, theta):
    n = na + nb
    e = lm_cal_e(y, na, theta)
    sse_old = np.dot(np.transpose(e), e)
    X = np.empty(shape=(len(y), n))
    for i in range(0, n):
        theta[i] = theta[i] + delta
        e_i = lm_cal_e(y, na, theta)
        x_i = (e - e_i) / delta
        X[:, i] = x_i[:, 0]
        theta[i] = theta[i] - delta
    A = np.dot(np.transpose(X), X)
    g = np.dot(np.transpose(X), e)
    return A, g, X, sse_old


def lm_step2(y, na, A, theta, mu, g):
    delta_theta = np.matmul(np.linalg.inv(A + (mu * np.identity(A.shape[0]))), g)
    theta_new = theta + delta_theta
    e_new = lm_cal_e(y, na, theta_new)
    sse_new = np.dot(np.transpose(e_new), e_new)
    if np.isnan(sse_new):
        sse_new = 10 ** 10
    return sse_new, delta_theta, theta_new


def lm_step3(y, na, nb):
    N = len(y)
    n = na+nb
    mu = 0.01
    mu_max = 10 ** 20
    max_iterations = 500
    delta = 10 ** -6
    var_error = 0
    covariance_theta_hat = 0
    sse_list = []
    theta = np.zeros(shape=(n, 1))

    for iterations in range(max_iterations):
        A, g, X, sse_old = lm_step1(y, na, nb, delta, theta)
        sse_new, delta_theta, theta_new = lm_step2(y, na, A, theta, mu, g)
        sse_list.append(sse_old[0][0])
        if iterations < max_iterations:
            if sse_new < sse_old:
                if np.linalg.norm(np.array(delta_theta), 2) < 10 ** -3:
                    theta_hat = theta_new
                    var_error = sse_new / (N - n)
                    covariance_theta_hat = var_error * np.linalg.inv(A)
                    print(f"Convergence Occured in {iterations} iterations")
                    break
                else:
                    theta = theta_new
                    mu = mu / 10
            while sse_new >= sse_old:
                mu = mu * 10
                if mu > mu_max:
                    print('No Convergence')
                    break
                sse_new, delta_theta, theta_new = lm_step2(y, na, A, theta, mu, g)
        if iterations > max_iterations:
            print('Max Iterations Reached')
            break
        theta = theta_new
    return theta_new, var_error[0][0], covariance_theta_hat, sse_list

theta_hat, variance_hat, covariance_hat, sse_array = lm_step3(y_diff,2,2)

#%%
# Display the estimated parameters
print("Estimated Parameters:")
print(f"Thetas: {theta_hat}")

# Display the standard deviation
print(f"Standard Deviation: {np.sqrt(variance_hat)}")


def display_confidence_intervals(theta_hat, covariance_hat):
    intervals = []
    for i, theta in enumerate(theta_hat.ravel()):
        variance = covariance_hat[i, i]
        margin = 2 * np.sqrt(variance)
        lower_bound = theta - margin
        upper_bound = theta + margin
        intervals.append((lower_bound, upper_bound))
        print(f"Theta {i}: [{lower_bound.round(3)}, {upper_bound.round(3)}]")

# Display confidence intervals
display_confidence_intervals(theta_hat, covariance_hat)

theta_hat= [item[0] for item in theta_hat]
#%%
#splitting y_diff
y_train_diff, y_test_diff = train_test_split(y_diff, test_size=0.2, shuffle=False)
#ARMA model
na = 2
nb = 2

# Initialize ARMA model
arma_model,arma_model_hat = prediction(y_diff,na,nb,forecast_steps= len(y_test_diff))


#%%
#ARIMA model
na = 2
nb = 2
d = 1
# Initialize ARIMA model
arima_model,arima_model_hat = prediction(y_diff,na,nb,d,forecast_steps= len(y_test_diff))
#%%
# SARIMA model

# Function to generate AR and MA parameters for a given SARIMA model
def num_den_ds_AR_MA(theta_ar, theta_ma, na, nb, seasonal_diff_order, d):
    """
    Function to calculate the numerator and denominator for SARIMA (Seasonal ARIMA) model.
    """
    den = np.zeros(max(na, nb) * seasonal_diff_order + 1)
    den[0] = 1
    
    num = np.zeros(max(na, nb) * seasonal_diff_order + 1)
    num[0] = 1

    if na > 0:
        for i in range(na):
            den[(i + 1) * seasonal_diff_order] = theta_ar[i]

    if d > 0:
        den_new = [-x if x != 0 else 0 for x in den]
        shifted_den_with_changed_signs = np.pad(den_new, (seasonal_diff_order, 0), 'constant')
        den_padded = np.pad(den, (0, len(shifted_den_with_changed_signs) - len(den)), 'constant')
        extended_den = den_padded + shifted_den_with_changed_signs
    else:
        extended_den = den

    if nb > 0:
        for i in range(nb):
            num[seasonal_diff_order + i] = theta_ma[i]

    num = np.pad(num, (0, len(extended_den) - len(num)), 'constant')
    
    return num, extended_den

# Function to display poles and zeros, with cancellation check
def display_poles_zeros(num, den):
    poles = np.roots(den)
    zeros = np.roots(num)
    print("\nPoles:", poles.round(3))
    print("Zeros:", zeros.round(3))
    
    # Check for pole-zero cancellations
    cancellation = any(np.isclose(pole, zero, atol=1e-3) for pole in poles for zero in zeros)
    print("Zero-pole cancellation?", "Yes" if cancellation else "No")
    
    # Perform zero-pole cancellation (remove poles near zeros)
    for zero in zeros:
        for i, pole in enumerate(poles):
            if np.isclose(zero, pole, atol=1e-3):
                poles = np.delete(poles, i)
                print(f"Pole {pole} near zero {zero} cancelled")
    
    return poles, zeros

# Function to calculate confidence intervals for AR/MA coefficients
def calculate_confidence_intervals(params, se_params, confidence_level=0.95):
    z_score = norm.ppf(1 - (1 - confidence_level) / 2)
    lower_bound = params - z_score * se_params
    upper_bound = params + z_score * se_params
    return lower_bound, upper_bound

# Diagnostic Test Function (whiteness test, residuals, etc.)
def diagnostic_test(y, forecast_steps, residuals, na, nb, lags=12):

    # Whiteness Chi-square Test for residuals
    re = cal_autocorr(residuals, lags)
    Q = len(y) * np.sum(np.square(re[1:]))
    DOF = lags - na - nb
    alfa = 0.01
    chi_critical = chi2.ppf(1 - alfa, DOF)

    print('Chi critical:', chi_critical)
    print('Q Value:', Q)
    print('Alfa value for 99% accuracy:', alfa)
    if Q < chi_critical:
        print("The residual is white noise")
    else:
        print("The residual is NOT white noise")

    # Variance of residuals (error variance)
    residual_variance = np.var(residuals)
    print(f"Estimated variance of residuals (error variance): {residual_variance:.4f}")
    
    # Bias check: mean of residuals (should be close to 0 for unbiased model)
    mean_residual = np.mean(residuals)
    print(f"Mean of residuals (should be close to 0 for unbiased model): {mean_residual:.4f}")

    # Forecast errors (out-of-sample prediction errors)
    forecast_errors = residuals[-forecast_steps:]
    forecast_variance = np.var(forecast_errors)
    
    # Compare variance of residual errors vs forecast errors
    print(f"Variance of residual errors (in-sample): {residual_variance:.4f}")
    print(f"Variance of forecast errors (out-of-sample): {forecast_variance:.4f}")


# Main function to create SARIMA model, forecast and perform diagnostic tests
def sarima_model_diagnostics(y, ar_params, ma_params, na, nb, seasonal_diff_order, d, forecast_steps, lags=12):
    """
    Main function to create SARIMA model, generate forecast and perform diagnostic tests
    """
    # SARIMA model creation using the parameters
    num, den = num_den_ds_AR_MA(ar_params, ma_params, na, nb, seasonal_diff_order, d)

    # Display poles and zeros and perform zero-pole cancellation
    display_poles_zeros(num, den)
    
    # Generate synthetic white noise (for simulation)
    n_samples = len(y)
    e = np.random.normal(0, 1, n_samples)  # White noise
    
    # Simulate the SARIMA process (use signal.dlsim to simulate the model)
    _, y_dlsim = signal.dlsim((num, den, 1), e)

    # Perform diagnostic tests
    residuals = y - y_dlsim
    diagnostic_test(y, forecast_steps, residuals, na, nb, lags)

    # Forecasting for future steps
    forecast_values = y_dlsim[-forecast_steps:]

    ar_lower, ar_upper = calculate_confidence_intervals(np.array(ar_params), np.std(ar_params))
    ma_lower, ma_upper = calculate_confidence_intervals(np.array(ma_params), np.std(ma_params))
    
    print(f"AR Coefficients confidence intervals: {list(zip(ar_lower, ar_upper))}")
    print(f"MA Coefficients confidence intervals: {list(zip(ma_lower, ma_upper))}")
    
    
    return y_dlsim, forecast_values
ar_params, ma_params = theta_hat[:na], theta_hat[na:]
seasonal_diff_order = 12
d = 1
forecast_steps = len(y_test_diff)
y_dlsim, forecast_values = sarima_model_diagnostics(y_diff, ar_params, ma_params, na, nb, seasonal_diff_order, d, forecast_steps)
#%%
#Box Jerkins model
#Pick one input
new_selected_features = ['generation fossil hard coal']
X  = data[new_selected_features].values 
y = data['price actual']
#%%
#GPAC Order Determination
K = 50
# Define the cal_corr function

def cal_corr(signal, signal_2, K):
    assert len(signal) == len(signal_2), "Signals must have the same length."
    
    N = len(signal)
    correlation = np.zeros(K)
    
    # Compute cross-correlation using the explicit loop approach
    for tau in range(K):
        correlation[tau] = np.sum([
            signal[k] * signal_2[k + tau] 
            for k in range(1,N - tau)
        ]) / (N - tau)
    
    return correlation

#%%
ru = cal_corr(X,X, K)
ru_reversed = ru[::-1]
ru = np.concatenate((ru_reversed[:-1], ru))
# Create R_u(τ) matrix
Ru = np.zeros((K, K))
for i in range(K):
    if i == 0:
        Ru[i] = np.hstack((ru[K-1-i:]))
    else:
        Ru[i] = np.hstack((ru[K-1-i:-i]))


# %%
Ruy = cal_corr(X, y, K)
g = np.linalg.inv(Ru) @ Ruy
g_zero = np.zeros((K-1))
g1 = np.concatenate((g_zero, g))
#%%
def calc_gpac_values(ry, J, K):

    den = np.zeros((K, K))

    for k in range(0,K):
        row = np.zeros(K)
        for i in range(0,K):
            row[i] = ry[np.abs(J + k - i)]
        den[k] = row
    col = np.zeros(K)
    for i in range(0,K):
        col[i] = ry[J+i+1]
    num = np.concatenate((den[:, :-1], col.reshape(-1, 1)), axis=1)
    num = np.array(num)
    den = np.array(den)
    if np.linalg.det(den) == 0:
        return np.inf
    if np.abs(np.linalg.det(num)/np.linalg.det(den)) < 0.00001:
        return 0
    return np.linalg.det(num)/np.linalg.det(den)


def generate_gpac_table(ry, J=7, K=7,title='GPAC Table'):
    gpac_arr = np.zeros((J, K))
    gpac_arr.fill(None)
    for k in range(1, K):
        for j in range(J):
            gpac_arr[j][k] = calc_gpac_values(ry, j, k)
    gpac_arr = np.delete(gpac_arr, 0, axis=1)
    # creating dataframe
    cols = []
    for k in range(1, K):
        cols.append(k)
    ind = []
    for j in range(J):
        ind.append(j)
    df = pd.DataFrame(gpac_arr, columns=cols, index=ind)
    
    fig = plt.figure()
    ax = sns.heatmap(df, annot=True, fmt='0.2f') 
    plt.title(f'{title}')
    plt.tight_layout()
    plt.show()
    print(df)
#%%
title = 'G-GPAC [B(q) and F(q)]'
generate_gpac_table(g, J=7, K=7,title=title)
# %%
def cal_v(y, u, g1, K):
    """Compute v(t) as y(t) - sum(g(i) * u(t-i)) for i = 0 to K."""
    v = np.zeros(len(y))  # Initialize v array with zeros
    for t in range(K, len(y)):  # Start from K because we need past values of u
        v[t] = y[t] - np.sum(g1[i] * u[t - i] for i in range(K))
    return v

# %%

v = cal_v(y,X, g, K)
rv= cal_autocorr(v, K)
title = 'H-PAC [C(q) and D(q)]'
generate_gpac_table(rv, J=7, K=7,title=title)
#%%
#LM Algorithm

def box_jenkins_cal_e(y, u, nf, nb, nc, nd, theta, seed=6313):
    np.random.seed(seed)
    # Split theta into transfer function components
    den_G = theta[:nb]  # B(q) coefficients
    num_G = theta[nb:nb+nf]  # F(q) coefficients
    den_H = theta[nb+nf:nb+nf+nc]  # C(q) coefficients
    num_H = theta[nb+nf+nc:]  # D(q) coefficients
    # Add leading 1 for system stability
    den_G = np.insert(den_G, 0, 1)  # B(q)
    num_G = np.insert(num_G, 0, 1)  # F(q)
    den_H = np.insert(den_H, 0, 1)  # C(q)
    num_H = np.insert(num_H, 0, 1)  # D(q)

    
    # Transfer functions for H(q) = D(q)/C(q)
    sys_H = (num_H, den_H, 1)
    
    # Simulate H(q) applied to y(t)
    _, y_H = dlsim(sys_H, y)

    # Transfer functions for G(q) = B(q)/F(q)
    sys_G = (den_G, num_G, 1) 
    # Simulate G(q) applied to u(t)
    _, u_g = dlsim(sys_G, u)

    # Simulate D(q)/C(q) * B(q)/F(q) applied to u(t)
    _,u_UD = dlsim(sys_H, u_g)  
    
    # Calculate the residuals
    e = np.squeeze(y_H) - np.squeeze(u_UD)
    
    return e

def lm_step1_bj(y, u, na, nb, nc, nd, delta, theta):
    n = na + nb + nc + nd
    e = box_jenkins_cal_e(y, u, na, nb, nc, nd, theta)
    sse_old = np.dot(e.T, e)
    X = np.empty(shape=(len(y), n))
    for i in range(n):
        theta[i] += delta
        e_i = box_jenkins_cal_e(y, u, na, nb, nc, nd, theta) 
        X[:, i] = (e - np.squeeze(e_i)) / delta
        theta[i] -= delta
    A = np.dot(X.T, X)
    g = np.dot(X.T, e)
    return A, g, X, sse_old

def lm_step2_bj(y, u, na, nb, nc, nd, A, theta, mu, g):
    delta_theta = np.matmul(np.linalg.inv(A + mu * np.identity(A.shape[0])), g)
    theta_new = theta + delta_theta
    print(f"Step 2 - theta_new: {theta_new}")
    e_new = box_jenkins_cal_e(y, u, na, nb, nc, nd, theta_new)
    sse_new = np.dot(e_new.T, e_new)
    if np.isnan(sse_new):
        sse_new = 10 ** 10
    return sse_new, delta_theta, theta_new

def lm_step3_bj(y, u, na, nb, nc, nd):
    N = len(y)
    n = na + nb + nc + nd
    mu = 0.01
    mu_max = 1e20
    max_iterations = 10
    delta = 1e-15
    var_error = 0
    covariance_theta_hat = 0
    sse_list = []
    theta = np.zeros(n)

    for iterations in range(max_iterations):
        print(f"Iteration {iterations + 1}")
        A, g, X, sse_old = lm_step1_bj(y, u, na, nb, nc, nd, delta, theta)
        sse_new, delta_theta, theta_new = lm_step2_bj(y, u, na, nb, nc, nd, A, theta, mu, g)
        sse_list.append(sse_old)
        print(f"Iteration {iterations + 1} - SSE old: {sse_old}")
        print(f"Iteration {iterations + 1} - SSE new: {sse_new}")

        if sse_new <= sse_old:
            if np.linalg.norm(np.array(delta_theta), 2) < 1e-15:
                var_error = sse_new / (N - n)
                covariance_theta_hat = var_error * np.linalg.inv(A)
                print(f"Convergence occurred in {iterations + 1} iterations")
                break
            else:
                theta = theta_new
                mu /= 10
        else:
            while sse_new > sse_old:
                mu *= 10
                if mu > mu_max:
                    print("No convergence")
                    break
                sse_new, delta_theta, theta_new = lm_step2_bj(y, u, na, nb, nc, nd, A, theta, mu, g)

        if iterations >= max_iterations:
            print("Max iterations reached")
            break
        var_error = sse_new / (N - n)
        covariance_theta_hat = var_error * np.linalg.inv(A)
    return theta, sse_new, var_error, covariance_theta_hat, sse_list
theta_new, sse_new, var_error, covariance_theta_hat, sse_list = lm_step3_bj(y,X, 1, 1, 1, 1)

print("Estimated parameters:", theta_new)

#%%
# Display the standard deviation 
print(f"Standard deviation: {np.sqrt(var_error)}")
#%%
covariance_theta_hat
# %%
#Confidence Intervals
num_H_root = [1]
den_H_root = [1]
num_G_root = [1]
den_G_root = [1]
nb = nf = nc = nd = 1

print("Confidence Intervals for Coefficients:")
for i in range(nb):  # Coefficients for B
    lower = theta_new[i] - 2 * np.sqrt(covariance_theta_hat[i][i])
    upper = theta_new[i] + 2 * np.sqrt(covariance_theta_hat[i][i])
    print(f"B[{i}] Interval: [{lower:.2f}, {upper:.2f}]")
    num_G_root.append(theta_new[i])

for i in range(nf):  # Coefficients for F
    lower = theta_new[i + nb] - 2 * np.sqrt(covariance_theta_hat[i + nb][i + nb])
    upper = theta_new[i + nb] + 2 * np.sqrt(covariance_theta_hat[i + nb][i + nc])
    print(f"F[{i}] Interval: [{lower:.2f}, {upper:.2f}]")
    den_G_root.append(theta_new[i + nb])

for i in range(nc):  # Coefficients for C
    lower = theta_new[i + nb + nf] - 2 * np.sqrt(covariance_theta_hat[i +  nb + nf][i + nb + nf])
    upper = theta_new[i + nb + nf] + 2 * np.sqrt(covariance_theta_hat[i + nb + nf][i + nb + nf])
    print(f"C[{i}] Interval: [{lower:.2f}, {upper:.2f}]")
    num_H_root.append(theta_new[i + nb + nf])

for i in range(nd):  # Coefficients for D
    lower = theta_new[i + nb + nf + nc] - 2 * np.sqrt(covariance_theta_hat[i + nb + nf + nc][i + nb + nf + nc])
    upper = theta_new[i + nb + nf + nc] + 2 * np.sqrt(covariance_theta_hat[i + nb + nf + nc][i + nb + nf + nc])
    print(f"D[{i}] Interval: [{lower:.2f}, {upper:.2f}]")
    den_H_root.append(theta_new[i + nb + nf + nc])

#%%
#Poles and Zeros
zeros_H = np.roots(num_H_root)
poles_H = np.roots(den_H_root)
zeros_G = np.roots(num_G_root)
poles_G = np.roots(den_G_root)
# Display poles and zeros
print("\nPoles and Zeros:")
for i in range(len(zeros_H)):
    print("The roots of the C(q) are ", zeros_H)
for i in range(len(poles_H)):
    print("The roots of the D(q) are ", poles_H)
for i in range(len(zeros_G)):
    print("The roots of the B(q) are ", zeros_G)
for i in range(len(poles_G)):
    print("The roots of the F(q) are ", poles_G)
# %%
b_pr = [1] + theta_new[:nb]
f_pr = [1] + theta_new[nb:nb+nf]
c_pr = [1] + theta_new[nb+nf:nb+nf+nc]
d_pr = [1] + theta_new[nb+nf+nc:]
def check_num_den_size(num, den):
    if len(num) > len(den):
        den = np.pad(den, (0, len(num) - len(den)), 'constant')
    elif len(den) > len(num):
        num = np.pad(num, (0, len(den) - len(num)), 'constant')
    return num, den
[num_H_pr, den_H_pr] = check_num_den_size(c_pr, d_pr)
[num_G_pr, den_G_pr] = check_num_den_size(b_pr, f_pr)
tf_H_inv = (den_H_pr, num_H_pr, 1)
tf_G_pr = (num_G_pr, den_G_pr, 1)

tf_e2 = (np.convolve(den_H_pr, num_G_pr), np.convolve(num_H_pr,den_G_pr), 1)

num_t_1 = [m-n for m,n in zip(num_H_pr,den_H_pr)]
tf_t_1 = (num_t_1, num_H_pr, 1)

lags = 50
_,y_1_t_1 = dlsim(tf_e2,X)
_,y_1_t_2 = dlsim(tf_t_1,y)
y_hat_t_1 = y_1_t_1 + y_1_t_2

_,e1 = dlsim(tf_H_inv,y)
_,e2 = dlsim(tf_e2,X)
e_pr = e1 + e2
# %%
re = cal_autocorr(e_pr,lags)
Q = len(y)*np.sum(np.square(re[lags:]))*1e-3
DOF = lags - nc - nd
alpha = 0.01
chi_critical_Q = chi2.ppf(1-alpha,DOF)
if Q < chi_critical_Q:
    print("The Q-state is passed - H(q) is accurate")
else:
    print("The Q-state is not passed - H(q) not accurate.")
print(f'Q: {Q}, Critical Q: {chi_critical_Q}')

#%%
S = 1
R = 1
alpha = X
sigma_alpha = np.std(alpha)
sigma_e_pr = np.std(e_pr)
r_alpha_e = cal_corr(alpha, e_pr, K)/(sigma_alpha*sigma_e_pr)
S =len(y)*np.sum(np.square(r_alpha_e)) *1e-3
DOF = lags -nb - nf 
alpha = 0.01 
chi_critical_S = chi2.ppf(1-alpha,DOF)
if S < chi_critical_S:
    print("The S-state is passed - G(q) is accurate")
else:
    print("The S-state is not passed - G(q) is not accurate.")
print(f'S: {S}, Critical S: {chi_critical_S}')
# %%
#I picked ARIMA model for forecasting
#Forecasting function
e = np.random.normal(0, 1, len(y_diff))
T = len(y_diff)



def forecast(y, e, T, h):
    y_hat = []
    len_y = len(y)
    len_e = len(e)
    
    for i in range(h):
        # Check if the required indices exist in y and e
        if (T - 12 >= 0 and T - 12 < len_y) and (T - 24 >= 0 and T - 24 < len_y) and (T - 36 >= 0 and T - 36 < len_y) and (T + i < len_e):
            if i == 0:
                # First forecast: safely calculate using historical values
                forecast = 0.4 * y[T - 12] + 0.63 * y[T - 24] - 0.03 * y[T - 36] + e[T] - 1.51 * e[T - 12] + 0.08 * e[T - 24]
            else:
                # For subsequent forecasts, check if the indices are within bounds
                if (T + i - 12 >= 0 and T + i - 12 < len_y) and (T + i - 24 >= 0 and T + i - 24 < len_y) and (T + i - 36 >= 0 and T + i - 36 < len_y):
                    forecast = 0.4 * y[T + i - 12] + 0.63 * y[T + i - 24] - 0.03 * y[T + i - 36] + e[T + i] - 1.51 * e[T + i - 12] + 0.08 * e[T + i - 24]
        
            y_hat.append(forecast)

    return np.array(y_hat)

# Define your `y` and `e` series along with T and h (you may need to update them as per your data)
y_test = y_test_diff
y = y_diff
print()
# Adjust forecast horizon to match test data length
y_hat_test = forecast(y, e, T, len(y_test))  # Use the length of your test data for forecasting
y_hat_test = pd.Series(arima_model_hat.ravel())

if isinstance(y_test, np.ndarray): 
    y_test = pd.Series(y_test) 

y_hat_test = pd.Series(y_hat_test) 

plt.figure(figsize=(10, 6))  # Set the figure size

# Plot true values (y_test)
plt.plot(y_test.index, y_test, label='True Values', color='blue', linestyle='-', linewidth=2)

# Plot predicted values (y_hat_test)
plt.plot(y_hat_test.index, y_hat_test, label='Predicted Values', color='red', linestyle='--', linewidth=2)

# Add titles and labels
plt.title("Predicted vs True Values", fontsize=16)
plt.xlabel("Time", fontsize=14)
plt.ylabel("Value", fontsize=14)

# Adjust X-axis range to match the range of both y_test and y_hat_test
plt.xlim(min(y_test.index.min(), y_hat_test.index.min()), max(y_test.index.max(), y_hat_test.index.max()))

# Show legend
plt.legend()

# Display the plot
plt.grid(True)
plt.show()
