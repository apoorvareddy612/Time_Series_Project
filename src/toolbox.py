import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
import statsmodels.api as sm
import seaborn as sns
import numpy.linalg as LA
from scipy import signal
from scipy.stats import chi2
from statsmodels.tsa.seasonal import STL 



def Cal_rolling_mean_var(data, label):
    """
    Function to calculate and plot the rolling mean and variance.
    
    Parameters:
    data (array-like): The data for which to calculate the rolling statistics.
    label (str): The label for the plot (e.g., 'Sales', 'AdBudget', 'GDP').
    """
    n = len(data)
    
    # Initialize lists to store rolling mean and variance
    rolling_means = []
    rolling_variances = []
    
    # Loop through the data and calculate rolling mean and variance
    for i in range(1, n + 1):
        current_data = data[:i]
        rolling_means.append(np.mean(current_data))
        rolling_variances.append(np.var(current_data, ddof=1))  # ddof=1 for sample variance

    # Plot rolling mean and variance
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Rolling mean plot
    axs[0].plot(range(0, n), rolling_means, color='blue')
    axs[0].set_title(f'Rolling Mean of {label}')
    axs[0].set_xlabel('Samples')
    axs[0].set_ylabel('Magnitude')
    axs[0].grid(True)
    
    # Rolling variance plot
    axs[1].plot(range(0, n), rolling_variances, color='red')
    axs[1].set_title(f'Rolling Variance of {label}')
    axs[1].set_xlabel('Samples')
    axs[1].set_ylabel('Magnitude')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()


# Check for stationarity using ADF test
def adf_test(series):
    result = adfuller(series)
    print('ADF Test Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    if result[1] < 0.05:
        print("Series is likely stationary.")
    else:
        print("Series is likely non-stationary.")

# Check for stationarity using KPSS test
def kpss_test(series):
    result = kpss(series, regression='c', nlags="auto")
    print('KPSS Test Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[3])
    if result[1] < 0.05:
        print("Series is likely non-stationary.")
    else:
        print("Series is likely stationary.")



def differencing(y, order, s=1):
    diff = []
    n = len(y)
    for i in range(n):
        if i-s < 0 or y[i] is None or y[i-s] is None:
            diff.append(None)
        else:
            diff.append(y[i] - y[i - s])
    if order == 1:
        return np.array(diff)
    return differencing(pd.Series(diff), order-1, s)



def check_stationarity(y, title):
    y = pd.Series(y.ravel())
    Cal_rolling_mean_var(y, title)
    adf_test(y)
    kpss_test(y)
    ACF_PACF_Plot(y, 50)


def logTransform(y):
    log_y = np.log(y)
    log_y = log_y.dropna().reset_index(drop=True)
    return log_y


# autocorrelation
def cal_autocorr(Y, lags):  # default value is set to None, i.e. the case when we don't need subplots
    T = len(Y)
    ry = []
    den = 0
    ybar = np.mean(Y)
    for y in Y:  # since denominator is constant for every iteration, we calculate it only once and store it.
        den = den + (y - ybar) ** 2

    for tau in range(lags+1):
        num = 0
        for t in range(tau, T):
            num = num + (Y[t] - ybar) * (Y[t - tau] - ybar)
        ry.append(num / den)

    ryy = ry[::-1]
    Ry = ryy[:-1] + ry  # to make the plot on both sides, reversed the list and added to the original list

    return Ry


def cal_error_MSE(y, yhat, skip=0):
    y = np.array(y)
    yhat = np.array(yhat)
    error = []
    error_square = []
    n = len(y)
    for i in range(n):
        if yhat[i] is None:
            error.append(None)
            error_square.append(None)
        else:
            error.append(y[i]-yhat[i])
            error_square.append((y[i]-yhat[i])**2)
    mse = 0
    for i in range(skip, n):
        mse = mse + error_square[i]

    mse = mse/(n-skip)

    return error, error_square, np.round(mse, 2)


def plot_forecasting_models(ytrain, ytest, yhatTest, title, axs=None):
    if axs is None:
        axs = plt
    x = np.arange(1, len(ytrain)+len(ytest)+1)
    x1 = x[:len(ytrain)]
    x2 = x[len(ytrain):]
    axs.plot(ytrain.index, ytrain, color='r', label='train')
    axs.plot(ytest.index, ytest, color='g', label='test')
    axs.plot(ytest.index, yhatTest, color='b', label='h step')
    if axs is plt:
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()
    else:
        axs.set_xlabel('Time')
        axs.set_ylabel('Values')
        axs.set_title(title)
        axs.grid()
        axs.legend()



def average_forecasting(ytrain, ytest):
    # Calculate the one-step prediction for each time step in the training set
    one_step_prediction = [0]  # First value is always 0 since there's no previous data
    running_sum = 0

    for i in range(len(ytrain)):
        if i > 0:  # Avoid division by zero
            one_step_prediction.append(running_sum / i)
        running_sum += ytrain[i]  # Update running sum

    # Calculate the average of the entire training set
    average_forecast = np.mean(ytrain)

    # Prepare h-step forecasts (using the average of the training set)
    h_step_forecasts = np.array([average_forecast] * len(ytest))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(ytrain) + 1), ytrain, marker='o', color='blue', label='Training Set')
    plt.plot(range(len(ytrain) + 1, len(ytrain) + len(ytest) + 1), ytest, marker='o', color='orange', label='Testing Set')
    plt.plot(range(len(ytrain) + 1, len(ytrain) + len(h_step_forecasts) + 1), h_step_forecasts, marker='x', color='green', linestyle='--', label='H-Step Forecast')

    # Adding titles and labels
    plt.title('Time Series Forecasting using Average Method')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid()
    plt.xticks(np.arange(1, len(ytrain) + len(ytest) + 1))
    plt.show()
    
    return one_step_prediction, h_step_forecasts

def Naive_forecasting(xtrain, xtest):
    # Forecast the last observed value for all h-steps
    h_step_forecast = [xtrain[-1]] * len(xtest)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(xtrain) + 1),xtrain, label='Training Set', marker='o', color='blue')
    plt.plot(range(len(xtrain) + 1, len(xtrain) + len(xtest) + 1), xtest, label='Testing Set', marker='x', color='green')
    plt.plot(range(len(xtrain)+1, len(xtrain) + len(h_step_forecast)+1), h_step_forecast, label='h-step Forecast', linestyle='--', color='red')
    plt.title('Time Series Forecasting using Naive Method')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.grid()
    plt.show()

    return xtrain[-1], h_step_forecast


def drift_method(t, train_data):
    if t == 1 or t == 2:
        return 0
    else:
        drift = train_data[t-2] + ((train_data[t-2] - train_data[0]) / (t-2))
    return drift

def drift_forecasting(xtrain, xtest):
    # Calculate drift based on the training set
    T = len(xtrain)
    drift = (xtrain[-1] - xtrain[0]) / (T - 1)
    h_step_forecast = [xtrain[-1] + (i + 1) * drift for i in range(len(xtest))]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(xtrain) + 1), xtrain, label='Training Set', marker='o', color='blue')
    plt.plot(range(len(xtrain) + 1, len(xtrain) + len(xtest) + 1), xtest, label='Testing Set', marker='x', color='green')
    plt.plot(range(len(xtrain) + 1, len(xtrain) + len(h_step_forecast) + 1), h_step_forecast, label='h-step Forecast (Drift)', linestyle='--', color='red')
    plt.title('Time Series Forecasting using Drift Method')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.grid()
    plt.show()

    return drift, h_step_forecast


def ses_forecasting(ytrain, ytest, initial_value, alpha):
    # Initialize predictions with the initial value
    ses_predictions = [initial_value]

    # Apply SES recursively for the training set
    for t in range(1, len(ytrain)):
        ses_predictions.append(alpha * ytrain[t] + (1 - alpha) * ses_predictions[-1])

    # Forecast for the test set using the last SES value
    h_step_forecast = [ses_predictions[-1]] * len(ytest)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(ytrain) + 1), ytrain, label='Training Data', marker='o', color='blue')
    plt.plot(range(len(ytrain) + 1, len(ytrain) + len(ytest) + 1), ytest, label='Testing Data', marker='x', color='green')
    plt.plot(range(len(ytrain) + 1, len(ytrain) + len(h_step_forecast) + 1), h_step_forecast, label='h-Step Forecast (SES)', linestyle='--', color='red')
    plt.title('Time Series Forecasting using SES Method')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.grid()
    plt.show()

    return ses_predictions[:-1], h_step_forecast



def cal_Q_value(y, title, lags=5):
    # title = 'Average forecasting train data'
    acf = cal_autocorr(y, lags)
    sum_rk = 0
    T = len(y)
    for i in range(1, lags+1):
        sum_rk += acf[i]**2
    Q = T * sum_rk
    # if Q < Q* then white residual
    return Q


def standardize(train, test):
    columns = train.columns
    X_train = train.copy()
    X_test = test.copy()
    for col in columns:
        xbar = np.mean(X_train[col])
        std = np.std(X_train[col])
        X_train[col] = (X_train[col] - xbar) / std
        X_test[col] = (X_test[col] - xbar) / std

    return X_train, X_test


def normal_equation_LSE(X, Y):
    normal_eqn = ((np.linalg.inv(X.T@X))@X.T)@Y
    return normal_eqn


def moving_average_decomposition(arr, order):
    m = order
    k = (m - 1) // 2
    res = []
    len_data = len(arr)

    if m == 2:
        res.append(None)
    else:
        for i in range(k):
            res.append(None)

    for i in range(len_data - m + 1):
        s = 0
        flag = True
        for j in range(i, i+m):
            if arr[j] is None:
                flag = False
                break
            s += arr[j]
        if flag is False:
            res.append(None)
        else:
            res.append(s/m)
    if m % 2 == 0 and m != 2:
        for i in range(k+1):
            res.append(None)
    elif m != 2:
        for i in range(k):
            res.append(None)

    return res


def create_process_general_AR(order, N, a):
    na = order
    np.random.seed(6313)
    mean = 0
    std = 1
    e = np.random.normal(mean, std, N)
    y = np.zeros(len(e))
    coef = np.zeros(na)
    for t in range(len(e)):
        sum_coef = 0
        y[t] = e[t]
        for i in range(1, na+1):
            if t-i < 0:
                break
            else:
                sum_coef += coef[i-1]*y[t-i]
        if t < na:
            coef[t] = a[t]

        y[t] -= sum_coef

    return y


def whitenoise(mean, std, samples, seed=0):
    np.random.seed(seed)
    return np.random.normal(mean, std, samples)


def ACF_PACF_Plot(y, lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()


def calc_val(Ry, J, K):

    den = np.zeros((K, K))

    for k in range(K):
        row = np.zeros(K)
        for i in range(K):
            row[i] = Ry[np.abs(J + k - i)]
        den[k] = row
    # num = den.copy()
    col = np.zeros(K)
    for i in range(K):
        col[i] = Ry[J+i+1]

    num = np.concatenate((den[:, :-1], col.reshape(-1, 1)), axis=1)
    num = np.array(num)
    den = np.array(den)

    if np.linalg.det(den) == 0:
        return np.inf
    if np.abs(np.linalg.det(num)/np.linalg.det(den)) < 0.00001:
        return 0
    return np.linalg.det(num)/np.linalg.det(den)


def cal_gpac(Ry, J=7, K=7):
    gpac_arr = np.zeros((J, K))
    gpac_arr.fill(None)
    for k in range(1, K):
        for j in range(J):
            gpac_arr[j][k] = calc_val(Ry, j, k)
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
    ax = sns.heatmap(df, annot=True, fmt='0.3f')  # cmap='Pastel2'
    plt.title('Generalized Partial Autocorrelation (GPAC) Table')
    plt.tight_layout()
    plt.show()
    print(df)


def check_AIC_BIC_adjR2(x, y):
    columns = x.columns
    res_df = pd.DataFrame(columns=['Removing column', 'AIC', 'BIC', 'AdjR2'])
    for col in columns:
        temp_df = x.copy()
        temp_df = temp_df.drop([col], axis=1)
        res = sm.OLS(y, temp_df).fit()
        res_df.loc[len(res_df.index)] = [col, res.aic, res.bic, res.rsquared_adj]

    res_df = res_df.sort_values(by=['AIC'], ascending=False)
    return res_df


# LM algo and supporting functions

def cal_e(num, den, y):
    system = (num, den, 1)
    _, e = signal.dlsim(system, y)
    return e


def num_den(theta, na, nb):
    theta = theta.ravel()
    num = np.concatenate(([1], theta[:na]))
    den = np.concatenate(([1], theta[na:]))
    max_len = max(len(num), len(den))
    num = np.pad(num, (0, max_len - len(num)), 'constant')
    den = np.pad(den, (0, max_len - len(den)), 'constant')
    return num, den


def cal_gradient_hessian(y, e, theta, na, nb):
    delta = 0.000001
    X = np.empty((len(e), 0))
    for i in range(len(theta)):
        temp_theta = theta.copy()
        temp_theta[i] = temp_theta[i] + delta
        num, den = num_den(temp_theta, na, nb)
        e_new = cal_e(num, den, y)
        x_temp = (e - e_new)/delta
        X = np.hstack((X, x_temp))

    A = np.dot(X.T, X)
    g = np.dot(X.T, e)
    return A, g


def SSE(theta, y, na, nb):
    num, den = num_den(theta, na, nb)
    e = cal_e(num, den, y)
    return np.dot(e.T, e)


def LM(y, na, nb):
    epoch = 0
    epochs = 50
    theta = np.zeros(na + nb)
    mu = 0.01
    n = len(theta)
    N = len(y)
    mu_max = 1e+20
    sse_array = []
    while epoch < epochs:
        sse_array.append(SSE(theta, y, na, nb).ravel())
        num, den = num_den(theta, na, nb)
        e = cal_e(num, den, y)
        A, g = cal_gradient_hessian(y, e, theta, na, nb)
        del_theta = LA.inv(A + mu*np.identity(A.shape[0])) @ g
        theta_new = theta.reshape(-1, 1) + del_theta
        sse_new = SSE(theta_new.ravel(), y, na, nb)
        sse_old = SSE(theta.ravel(), y, na, nb)
        if sse_new[0][0] < sse_old[0][0]:
            if LA.norm(del_theta) < 1e-3:
                theta_hat = theta_new.copy()
                sse_array.append(SSE(theta_new, y, na, nb).ravel())
                variance_hat = SSE(theta_new.ravel(), y, na, nb)/(N-n)
                covariance_hat = variance_hat * LA.inv(A)
                return theta_hat, variance_hat, covariance_hat, sse_array
            else:
                mu = mu/10
        while SSE(theta_new.ravel(), y, na, nb) >= SSE(theta.ravel(), y, na, nb):
            mu = mu*10
            # theta = theta_new.copy()
            if mu > mu_max:
                print('Error')
                break
            del_theta = LA.inv(A + mu * np.identity(A.shape[0])) @ g
            theta_new = theta.reshape(-1, 1) + del_theta
        epoch += 1
        theta = theta_new.copy()
    return


def removeNA(y):
    y = pd.Series(y)
    return y.dropna()


def STL_analysis(y, periods):
    y = pd.Series(y)
    stl = STL(y, period=periods)
    res = stl.fit()
    fig = res.plot()
    plt.suptitle('Trend, seasonality, and remainder plot')
    plt.xlabel('Time')
    plt.tight_layout()
    plt.show()

    T = res.trend
    S = res.seasonal
    R = res.resid

    plt.figure(figsize=[16, 8])
    plt.plot(y, label='Original')
    plt.plot(y - S, label='Seasonally adjusted')
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Seasonally adjusted vs original curve')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=[16, 8])
    plt.plot(y, label='Original')
    plt.plot(y - T, label='Detrended')
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Detrended vs original curve')
    plt.legend()
    plt.tight_layout()
    plt.show()

    Ft = max(0, 1 - np.var(R) / (np.var(T + R)))
    print("Strength of Trend for this dataset is ", Ft)

    seas = 1 - np.var(R) / (np.var(S + R))
    Fs = max(0, 1 - np.var(R) / (np.var(S + R)))
    print("Strength of seasonality for this dataset is ", Fs)

    return T, S, R




def reverse_transform_and_plot(prediction, y_train, y_test, title):
    forecast = []
    s = 365
    for i in range(len(y_test)):
        if i < s:
            forecast.append(prediction[i] + y_train[- s + i])
        else:
            temp = i - s
            forecast.append(prediction[i] + forecast[temp])
    forecast = pd.Series(forecast)
    forecast.index = prediction.index
    plt.plot(y_train.index, y_train.values, label='Train')
    plt.plot(forecast.index, forecast.values, label='Forecast')
    plt.plot(y_test.index, y_test.values, label='Actual Test Data')
    str = f'Predictions using {title}'
    plt.title(str)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return forecast

# Function to display poles and zeros, with cancellation check
def display_poles_zeros(num, den):
    poles = np.roots(den)
    zeros = np.roots(num)
    print("\nPoles:", poles.round(3))
    print("Zeros:", zeros.round(3))
    
    # Check for pole-zero cancellations
    cancellation = any(np.isclose(pole, zero, atol=1e-3) for pole in poles for zero in zeros)
    print("Zero-pole cancellation?", "Yes" if cancellation else "No")


# Function to perform model fitting, prediction, and residual analysis
def prediction(y, na, nb, d=0, forecast_steps=10):
    np.random.seed(6313)
    lags = 25
    N = len(y)
    
    # Fit the ARMA or ARIMA model
    model = sm.tsa.ARIMA(y, order=(na, d, nb)).fit()
    
    # Generate predictions (one-step ahead forecast for all samples)
    model_hat = model.predict(start=0, end=N - 1)
    
    # Calculate residuals (difference between true and predicted values)
    residuals = y - model_hat
    
    # Whiteness Chi-square Test for residuals
    re = cal_autocorr(np.array(residuals), lags)
    Q = len(y) * np.sum(np.square(re[1:]))
    DOF = lags - na - nb
    alfa = 0.01  # Significance level
    chi_critical = chi2.ppf(1 - alfa, DOF)
    
    # Print the whiteness test results
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
    
    # Covariance of the estimated parameters (model parameters)
    param_covariance = model.cov_params()
    print(f"Estimated covariance of the parameters: \n{param_covariance}")
    
    # Bias check: mean of residuals (should be close to 0 for unbiased model)
    mean_residual = np.mean(residuals)
    print(f"Mean of residuals (should be close to 0 for unbiased model): {mean_residual:.4f}")
    
    # Forecast errors (out-of-sample prediction errors)
    forecast_values = model.forecast(steps=forecast_steps)
    forecast_errors = forecast_values - model.forecast(steps=forecast_steps)[0]
    forecast_variance = np.var(forecast_errors)
    
    # Compare variance of residual errors vs forecast errors
    print(f"Variance of residual errors (in-sample): {residual_variance:.4f}")
    print(f"Variance of forecast errors (out-of-sample): {forecast_variance:.4f}")

    # Model simplification: Zero-pole cancellation
    ar_params = model.arparams if hasattr(model, 'arparams') else []
    ma_params = model.maparams if hasattr(model, 'maparams') else []
    num = np.r_[1, -np.array(ar_params)]  # Numerator coefficients (AR part)
    den = np.r_[1, np.array(ma_params)]   # Denominator coefficients (MA part)
    
    # Display poles and zeros
    display_poles_zeros(num, den)
    
    # Display final coefficient confidence intervals
    conf_intervals = model.conf_int(alpha=0.01)  # 99% confidence intervals
    print("Final coefficient confidence intervals:")
    print(conf_intervals)
    

    return model, forecast_values


