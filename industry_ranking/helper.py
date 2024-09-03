# June 18, 2024
# Guangda Fei

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats


def ewma_weights(length, half_life):
    """
    Calculate EWMA weights for a given length and half-life.

    Parameters:
    length (int): The number of data points.
    half_life (float): The half-life for the weights.

    Returns:
    np.ndarray: Array of EWMA weights.
    """
    # Calculate the smoothing factor alpha from the half-life
    alpha = 1 - np.exp(-np.log(2) / half_life)

    # Generate the weights
    weights = alpha * (1 - alpha) ** np.arange(length)[::-1]

    # Normalize the weights so they sum to 1
    weights /= weights.sum()

    return weights


def calculate_ewma(data, half_life):
    """
    Calculate the Exponentially Weighted Moving Average (EWMA) for a given data series.

    Parameters:
    data (list or numpy array): The input data series.
    half_life (int): The half-life period for the EWMA. 半衰期

    Returns:
    numpy array: The EWMA values.
    """
    data = np.array(data)
    # Calculate the decay factor lambda
    lambda_ = np.exp(-np.log(2) / half_life)

    # Initialize the EWMA array
    ewma = np.zeros_like(data)

    # Set the first EWMA value to the first data point
    ewma[0] = data[0]

    # Calculate the EWMA for each data point
    for t in range(1, len(data)):
        ewma[t] = lambda_ * ewma[t - 1] + (1 - lambda_) * data[t]
    # ewma = ewma/float(sum(ewma))
    return ewma

# # Example usage
# data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Example data series
# half_life = 2  # Half-life period
#
# ewma_values = calculate_ewma(data, half_life)
# print(ewma_values)


def calculate_annualized_volatility(close_prices, freq=252):
    # Calculate daily returns (percentage change)
    close_prices = np.array(close_prices)
    daily_returns = np.diff(close_prices) / close_prices[:-1]
    # Calculate the standard deviation of daily returns
    daily_volatility = np.std(daily_returns)
    # Annualize the volatility
    annualized_volatility = daily_volatility * np.sqrt(freq)

    return annualized_volatility


def calculate_max_drawdown(close_prices):
    close_prices = np.array(close_prices)
    drawdowns = (close_prices / np.maximum.accumulate(close_prices)) - 1
    # print(drawdowns)
    max_drawdown = np.min(drawdowns)
    return max_drawdown

# # Example usage
# close_prices = np.array([95, 91, 93, 94, 95, 96, 97, 98, 99, 100, 90])  # Example closing prices
#
# annualized_volatility = calculate_annualized_volatility(close_prices)
# max_drawdown = calculate_max_drawdown(close_prices)
#
# print(annualized_volatility)
# print(max_drawdown)


def calculate_annualized_return_daily(close_prices):
    """
    Calculate the annualized return for a given series of closing prices.

    Parameters:
    close_prices (numpy array or list): The input series of closing prices.

    Returns:
    float: The annualized return.
    """
    close_prices = np.array(close_prices)
    n = len(close_prices)
    total_return = close_prices[-1] / close_prices[0] - 1
    annualized_return = (1 + total_return) ** (252 / (n - 1)) - 1  # Assuming 252 trading days in a year

    return annualized_return


def calculate_annualized_return_monthly(close_prices):
    """
    Calculate the annualized return for a given series of closing prices.

    Parameters:
    close_prices (numpy array or list): The input series of closing prices.

    Returns:
    float: The annualized return.
    """
    close_prices = np.array(close_prices)
    n = len(close_prices)
    total_return = close_prices[-1] / close_prices[0] - 1
    annualized_return = (1 + total_return) ** (12 / (n - 1)) - 1  # Assuming 252 trading days in a year

    return annualized_return


def calculate_RSTR(returns, risk_free, half_life=126):
    '''
    动量因子 momentum factor
    至少526天数据 at least 526 days of data
    0-503 to calculate，504-524 relay
    公式formula: https://quantlib.readthedocs.io/api/barra.html
    '''
    T = 504
    L = 21
    # check data size
    returns = np.array(returns)
    risk_free = np.array(risk_free) # array of risk-free returns.
    if len(returns) < 525:
        print("RSTR calculator: date size smaller than minimum requirement")
        return
    if len(returns) != len(risk_free):
        print('input data size different')
        return

    ln_return = np.log(1 + returns) - np.log(1 + risk_free)  # for calculating the RSTR.
    output = np.zeros_like(returns) # initializing output array.
    # weights = ewma_weights(T, half_life)
    for i in range(525, len(returns)): # from day 525 to the end
        needed_ln_return = calculate_ewma(ln_return[i-525:i-21], half_life)
        # output[i] = np.sum(np.array(ln_return[i-525:i-21])*weights)
        output[i] = needed_ln_return[-1]

    return output


def standardize_and_trim(arr, lower_percentile=0.2, upper_percentile=99.8):
    # Ensure arr is a numpy array
    arr = np.asarray(arr).reshape(-1, 1)  # Reshape for StandardScaler if needed

    # Standardize using StandardScaler
    scaler = StandardScaler()
    standardized_arr = scaler.fit_transform(arr).flatten()

    lower_bound = stats.scoreatpercentile(standardized_arr, lower_percentile)
    upper_bound = stats.scoreatpercentile(standardized_arr, upper_percentile)

    # Trim values outside the percentiles
    trimmed_arr = np.clip(standardized_arr, lower_bound, upper_bound)

    print(lower_bound)
    print(upper_bound)

    return trimmed_arr







