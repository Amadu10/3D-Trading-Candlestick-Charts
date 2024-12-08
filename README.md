# 3D-Trading-Candlestick-Charts
A 3D candlestick chart that visualizes volume and price movements in financial markets over time. Each "candle" represents open, high, low, and close (OHLC) prices and volume for a specific period.
import numpy as np
from scipy.fft import fft
from scipy.stats import norm

# Parameters for the European Call Option
S0 = 100   # Initial stock price
K = 100    # Strike price
T = 1.0    # Time to maturity (in years)
r = 0.05   # Risk-free interest rate
sigma = 0.2  # Volatility

# Number of terms for the COS method
N = 256
L = 10  # Truncation range

# Black-Scholes characteristic function
def char_func(u):
    i = 1j
    return np.exp(i * u * (np.log(S0) + (r - 0.5 * sigma**2) * T) - 0.5 * sigma**2 * T * u**2)

# COS method for European Call Option pricing
def cos_method(S0, K, T, r, sigma, N, L):
    # Define the range for integration
    a = np.log(S0) - L * np.sqrt(T)
    b = np.log(S0) + L * np.sqrt(T)

    # Calculate the weights for the COS expansion
    k = np.arange(N)
    u = k * np.pi / (b - a)

    # Compute the characteristic function and cosine terms
    char_values = char_func(u)
    cos_terms = np.cos(np.outer(u, np.linspace(a, b, N)))

    # Payoff function in the transformed domain
    payoff = 2 / (b - a) * (char_values * (np.sin((b - a) * u / 2) / (u / 2))).real
    payoff[0] = payoff[0] / 2  # First term adjustment

    # Compute the COS expansion
    price = np.exp(-r * T) * np.sum(payoff @ cos_terms, axis=0)
    return price

# Calculate the option price using the COS method
price = cos_method(S0, K, T, r, sigma, N, L)
print(f"The European Call Option price using the COS method is: {price:.2f}")
