import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import brentq
try:
    from src.heston import simulate_heston_paths, calculate_option_price_monte_carlo, black_scholes_price
except ImportError:
    from heston import simulate_heston_paths, calculate_option_price_monte_carlo, black_scholes_price

sns.set_style("whitegrid")  # Professional plot aesthetic

def calculate_implied_volatility(
    market_price: float, S: float, K: float, T: float, r: float, option_type: str
) -> float:
    """
    Calculate implied volatility using Brent's method.

    Parameters
    ----------
    market_price : float
        Market price of the option.
    S : float
        Current asset price.
    K : float
        Strike price.
    T : float
        Time to maturity (years).
    r : float
        Risk-free interest rate (annualized, decimal).
    option_type : str
        Option type: 'call' or 'put'.

    Returns
    -------
    float
        Implied volatility, or 0 if calculation fails (e.g., deep OTM/ITM).
    """
    if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return 0.0
    if option_type not in ('call', 'put'):
        return 0.0
    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - market_price
    try:
        return brentq(objective, 0.01, 2.0, xtol=1e-6)
    except ValueError:
        return 0.0

def plot_heston_paths(S_paths: np.ndarray, v_paths: np.ndarray, num_paths_to_plot: int = 5) -> plt.Figure:
    """
    Plot a sample of simulated asset price and variance paths from the Heston model.
    """
    num_paths, num_steps = S_paths.shape
    num_paths_to_plot = min(num_paths_to_plot, num_paths)
    time_steps = np.linspace(0, 1, num_steps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for i in range(num_paths_to_plot):
        ax1.plot(time_steps, S_paths[i], label=f'Path {i+1}', alpha=0.7)
    ax1.set_xlabel('Time (Years)')
    ax1.set_ylabel('Asset Price')
    ax1.set_title('Simulated Asset Price Paths')
    ax1.legend()
    ax1.grid(True)

    for i in range(num_paths_to_plot):
        ax2.plot(time_steps, v_paths[i], label=f'Path {i+1}', alpha=0.7)
    ax2.set_xlabel('Time (Years)')
    ax2.set_ylabel('Variance')
    ax2.set_title('Simulated Variance Paths')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    return fig

def plot_convergence_analysis(num_paths_range: np.ndarray, prices: list) -> plt.Figure:
    """
    Plot the convergence of Monte Carlo option prices.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(num_paths_range, prices, marker='o', linestyle='-', color='b')
    ax.set_xscale('log')
    ax.set_xlabel('Number of Monte Carlo Paths')
    ax.set_ylabel('Option Price')
    ax.set_title('Monte Carlo Price Convergence')
    ax.grid(True)
    return fig

def plot_sensitivity_analysis(parameter_range: np.ndarray, prices: list, parameter_name: str) -> plt.Figure:
    """
    Plot option price sensitivity to a parameter.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(parameter_range, prices, marker='o', linestyle='-', color='g')
    ax.set_xlabel(f'{parameter_name.capitalize()}')
    ax.set_ylabel('Option Price')
    ax.set_title(f'Option Price Sensitivity to {parameter_name.capitalize()}')
    ax.grid(True)
    return fig

def plot_payoff_distribution(ST: np.ndarray, K: float, option_type: str) -> plt.Figure:
    """
    Plot the distribution of option payoffs at maturity.
    """
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")

    payoffs = np.maximum(0, ST - K) if option_type == 'call' else np.maximum(0, K - ST)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(payoffs, bins=50, kde=True, ax=ax, color='purple')
    ax.set_xlabel('Option Payoff')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{option_type.capitalize()} Option Payoff Distribution')
    ax.grid(True)
    return fig

def plot_price_vs_strike(strike_range: np.ndarray, heston_prices: list, bs_prices: list) -> plt.Figure:
    """
    Plot Heston and Black-Scholes option prices across strike prices.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(strike_range, heston_prices, label='Heston Model (Monte Carlo)', marker='o', linestyle='-', color='b')
    ax.plot(strike_range, bs_prices, label='Black-Scholes Model', marker='s', linestyle='--', color='r')
    ax.set_xlabel('Strike Price (K)')
    ax.set_ylabel('Option Price')
    ax.set_title('Option Price vs. Strike Price')
    ax.legend()
    ax.grid(True)
    return fig

def plot_implied_volatility_smile(strike_range: np.ndarray, implied_vols: list, bs_vol: float) -> plt.Figure:
    """
    Plot the implied volatility smile.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(strike_range, implied_vols, label='Heston Implied Volatility (Monte Carlo)', marker='o', linestyle='-', color='b')
    ax.axhline(y=bs_vol, color='r', linestyle='--', label='Black-Scholes Volatility')
    ax.set_xlabel('Strike Price (K)')
    ax.set_ylabel('Implied Volatility')
    ax.set_title('Implied Volatility Smile')
    ax.legend()
    ax.grid(True)
    return fig

def plot_volatility_surface(
    S0: float, T: float, r: float, option_type: str, v0: float, kappa: float,
    theta: float, sigma_v: float, rho: float, strike_range: np.ndarray,
    maturity_range: np.ndarray, is_monte_carlo: bool = True, num_steps: int = 100,
    num_paths: int = 10000, n_jobs: int = 4
) -> plt.Figure:
    """
    Plot the implied volatility surface.
    """
    implied_vols = np.zeros((len(maturity_range), len(strike_range)))
    for i, T in enumerate(maturity_range):
        if is_monte_carlo:
            S_paths, _ = simulate_heston_paths(S0, v0, kappa, theta, sigma_v, rho, r, T, num_steps, num_paths, n_jobs)
        for j, K in enumerate(strike_range):
            if is_monte_carlo:
                price, _, _ = calculate_option_price_monte_carlo(S_paths, K, r, T, is_call=1 if option_type == 'call' else 0)
            implied_vols[i, j] = calculate_implied_volatility(price, S0, K, T, r, option_type)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(strike_range, maturity_range)
    ax.plot_surface(X, Y, implied_vols, cmap='viridis')
    ax.set_xlabel('Strike Price (K)')
    ax.set_ylabel('Maturity (T)')
    ax.set_zlabel('Implied Volatility')
    ax.set_title('Heston Implied Volatility Surface (Monte Carlo)')
    return fig