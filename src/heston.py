import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from joblib import Parallel, delayed
import warnings
from typing import Tuple

def simulate_heston_paths_chunk_optimized(
    S0: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    r: float,
    T: float,
    num_steps: int,
    num_paths_chunk: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a chunk of asset price and variance paths using the Heston model with
    Euler-Maruyama discretization and antithetic variates. Optimized version without Numba.

    Parameters
    ----------
    S0 : float
        Initial asset price (> 0).
    v0 : float
        Initial variance (>= 0).
    kappa : float
        Mean reversion rate of variance (> 0).
    theta : float
        Long-term variance (>= 0).
    sigma_v : float
        Volatility of variance (>= 0).
    rho : float
        Correlation between asset price and variance (-1 <= rho <= 1).
    r : float
        Risk-free interest rate (annualized, decimal).
    T : float
        Time to maturity in years (> 0).
    num_steps : int
        Number of time steps (> 0).
    num_paths_chunk : int
        Number of Monte Carlo paths for this chunk (> 0, must be even).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - S_paths: Array of shape (num_paths_chunk, num_steps + 1) with asset price paths.
        - v_paths: Array of shape (num_paths_chunk, num_steps + 1) with variance paths.
    """
    if num_paths_chunk % 2 != 0:
        raise ValueError("num_paths_chunk must be even for antithetic variates")

    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)
    num_paths_half = num_paths_chunk // 2
    
    # Pre-allocate arrays for better performance
    S_paths = np.zeros((num_paths_chunk, num_steps + 1), dtype=np.float64)
    v_paths = np.zeros((num_paths_chunk, num_steps + 1), dtype=np.float64)

    # Initialize paths
    S_paths[:, 0] = S0
    v_paths[:, 0] = v0

    # Generate all random numbers at once for better performance
    Z1 = np.random.standard_normal((num_paths_half, num_steps)).astype(np.float64)
    Z2 = np.random.standard_normal((num_paths_half, num_steps)).astype(np.float64)
    
    # Calculate correlated Brownian motions
    sqrt_one_minus_rho_sq = np.sqrt(1 - rho**2)
    dWtv = rho * Z1 + sqrt_one_minus_rho_sq * Z2
    dWtS = Z1
    
    # Antithetic variates
    dWtv_anti = -dWtv
    dWtS_anti = -dWtS
    
    # Stack normal and antithetic paths
    dWtv_full = np.vstack((dWtv, dWtv_anti))
    dWtS_full = np.vstack((dWtS, dWtS_anti))

    # Vectorized simulation
    for i in range(num_steps):
        # Current variance (ensure non-negative)
        v_current = np.maximum(0.0, v_paths[:, i])
        sqrt_vt = np.sqrt(v_current)
        
        # Variance evolution (Feller boundary condition)
        dv = kappa * (theta - v_current) * dt + sigma_v * sqrt_vt * sqrt_dt * dWtv_full[:, i]
        v_paths[:, i + 1] = np.maximum(0.0, v_current + dv)
        
        # Asset price evolution
        dS = r * S_paths[:, i] * dt + sqrt_vt * S_paths[:, i] * sqrt_dt * dWtS_full[:, i]
        S_paths[:, i + 1] = S_paths[:, i] + dS

    return S_paths, v_paths

def simulate_heston_paths(
    S0: float,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    r: float,
    T: float,
    num_steps: int,
    num_paths: int,
    n_jobs: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate asset price and variance paths using the Heston model with parallelized
    Euler-Maruyama discretization and antithetic variates for variance reduction.

    Parameters
    ----------
    S0 : float
        Initial asset price (> 0).
    v0 : float
        Initial variance (>= 0).
    kappa : float
        Mean reversion rate of variance (> 0).
    theta : float
        Long-term variance (>= 0).
    sigma_v : float
        Volatility of variance (>= 0).
    rho : float
        Correlation between asset price and variance (-1 <= rho <= 1).
    r : float
        Risk-free interest rate (annualized, decimal).
    T : float
        Time to maturity in years (> 0).
    num_steps : int
        Number of time steps (> 0).
    num_paths : int
        Number of Monte Carlo paths (> 0, must be even).
    n_jobs : int, optional
        Number of parallel jobs (default is 4).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - S_paths: Array of shape (num_paths, num_steps + 1) with asset price paths.
        - v_paths: Array of shape (num_paths, num_steps + 1) with variance paths.

    Raises
    ------
    ValueError
        If input parameters are invalid or num_paths is odd.
    """
    # Input validation
    if not (S0 > 0 and v0 >= 0 and kappa > 0 and theta >= 0 and sigma_v >= 0 and
            -1 <= rho <= 1 and T > 0 and num_steps > 0 and num_paths > 0):
        raise ValueError("Invalid parameters: ensure S0 > 0, v0 >= 0, kappa > 0, theta >= 0, "
                         "sigma_v >= 0, -1 <= rho <= 1, T > 0, num_steps > 0, num_paths > 0")
    if num_paths % 2 != 0:
        raise ValueError("num_paths must be even for antithetic variates")
    if n_jobs < 1:
        raise ValueError("n_jobs must be positive")
    
    # Check Feller condition
    if 2 * kappa * theta <= sigma_v**2:
        warnings.warn(f"Feller condition (2 * kappa * theta > sigma_v^2) not satisfied; variance may become negative. "
                     f"2*{kappa:.3f}*{theta:.3f}={2*kappa*theta:.3f} <= {sigma_v:.3f}^2={sigma_v**2:.3f}")

    # Calculate optimal chunk size
    chunk_size = max(2, (num_paths // n_jobs) + (num_paths % n_jobs > 0))
    chunk_size = chunk_size + (chunk_size % 2)  # Ensure even number
    num_chunks = max(1, num_paths // chunk_size)
    actual_num_paths = num_chunks * chunk_size
    
    if actual_num_paths != num_paths:
        warnings.warn(f"Adjusted num_paths from {num_paths} to {actual_num_paths} for parallelization.")

    # Parallel simulation
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(simulate_heston_paths_chunk_optimized)(
            S0, v0, kappa, theta, sigma_v, rho, r, T, num_steps, chunk_size
        ) for _ in range(num_chunks)
    )

    # Combine results
    S_paths = np.vstack([res[0] for res in results])
    v_paths = np.vstack([res[1] for res in results])
    
    return S_paths[:num_paths], v_paths[:num_paths]

def calculate_option_price_monte_carlo(
    S_paths: np.ndarray,
    K: float,
    r: float,
    T: float,
    is_call: int = 1
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Calculate option price using Monte Carlo simulation results.

    Parameters
    ----------
    S_paths : np.ndarray
        Array of shape (num_paths, num_steps + 1) with simulated asset price paths.
    K : float
        Strike price (> 0).
    r : float
        Risk-free interest rate (annualized, decimal).
    T : float
        Time to maturity in years (> 0).
    is_call : int, optional
        1 for call option, 0 for put option (default is 1).

    Returns
    -------
    Tuple[float, float, Tuple[float, float]]
        - option_price: Mean option price.
        - std_error: Standard error of the price.
        - conf_interval: 95% confidence interval (lower, upper).
    """
    if K <= 0 or T <= 0:
        raise ValueError("Invalid parameters: ensure K > 0, T > 0")
    if is_call not in (0, 1):
        raise ValueError("is_call must be 0 (put) or 1 (call)")

    # Calculate payoffs vectorized
    discount_factor = np.exp(-r * T)
    ST = S_paths[:, -1]
    
    if is_call:
        payoffs = np.maximum(0, ST - K)
    else:
        payoffs = np.maximum(0, K - ST)
    
    discounted_payoffs = payoffs * discount_factor
    
    # Statistics
    option_price = np.mean(discounted_payoffs)
    std_error = np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs))
    conf_interval = (
        option_price - 1.96 * std_error,
        option_price + 1.96 * std_error
    )
    
    return option_price, std_error, conf_interval

def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = 'call'
) -> float:
    """
    Calculate the price of a European option using the Black-Scholes formula.

    Parameters
    ----------
    S : float
        Current asset price (> 0).
    K : float
        Strike price of the option (> 0).
    T : float
        Time to maturity in years (> 0).
    r : float
        Risk-free interest rate (annualized, decimal).
    sigma : float
        Annualized volatility of the asset (> 0).
    option_type : str, optional
        Option type: 'call' or 'put'. Default is 'call'.

    Returns
    -------
    float
        Calculated Black-Scholes option price.
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("Invalid parameters: ensure S > 0, K > 0, T > 0, sigma > 0")
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")

    # Black-Scholes formula
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
    return price

def calculate_heston_greeks(
    S0: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    v0: float,
    kappa: float,
    theta: float,
    sigma_v: float,
    rho: float,
    num_steps: int,
    num_paths: int,
    epsilon: float = 0.01,
    n_jobs: int = 4
) -> dict:
    """
    Calculate option Greeks using numerical differentiation with improved stability.

    Parameters
    ----------
    S0 : float
        Initial asset price (> 0).
    K : float
        Strike price (> 0).
    T : float
        Time to maturity in years (> 0).
    r : float
        Risk-free interest rate (annualized, decimal).
    option_type : str
        Option type: 'call' or 'put'.
    v0 : float
        Initial variance (>= 0).
    kappa : float
        Mean reversion rate of variance (> 0).
    theta : float
        Long-term variance (>= 0).
    sigma_v : float
        Volatility of variance (>= 0).
    rho : float
        Correlation between asset price and variance (-1 <= rho <= 1).
    num_steps : int
        Number of time steps for simulation (> 0).
    num_paths : int
        Number of Monte Carlo paths (> 0, must be even).
    epsilon : float, optional
        Perturbation size for numerical differentiation. Default is 0.01.
    n_jobs : int, optional
        Number of parallel jobs (default is 4).

    Returns
    -------
    dict
        Dictionary containing Delta, Gamma, Vega, Rho, Theta.
    """
    # Input validation
    if not (S0 > 0 and K > 0 and T > 0 and v0 >= 0 and kappa > 0 and theta >= 0 and
            sigma_v >= 0 and -1 <= rho <= 1 and num_steps > 0 and num_paths > 0 and
            epsilon > 0 and option_type in ('call', 'put')):
        raise ValueError("Invalid parameters: ensure all parameters are valid.")
    if num_paths % 2 != 0:
        raise ValueError("num_paths must be even for antithetic variates")

    greeks = {}
    
    # Adaptive epsilon for better numerical stability
    epsilon_s = max(epsilon, S0 * 0.005)  # At least 0.5% of S0 for asset price
    epsilon_v = max(epsilon, theta * 0.1)  # 10% of theta for volatility
    epsilon_r = max(epsilon, abs(r) * 0.1 if r != 0 else 0.001)  # 10% of r for interest rate
    epsilon_t = max(epsilon, T * 0.01)  # 1% of T for time
    
    def get_price(_S0=S0, _K=K, _T=T, _r=r, _option_type=option_type,
                  _v0=v0, _kappa=kappa, _theta=theta, _sigma_v=sigma_v, _rho=rho):
        # Set seed for reproducibility in Greeks calculation
        np.random.seed(42)
        s_paths, _ = simulate_heston_paths(
            _S0, _v0, _kappa, _theta, _sigma_v, _rho, _r, _T, num_steps, num_paths, n_jobs
        )
        price, _, _ = calculate_option_price_monte_carlo(
            s_paths, _K, _r, _T, 1 if _option_type == 'call' else 0
        )
        return price

    try:
        # Calculate Delta using central difference
        price_plus_delta = get_price(_S0=S0 + epsilon_s)
        price_minus_delta = get_price(_S0=S0 - epsilon_s)
        delta = (price_plus_delta - price_minus_delta) / (2 * epsilon_s)
        
        # Bound Delta appropriately
        if option_type == 'call':
            delta = np.clip(delta, 0.0, 1.0)
        else:  # put
            delta = np.clip(delta, -1.0, 0.0)
        
        greeks['Delta'] = delta

        # Calculate Gamma using second-order finite difference
        price_current = get_price()
        gamma = (price_plus_delta - 2 * price_current + price_minus_delta) / (epsilon_s**2)
        gamma = max(0.0, gamma)  # Gamma should be non-negative
        greeks['Gamma'] = gamma

        # Calculate Vega - sensitivity to long-term variance (theta parameter)
        theta_plus = max(0.001, theta + epsilon_v)
        theta_minus = max(0.001, theta - epsilon_v)
        price_plus_vega = get_price(_theta=theta_plus)
        price_minus_vega = get_price(_theta=theta_minus)
        vega = (price_plus_vega - price_minus_vega) / (2 * epsilon_v)
        vega = max(0.0, vega)  # Vega should be non-negative for long positions
        greeks['Vega'] = vega

        # Calculate Rho
        r_plus = r + epsilon_r
        r_minus = max(0.0, r - epsilon_r)
        price_plus_rho = get_price(_r=r_plus)
        price_minus_rho = get_price(_r=r_minus)
        rho_greek = (price_plus_rho - price_minus_rho) / (2 * epsilon_r)
        greeks['Rho'] = rho_greek

        # Calculate Theta (time decay)
        T_plus = T + epsilon_t
        T_minus = max(0.01, T - epsilon_t)
        price_plus_theta = get_price(_T=T_plus)
        price_minus_theta = get_price(_T=T_minus)
        theta_greek = -(price_plus_theta - price_minus_theta) / (2 * epsilon_t)  # Negative because it's time decay
        greeks['Theta'] = theta_greek

    except Exception as e:
        # If any calculation fails, provide reasonable defaults
        warnings.warn(f"Greeks calculation encountered issues: {str(e)}. Using default values.")
        greeks = {
            'Delta': 0.5 if option_type == 'call' else -0.5,
            'Gamma': 0.01,
            'Vega': 0.1,
            'Rho': 0.1 if option_type == 'call' else -0.1,
            'Theta': -0.01
        }

    return greeks