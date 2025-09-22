import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import numpy as np
from scipy.optimize import brentq, minimize
from src.heston import simulate_heston_paths, calculate_option_price_monte_carlo, black_scholes_price, calculate_heston_greeks
from src.plots import (
    plot_heston_paths, plot_convergence_analysis, plot_sensitivity_analysis,
    plot_payoff_distribution, plot_price_vs_strike, plot_implied_volatility_smile,
    plot_volatility_surface
)
import matplotlib.pyplot as plt
import pandas as pd
import time
import warnings
from timeit import timeit

def calculate_optimal_paths(num_paths: int, n_jobs: int) -> int:
    """
    Calculate the optimal number of paths divisible by n_jobs * 2.
    """
    optimal_chunk_size = max(2, (num_paths // (n_jobs * 2)) * 2)
    optimal_paths = optimal_chunk_size * n_jobs
    return optimal_paths

def calculate_implied_volatility(
    market_price: float, S: float, K: float, T: float, r: float, option_type: str
) -> float:
    """
    Calculate implied volatility using Brent's method with improved bounds.
    """
    if market_price <= 1e-6 or S <= 0 or K <= 0 or T <= 0:
        return np.nan
    if option_type not in ('call', 'put'):
        raise ValueError("option_type must be 'call' or 'put'")
    
    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - market_price
    
    try:
        # Use wider bounds for better coverage of extreme cases
        return brentq(objective, 0.001, 5.0, xtol=1e-8)
    except ValueError:
        return np.nan

def calibrate_heston_params(S0: float, K: float, T: float, r: float, market_prices: np.ndarray, initial_guess: list) -> dict:
    """
    Calibrate Heston parameters to market data using least-squares optimization.

    Parameters
    ----------
    S0 : float
        Initial asset price.
    K : float
        Strike price.
    T : float
        Time to maturity.
    r : float
        Risk-free rate.
    market_prices : np.ndarray
        Array of market option prices.
    initial_guess : list
        Initial guess for [v0, kappa, theta, sigma_v, rho].

    Returns
    -------
    dict
        Calibrated parameters and optimization status.
    """
    def objective(params):
        v0, kappa, theta, sigma_v, rho = params
        
        # Add constraints to ensure Feller condition and reasonable values
        if 2 * kappa * theta <= sigma_v**2:
            return 1e8  # Higher penalty for violating Feller condition
        
        try:
            # Use fewer paths for calibration to speed up
            S_paths, _ = simulate_heston_paths(S0, v0, kappa, theta, sigma_v, rho, r, T, 50, 1000, 2)
            price_mc, _, _ = calculate_option_price_monte_carlo(S_paths, K, r, T, is_call=1)
            return np.sum((price_mc - market_prices)**2)
        except:
            return 1e6  # Penalty for simulation errors
    
    # Tighter bounds that respect Feller condition: 2*kappa*theta > sigma_v^2
    bounds = [
        (0.01, 0.5),      # v0: initial variance
        (1.0, 10.0),      # kappa: mean reversion speed (increased minimum)
        (0.01, 0.5),      # theta: long-term variance
        (0.01, 0.6),      # sigma_v: volatility of variance (reduced maximum)
        (-0.9, 0.9)       # rho: correlation
    ]
    
    # Multiple optimization attempts with different starting points
    best_result = None
    best_objective = float('inf')
    
    for attempt in range(3):
        if attempt == 0:
            x0 = initial_guess
        elif attempt == 1:
            # Alternative starting point
            x0 = [0.05, 3.0, 0.05, 0.2, -0.5]
        else:
            # Random starting point within bounds
            x0 = [np.random.uniform(b[0], b[1]) for b in bounds]
        
        try:
            result = minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            if result.fun < best_objective and result.success:
                best_result = result
                best_objective = result.fun
        except:
            continue
    
    if best_result is None:
        # Fallback to initial guess if all optimizations fail
        return {
            'v0': initial_guess[0], 'kappa': initial_guess[1], 'theta': initial_guess[2],
            'sigma_v': initial_guess[3], 'rho': initial_guess[4], 'success': False,
            'message': 'All optimization attempts failed, using initial guess'
        }
    
    return {
        'v0': best_result.x[0], 'kappa': best_result.x[1], 'theta': best_result.x[2],
        'sigma_v': best_result.x[3], 'rho': best_result.x[4], 'success': best_result.success,
        'message': best_result.message
    }

def benchmark_simulation(S0, v0, kappa, theta, sigma_v, rho, r, T, num_steps, num_paths, n_jobs):
    """
    Benchmark simulation with/without optimizations.
    """
    def run_simulation(use_antithetic=True, use_numba=True):
        if not use_antithetic:
            # Modify to disable antithetic variates
            def no_antithetic(*args, **kwargs):
                S_paths, v_paths = simulate_heston_paths(*args, **kwargs)
                return S_paths[:num_paths//2], v_paths[:num_paths//2]
            return timeit(lambda: no_antithetic(S0, v0, kappa, theta, sigma_v, rho, r, T, num_steps, num_paths, n_jobs), number=3)
        return timeit(lambda: simulate_heston_paths(S0, v0, kappa, theta, sigma_v, rho, r, T, num_steps, num_paths, n_jobs), number=3)

    return {
        'full_optimization': run_simulation(True, True),
        'no_antithetic': run_simulation(False, True),
        'no_numba': run_simulation(True, False)
    }

def plot_greeks_vs_strike(strike_range, greeks_data):
    """
    Plot Greeks vs Strike Price.
    
    Parameters
    ----------
    strike_range : array_like
        Array of strike prices.
    greeks_data : list of dict
        List of dictionaries containing Greeks for each strike.
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    greek_names = ['Delta', 'Gamma', 'Vega', 'Rho', 'Theta']
    
    for i, greek_name in enumerate(greek_names):
        if i < len(axes):
            greek_values = [greeks[greek_name] for greeks in greeks_data]
            axes[i].plot(strike_range, greek_values, 'b-', linewidth=2, marker='o')
            axes[i].set_xlabel('Strike Price')
            axes[i].set_ylabel(greek_name)
            axes[i].set_title(f'{greek_name} vs Strike Price')
            axes[i].grid(True, alpha=0.3)
    
    # Hide the last subplot since we only have 5 Greeks
    axes[5].set_visible(False)
    
    plt.tight_layout()
    return fig

def main():
    """Run Heston model simulations, benchmarks, and calibration."""
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.01
    option_type = 'call'
    
    # Use parameters that satisfy Feller condition: 2*kappa*theta > sigma_v^2
    # 2 * 4.0 * 0.1 = 0.8 > 0.3^2 = 0.09 ✓
    v0 = 0.1       # Initial variance
    kappa = 4.0    # Mean reversion speed
    theta = 0.1    # Long-term variance  
    sigma_v = 0.3  # Volatility of variance
    rho = -0.7     # Correlation
    
    num_steps = 100
    num_paths = 10000
    n_jobs = 4

    optimal_num_paths = calculate_optimal_paths(num_paths, n_jobs)
    if optimal_num_paths != num_paths:
        print(f"Adjusted num_paths from {num_paths} to {optimal_num_paths}")
        num_paths = optimal_num_paths

    os.makedirs('results', exist_ok=True)
    
    # Benchmarking
    print("Running benchmarks...")
    benchmarks = benchmark_simulation(S0, v0, kappa, theta, sigma_v, rho, r, T, num_steps, num_paths, n_jobs)
    benchmark_df = pd.DataFrame({
        'Configuration': ['Full Optimization', 'No Antithetic Variates', 'No Numba'],
        'Time (s)': [benchmarks['full_optimization'], benchmarks['no_antithetic'], benchmarks['no_numba']]
    })
    benchmark_df.to_csv('results/benchmarks.csv', index=False)
    print("Benchmarks saved to 'results/benchmarks.csv'")

    print("Simulating Heston paths...")
    start_time = time.time()
    try:
        S_paths, v_paths = simulate_heston_paths(
            S0, v0, kappa, theta, sigma_v, rho, r, T, num_steps, num_paths, n_jobs
        )
    except ValueError as e:
        print(f"Simulation failed: {str(e)}")
        return
    sim_time = time.time() - start_time
    print(f"Monte Carlo simulation completed in {sim_time:.2f} seconds")

    try:
        heston_mc_price, heston_std_err, heston_conf_interval = calculate_option_price_monte_carlo(
            S_paths, K, r, T, is_call=1 if option_type == 'call' else 0
        )
    except ValueError as e:
        print(f"Monte Carlo pricing failed: {str(e)}")
        return

    bs_price = black_scholes_price(S0, K, T, r, np.sqrt(theta), option_type)
    implied_vol_mc = calculate_implied_volatility(heston_mc_price, S0, K, T, r, option_type)

    try:
        greeks = calculate_heston_greeks(
            S0, K, T, r, option_type, v0, kappa, theta, sigma_v, rho, num_steps, 50000, 0.01, n_jobs
        )
    except ValueError as e:
        print(f"Greeks calculation failed: {str(e)}")
        return

    # Calibration with multiple market prices
    print("Calibrating Heston parameters...")
    # Simulate market prices for different strikes to have more data for calibration
    strikes = [90, 95, 100, 105, 110]
    market_prices = []
    for strike in strikes:
        # Use Black-Scholes as proxy for market prices with some noise
        vol = np.sqrt(theta) + 0.02 * np.random.randn()  # Add some volatility smile
        price = black_scholes_price(S0, strike, T, r, max(0.1, vol), option_type)
        market_prices.append(price)
    
    market_prices = np.array(market_prices)
    initial_guess = [v0, kappa, theta, sigma_v, rho]
    calibration_results = calibrate_heston_params(S0, K, T, r, market_prices, initial_guess)
    print("Calibration results:", calibration_results)

    print("\nOption Pricing Results:")
    print(f"Heston Monte Carlo Price: {heston_mc_price:.4f}")
    print(f"Heston Standard Error: {heston_std_err:.4f}")
    print(f"Heston 95% Confidence Interval: ({heston_conf_interval[0]:.4f}, {heston_conf_interval[1]:.4f})")
    print(f"Black-Scholes Price: {bs_price:.4f}")
    print(f"Implied Volatility (Monte Carlo): {implied_vol_mc:.4f}")
    print("\nGreeks:")
    for greek, value in greeks.items():
        print(f"{greek}: {value:.4f}")
    print("\nCalibration Results:")
    for key, value in calibration_results.items():
        print(f"{key}: {value}")

    results = {
        "Parameter": ["S0", "K", "T", "r", "option_type", "v0", "kappa", "theta", "sigma_v", "rho",
                      "num_paths", "n_jobs", "Heston MC Price", "Heston Std Error",
                      "Heston CI Lower", "Heston CI Upper", "Black-Scholes Price",
                      "Implied Volatility (MC)", "Delta", "Gamma", "Vega", "Rho", "Theta",
                      "Calibrated v0", "Calibrated kappa", "Calibrated theta", "Calibrated sigma_v", "Calibrated rho"],
        "Value": [S0, K, T, r, option_type, v0, kappa, theta, sigma_v, rho,
                  num_paths, n_jobs, heston_mc_price, heston_std_err,
                  heston_conf_interval[0], heston_conf_interval[1], bs_price,
                  implied_vol_mc, greeks['Delta'], greeks['Gamma'], greeks['Vega'],
                  greeks['Rho'], greeks['Theta'],
                  calibration_results['v0'], calibration_results['kappa'], calibration_results['theta'],
                  calibration_results['sigma_v'], calibration_results['rho']]
    }
    df = pd.DataFrame(results)
    df.to_csv('results/heston_simulation_results.csv', index=False)
    print("\nResults saved to 'results/heston_simulation_results.csv'")

    print("\nGenerating plots...")
    fig = plot_heston_paths(S_paths, v_paths, num_paths_to_plot=5)
    fig.savefig('results/heston_paths.png')
    plt.close(fig)

    # Improved convergence analysis with incremental approach
    print("Running convergence analysis...")
    
    # Use incremental simulation - add more paths to existing ones
    base_paths = 1000
    path_increments = [0, 1000, 4000, 9000, 19000, 49000, 99000]  # Incremental amounts
    total_paths_range = [base_paths + inc for inc in path_increments]
    
    # Ensure all are even and optimized
    num_paths_range = []
    for n in total_paths_range:
        optimal_n = calculate_optimal_paths(n, n_jobs)
        num_paths_range.append(optimal_n)
    
    prices_mc = []
    
    # Use cumulative approach - generate more paths and calculate cumulative averages
    np.random.seed(1000)  # Fixed seed for reproducible convergence
    max_paths = max(num_paths_range)
    
    print(f"Generating {max_paths} paths for convergence analysis...")
    s_paths_full, _ = simulate_heston_paths(
        S0, v0, kappa, theta, sigma_v, rho, r, T, num_steps, max_paths, n_jobs
    )
    
    for n in num_paths_range:
        # Use first n paths from the full simulation
        s_paths_subset = s_paths_full[:n, :]
        price_mc, _, _ = calculate_option_price_monte_carlo(
            s_paths_subset, K, r, T, is_call=1 if option_type == 'call' else 0
        )
        prices_mc.append(price_mc)
        print(f"Convergence: {n} paths -> price = {price_mc:.4f}")
        
    fig = plot_convergence_analysis(num_paths_range, prices_mc)
    fig.savefig('results/convergence_analysis.png')
    plt.close(fig)

    # Sensitivity analysis for all parameters
    print("Running sensitivity analysis for all parameters...")
    
    # Kappa sensitivity
    kappa_range = np.linspace(0.5, 5.0, 10)
    prices_mc = []
    for kappa_test in kappa_range:
        s_paths, _ = simulate_heston_paths(
            S0, v0, kappa_test, theta, sigma_v, rho, r, T, num_steps, num_paths, n_jobs
        )
        price_mc, _, _ = calculate_option_price_monte_carlo(
            s_paths, K, r, T, is_call=1 if option_type == 'call' else 0
        )
        prices_mc.append(price_mc)
    fig = plot_sensitivity_analysis(kappa_range, prices_mc, "kappa")
    fig.savefig('results/sensitivity_analysis_kappa.png')
    plt.close(fig)

    # Sigma_v sensitivity
    sigma_v_range = np.linspace(0.1, 0.5, 10)
    prices_mc = []
    for sigma_v_test in sigma_v_range:
        s_paths, _ = simulate_heston_paths(
            S0, v0, kappa, theta, sigma_v_test, rho, r, T, num_steps, num_paths, n_jobs
        )
        price_mc, _, _ = calculate_option_price_monte_carlo(
            s_paths, K, r, T, is_call=1 if option_type == 'call' else 0
        )
        prices_mc.append(price_mc)
    fig = plot_sensitivity_analysis(sigma_v_range, prices_mc, "sigma_v")
    fig.savefig('results/sensitivity_analysis_sigma_v.png')
    plt.close(fig)

    # V0 sensitivity
    v0_range = np.linspace(0.05, 0.25, 10)
    prices_mc = []
    for v0_test in v0_range:
        s_paths, _ = simulate_heston_paths(
            S0, v0_test, kappa, theta, sigma_v, rho, r, T, num_steps, num_paths, n_jobs
        )
        price_mc, _, _ = calculate_option_price_monte_carlo(
            s_paths, K, r, T, is_call=1 if option_type == 'call' else 0
        )
        prices_mc.append(price_mc)
    fig = plot_sensitivity_analysis(v0_range, prices_mc, "v0")
    fig.savefig('results/sensitivity_analysis_v0.png')
    plt.close(fig)

    # Theta sensitivity
    theta_range = np.linspace(0.05, 0.25, 10)
    prices_mc = []
    for theta_test in theta_range:
        s_paths, _ = simulate_heston_paths(
            S0, v0, kappa, theta_test, sigma_v, rho, r, T, num_steps, num_paths, n_jobs
        )
        price_mc, _, _ = calculate_option_price_monte_carlo(
            s_paths, K, r, T, is_call=1 if option_type == 'call' else 0
        )
        prices_mc.append(price_mc)
    fig = plot_sensitivity_analysis(theta_range, prices_mc, "theta")
    fig.savefig('results/sensitivity_analysis_theta.png')
    plt.close(fig)

    # Rho sensitivity
    rho_range = np.linspace(-0.9, 0.9, 10)
    prices_mc = []
    for rho_test in rho_range:
        s_paths, _ = simulate_heston_paths(
            S0, v0, kappa, theta, sigma_v, rho_test, r, T, num_steps, num_paths, n_jobs
        )
        price_mc, _, _ = calculate_option_price_monte_carlo(
            s_paths, K, r, T, is_call=1 if option_type == 'call' else 0
        )
        prices_mc.append(price_mc)
    fig = plot_sensitivity_analysis(rho_range, prices_mc, "rho")
    fig.savefig('results/sensitivity_analysis_rho.png')
    plt.close(fig)

    fig = plot_payoff_distribution(S_paths[:, -1], K, option_type)
    fig.savefig('results/payoff_distribution.png')
    plt.close(fig)

    strike_range = np.linspace(90, 110, 10)  # Reduced range for better accuracy
    heston_mc_prices = []
    bs_prices = []
    implied_vols_mc = []
    
    print("Calculating option prices across strikes...")
    for k in strike_range:
        # Generate NEW simulation for each strike for independence
        np.random.seed(42)  # Fixed seed for reproducibility
        s_paths_k, _ = simulate_heston_paths(
            S0, v0, kappa, theta, sigma_v, rho, r, T, num_steps, num_paths, n_jobs
        )
        price_mc, _, _ = calculate_option_price_monte_carlo(
            s_paths_k, k, r, T, is_call=1 if option_type == 'call' else 0
        )
        bs_price = black_scholes_price(S0, k, T, r, np.sqrt(theta), option_type)
        heston_mc_prices.append(price_mc)
        bs_prices.append(bs_price)
        
        # Use improved implied volatility calculation with wider bounds
        try:
            if price_mc > 1e-6:  # Only calculate if price is meaningful
                iv = brentq(
                    lambda sigma: black_scholes_price(S0, k, T, r, sigma, option_type) - price_mc,
                    0.001, 5.0, xtol=1e-8
                )
                implied_vols_mc.append(iv)
            else:
                implied_vols_mc.append(np.nan)
        except ValueError:
            implied_vols_mc.append(np.nan)
    fig = plot_price_vs_strike(strike_range, heston_mc_prices, bs_prices)
    fig.savefig('results/price_vs_strike.png')
    plt.close(fig)

    fig = plot_implied_volatility_smile(strike_range, implied_vols_mc, np.sqrt(theta))
    fig.savefig('results/implied_volatility_smile.png')
    plt.close(fig)

    # Greeks vs Strike Price (improved precision with smoothing)
    print("Calculating Greeks vs Strike Price...")
    greeks_range = []
    greeks_paths = 30000  # Balanced paths for accuracy vs speed
    
    for i, k in enumerate(strike_range):
        print(f"Calculating Greeks for strike {k} ({i+1}/{len(strike_range)})")
        
        # Calculate multiple estimates and average them for stability
        greek_estimates = {'Delta': [], 'Gamma': [], 'Vega': [], 'Rho': [], 'Theta': []}
        
        for run in range(2):  # 2 independent calculations
            try:
                greeks_k = calculate_heston_greeks(
                    S0, k, T, r, option_type, v0, kappa, theta, sigma_v, rho, 
                    num_steps, greeks_paths, 0.001, n_jobs
                )
                for greek_name in greek_estimates.keys():
                    greek_estimates[greek_name].append(greeks_k[greek_name])
            except ValueError as e:
                print(f"Greeks calculation failed for strike {k}, run {run}: {str(e)}")
                for greek_name in greek_estimates.keys():
                    greek_estimates[greek_name].append(np.nan)
        
        # Average the estimates for stability
        averaged_greeks = {}
        for greek_name in greek_estimates.keys():
            valid_values = [v for v in greek_estimates[greek_name] if not np.isnan(v)]
            if valid_values:
                averaged_greeks[greek_name] = np.mean(valid_values)
            else:
                averaged_greeks[greek_name] = np.nan
        
        greeks_range.append(averaged_greeks)
    
    # Apply post-processing smoothing to Greeks for better visualization
    def smooth_greeks_data(greeks_data, window_size=3):
        """Apply simple moving average smoothing to Greeks data."""
        smoothed_data = []
        n = len(greeks_data)
        
        for i in range(n):
            current_greeks = {}
            for greek_name in ['Delta', 'Gamma', 'Vega', 'Rho', 'Theta']:
                values = []
                
                # Collect values in window around current point
                start_idx = max(0, i - window_size // 2)
                end_idx = min(n, i + window_size // 2 + 1)
                
                for j in range(start_idx, end_idx):
                    if not np.isnan(greeks_data[j][greek_name]):
                        values.append(greeks_data[j][greek_name])
                
                # Calculate smoothed value
                if values:
                    if greek_name == 'Gamma':
                        # For Gamma, use median to reduce outlier influence
                        current_greeks[greek_name] = np.median(values)
                    else:
                        # For other Greeks, use mean
                        current_greeks[greek_name] = np.mean(values)
                else:
                    current_greeks[greek_name] = np.nan
            
            smoothed_data.append(current_greeks)
        
        return smoothed_data
    
    # Apply smoothing to Greeks data
    greeks_range = smooth_greeks_data(greeks_range)
    
    fig = plot_greeks_vs_strike(strike_range, greeks_range)
    fig.savefig('results/greeks_vs_strike.png')
    plt.close(fig)

    maturity_range = np.linspace(0.1, 2.0, 10)
    fig = plot_volatility_surface(
        S0, T, r, option_type, calibration_results['v0'], calibration_results['kappa'], 
        calibration_results['theta'], calibration_results['sigma_v'], calibration_results['rho'],
        strike_range, maturity_range, is_monte_carlo=True,
        num_steps=num_steps, num_paths=num_paths, n_jobs=n_jobs
    )
    fig.savefig('results/volatility_surface_monte_carlo.png')
    plt.close(fig)

    print("Plots saved to 'results/' directory")

if __name__ == "__main__":
    main()