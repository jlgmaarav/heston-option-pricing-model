import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pytest
import numpy as np
from src.heston import simulate_heston_paths, calculate_option_price_monte_carlo, black_scholes_price, calculate_heston_greeks
from src.plots import calculate_implied_volatility

@pytest.mark.parametrize("S0,v0,kappa,theta,sigma_v,rho,r,T,num_steps,num_paths,n_jobs", [
    (100.0, 0.04, 2.0, 0.04, 0.3, -0.7, 0.01, 1.0, 100, 1000, 2),
    (50.0, 0.02, 1.0, 0.02, 0.2, -0.5, 0.02, 0.5, 50, 500, 1)
])
def test_simulate_heston_paths_valid(S0, v0, kappa, theta, sigma_v, rho, r, T, num_steps, num_paths, n_jobs):
    """Test valid Heston path simulation."""
    S_paths, v_paths = simulate_heston_paths(S0, v0, kappa, theta, sigma_v, rho, r, T, num_steps, num_paths, n_jobs)
    assert S_paths.shape == (num_paths, num_steps + 1), "S_paths shape mismatch"
    assert v_paths.shape == (num_paths, num_steps + 1), "v_paths shape mismatch"
    assert np.all(S_paths >= 0), "S_paths should be non-negative"
    assert np.all(v_paths >= 0), "v_paths should be non-negative"

@pytest.mark.parametrize("S0,K,r,T,option_type,v0,kappa,theta,sigma_v,rho,num_steps,num_paths,n_jobs", [
    (100.0, 100.0, 0.01, 1.0, 'call', 0.04, 2.0, 0.04, 0.3, -0.7, 100, 1000, 2),
    (50.0, 50.0, 0.02, 0.5, 'put', 0.02, 1.0, 0.02, 0.2, -0.5, 50, 500, 1)
])
def test_calculate_option_price_monte_carlo_valid(S0, K, r, T, option_type, v0, kappa, theta, sigma_v, rho, num_steps, num_paths, n_jobs):
    """Test valid Monte Carlo option price calculation."""
    S_paths, _ = simulate_heston_paths(S0, v0, kappa, theta, sigma_v, rho, r, T, num_steps, num_paths, n_jobs)
    price, std_err, conf_interval = calculate_option_price_monte_carlo(S_paths, K, r, T, is_call=1 if option_type == 'call' else 0)
    assert isinstance(price, float), "Price should be a float"
    assert price >= 0, "Price should be non-negative"
    assert isinstance(std_err, float), "Standard error should be a float"
    assert len(conf_interval) == 2, "Confidence interval should have two elements"
    assert conf_interval[0] <= price <= conf_interval[1], "Price should lie within confidence interval"

def test_invalid_inputs():
    """Test invalid inputs for Heston simulation."""
    with pytest.raises(ValueError):
        simulate_heston_paths(-100, 0.04, 2.0, 0.04, 0.3, -0.7, 0.01, 1.0, 100, 1000, 2)
    with pytest.raises(ValueError):
        simulate_heston_paths(100, -0.04, 2.0, 0.04, 0.3, -0.7, 0.01, 1.0, 100, 1000, 2)
    with pytest.raises(ValueError):
        simulate_heston_paths(100, 0.04, 2.0, 0.04, 0.3, -0.7, 0.01, 1.0, 100, 999, 2)

def test_black_scholes_consistency():
    """Test Black-Scholes price accent parity and intrinsic value."""
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.01
    sigma = 0.2
    call_price = black_scholes_price(S0, K, T, r, sigma, 'call')
    put_price = black_scholes_price(S0, K, T, r, sigma, 'put')
    assert abs(call_price - put_price - (S0 - K * np.exp(-r * T))) < 1e-6, "Put-call parity not satisfied"
    assert call_price > S0 - K * np.exp(-r * T), "Call price below intrinsic value"
    assert put_price > K * np.exp(-r * T) - S0, "Put price below intrinsic value"

@pytest.mark.parametrize("S0,K,T,r,option_type,v0,kappa,theta,sigma_v,rho", [
    (100.0, 100.0, 1.0, 0.01, 'call', 0.04, 2.0, 0.04, 0.3, -0.7),
    (50.0, 50.0, 0.5, 0.02, 'put', 0.02, 1.0, 0.02, 0.2, -0.5)
])
def test_heston_greeks(S0, K, T, r, option_type, v0, kappa, theta, sigma_v, rho):
    """Test Heston Greeks calculation."""
    greeks = calculate_heston_greeks(S0, K, T, r, option_type, v0, kappa, theta, sigma_v, rho, 100, 1000, 0.01, 2)
    assert all(key in greeks for key in ['Delta', 'Gamma', 'Vega', 'Rho', 'Theta'])
    assert abs(greeks['Delta']) <= 1.0, "Delta should be between -1 and 1"
    assert greeks['Gamma'] >= 0, "Gamma should be non-negative"

def test_implied_volatility():
    """Test implied volatility calculation."""
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.01
    sigma = 0.2
    call_price = black_scholes_price(S0, K, T, r, sigma, 'call')
    implied_vol = calculate_implied_volatility(call_price, S0, K, T, r, 'call')
    assert abs(implied_vol - sigma) < 1e-2, "Implied volatility should match input"

# @pytest.mark.benchmark
def test_simulation_performance():
    """Test simulation performance without benchmark dependency."""
    S0 = 100.0
    v0 = 0.04
    kappa = 2.0
    theta = 0.04
    sigma_v = 0.3
    rho = -0.7
    r = 0.01
    T = 1.0
    num_steps = 100
    num_paths = 1000
    n_jobs = 2
    
    # Simple performance test without benchmark
    import time
    start_time = time.time()
    S_paths, v_paths = simulate_heston_paths(S0, v0, kappa, theta, sigma_v, rho, r, T, num_steps, num_paths, n_jobs)
    end_time = time.time()
    
    execution_time = end_time - start_time
    assert execution_time < 60.0, "Simulation should complete within 60 seconds"
    assert S_paths.shape == (num_paths, num_steps + 1), "S_paths shape mismatch"
    assert v_paths.shape == (num_paths, num_steps + 1), "v_paths shape mismatch"