import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import time
import psutil
from functools import lru_cache
from scipy import stats
from src.heston import simulate_heston_paths, calculate_option_price_monte_carlo, black_scholes_price, calculate_heston_greeks
from src.plots import (
    plot_heston_paths, plot_convergence_analysis, plot_sensitivity_analysis,
    plot_payoff_distribution, plot_price_vs_strike, plot_implied_volatility_smile,
    plot_volatility_surface
)

st.set_page_config(page_title="Heston Model Option Pricing - Advanced", layout="wide")

@lru_cache(maxsize=128)
def cached_simulate_heston_paths(S0, v0, kappa, theta, sigma_v, rho, r, T, num_steps, num_paths, n_jobs):
    """Cached Heston path simulation for performance."""
    return simulate_heston_paths(S0, v0, kappa, theta, sigma_v, rho, r, T, num_steps, num_paths, n_jobs)

@lru_cache(maxsize=128)
def cached_calculate_implied_volatility(market_price, S, K, T, r, option_type):
    """Calculate implied volatility using Brent's method with caching."""
    from scipy.optimize import brentq
    if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return np.nan
    if option_type not in ('call', 'put'):
        return np.nan
    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - market_price
    try:
        return brentq(objective, 0.01, 2.0, xtol=1e-6)
    except ValueError:
        return np.nan

def calculate_risk_metrics(S_paths, K, r, T, option_type='call'):
    """Calculate advanced risk metrics."""
    ST = S_paths[:, -1]
    
    # Calculate returns from price paths  
    returns = np.diff(np.log(S_paths), axis=1)
    final_returns = returns[:, -1]
    
    # Calculate option payoffs
    if option_type == 'call':
        payoffs = np.maximum(0, ST - K)
    else:
        payoffs = np.maximum(0, K - ST)
    
    discounted_payoffs = payoffs * np.exp(-r * T)
    
    # Calculate P&L from option perspective (what we make/lose)
    option_price = np.mean(discounted_payoffs)
    pnl = discounted_payoffs - option_price  # P&L relative to expected price
    
    # VaR calculations (losses are negative)
    var_95 = np.percentile(pnl, 5)  # 5th percentile (potential loss)
    var_99 = np.percentile(pnl, 1)  # 1st percentile (extreme loss)
    
    # Expected Shortfall (CVaR) - expected loss beyond VaR
    cvar_95 = np.mean(pnl[pnl <= var_95]) if np.any(pnl <= var_95) else var_95
    cvar_99 = np.mean(pnl[pnl <= var_99]) if np.any(pnl <= var_99) else var_99
    
    # Maximum Drawdown from cumulative returns
    cumulative_returns = np.cumprod(1 + returns, axis=1)
    running_max = np.maximum.accumulate(cumulative_returns, axis=1)
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Return statistics
    skewness = stats.skew(final_returns)
    kurtosis = stats.kurtosis(final_returns)
    
    # Sharpe ratio (annualized)
    mean_return = np.mean(final_returns) * 252  # Annualized
    vol_return = np.std(final_returns) * np.sqrt(252)  # Annualized vol
    sharpe_ratio = (mean_return - r) / vol_return if vol_return > 0 else 0
    
    return {
        'VaR_95': var_95,
        'VaR_99': var_99,
        'CVaR_95': cvar_95,
        'CVaR_99': cvar_99,
        'Max_Drawdown': max_drawdown,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Sharpe_Ratio': sharpe_ratio,
        'Volatility_Annualized': vol_return
    }

def monte_carlo_diagnostics(S_paths):
    """Perform Monte Carlo diagnostic tests."""
    # Extract returns
    returns = np.diff(np.log(S_paths), axis=1)
    final_returns = returns[:, -1]
    
    # Autocorrelation test
    def autocorr(x, lags=10):
        n = len(x)
        x = x - np.mean(x)
        autocorrelations = np.correlate(x, x, mode='full')
        autocorrelations = autocorrelations[n-1:n+lags]
        autocorrelations = autocorrelations / autocorrelations[0]
        return autocorrelations[1:]
    
    autocorr_values = autocorr(final_returns)
    
    # Normality test (Jarque-Bera)
    jb_stat, jb_pvalue = stats.jarque_bera(final_returns)
    
    # Kolmogorov-Smirnov test against normal
    ks_stat, ks_pvalue = stats.kstest(final_returns, 'norm', args=(np.mean(final_returns), np.std(final_returns)))
    
    # Variance ratio test
    def variance_ratio_test(returns, k=2):
        n = len(returns)
        mean_return = np.mean(returns)
        
        # k-period variance
        k_returns = np.sum(returns[:n//k*k].reshape(-1, k), axis=1)
        var_k = np.var(k_returns, ddof=1)
        
        # 1-period variance
        var_1 = np.var(returns, ddof=1)
        
        vr = var_k / (k * var_1)
        return vr
    
    vr_2 = variance_ratio_test(final_returns, 2)
    vr_4 = variance_ratio_test(final_returns, 4)
    
    # Effective sample size
    rho_1 = np.corrcoef(final_returns[:-1], final_returns[1:])[0,1] if len(final_returns) > 1 else 0
    # Handle NaN correlation coefficient
    if np.isnan(rho_1):
        rho_1 = 0.0
    # Cap correlation to prevent negative effective sample sizes and ensure realistic efficiency
    rho_1 = np.clip(rho_1, -0.99, 0.99)
    eff_sample_size = len(final_returns) * (1 - rho_1) / (1 + rho_1) if (1 + rho_1) != 0 else len(final_returns)
    # Ensure effective sample size doesn't exceed actual sample size
    eff_sample_size = min(eff_sample_size, len(final_returns))
    
    return {
        'autocorr_lag1': autocorr_values[0] if len(autocorr_values) > 0 else 0,
        'autocorr_lag2': autocorr_values[1] if len(autocorr_values) > 1 else 0,
        'autocorr_lag5': autocorr_values[4] if len(autocorr_values) > 4 else 0,
        'jarque_bera_stat': jb_stat,
        'jarque_bera_pvalue': jb_pvalue,
        'ks_stat': ks_stat,
        'ks_pvalue': ks_pvalue,
        'variance_ratio_2': vr_2,
        'variance_ratio_4': vr_4,
        'effective_sample_size': eff_sample_size,
        'final_returns': final_returns
    }

def plot_3d_volatility_surface_plotly(S0, r, option_type, v0, kappa, theta, sigma_v, rho, num_steps, num_paths, n_jobs):
    """Create interactive 3D volatility surface using Plotly."""
    strike_range = np.linspace(80, 120, 8)
    maturity_range = np.linspace(0.1, 2.0, 8)
    
    implied_vols = np.zeros((len(maturity_range), len(strike_range)))
    
    for i, T in enumerate(maturity_range):
        S_paths, _ = simulate_heston_paths(S0, v0, kappa, theta, sigma_v, rho, r, T, num_steps, min(num_paths, 5000), n_jobs)
        for j, K in enumerate(strike_range):
            price_mc, _, _ = calculate_option_price_monte_carlo(
                S_paths, K, r, T, is_call=1 if option_type == 'call' else 0
            )
            iv = cached_calculate_implied_volatility(price_mc, S0, K, T, r, option_type)
            implied_vols[i, j] = iv if not np.isnan(iv) else np.sqrt(theta)
    
    fig = go.Figure(data=[go.Surface(
        x=strike_range,
        y=maturity_range,
        z=implied_vols,
        colorscale='Viridis',
        showscale=True
    )])
    
    fig.update_layout(
        title='Interactive 3D Implied Volatility Surface',
        scene=dict(
            xaxis_title='Strike Price',
            yaxis_title='Time to Maturity',
            zaxis_title='Implied Volatility',
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.8))
        ),
        width=800,
        height=600
    )
    
    return fig

def calculate_greeks_time_series(S0, K, r, option_type, v0, kappa, theta, sigma_v, rho, num_steps, num_paths, n_jobs):
    """Calculate Greeks across different times to expiry."""
    time_range = np.linspace(0.1, 2.0, 10)
    strike_range = [90, 100, 110]  # Different moneyness levels
    
    greeks_data = {'Time': [], 'Strike': [], 'Delta': [], 'Gamma': [], 'Vega': [], 'Theta': [], 'Rho': []}
    
    for T in time_range:
        for K_test in strike_range:
            try:
                greeks = calculate_heston_greeks(
                    S0, K_test, T, r, option_type, v0, kappa, theta, sigma_v, rho, 
                    num_steps, min(num_paths, 2000), 0.01, n_jobs
                )
                
                greeks_data['Time'].append(T)
                greeks_data['Strike'].append(K_test)
                greeks_data['Delta'].append(greeks['Delta'])
                greeks_data['Gamma'].append(greeks['Gamma'])
                greeks_data['Vega'].append(greeks['Vega'])
                greeks_data['Theta'].append(greeks['Theta'])
                greeks_data['Rho'].append(greeks['Rho'])
            except:
                continue
    
    return pd.DataFrame(greeks_data)

def initialize_session_state():
    """Initialize Streamlit session state with default parameters."""
    defaults = {
        'S0': 100.0,
        'K': 100.0,
        'T': 1.0,
        'r': 0.01,
        'option_type': 'call',
        'v0': 0.04,
        'kappa': 2.0,
        'theta': 0.04,
        'sigma_v': 0.3,
        'rho': -0.7,
        'num_steps': 100,
        'num_paths': 10000,
        'n_jobs': 4,
        'min_spot': 50.0,
        'max_spot': 150.0,
        'min_vol': 0.1,
        'max_vol': 0.5
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

st.sidebar.header("Heston Model Parameters")
with st.sidebar.expander("Asset and Option Parameters"):
    col_s0, col_k = st.columns(2)
    with col_s0:
        st.session_state.S0 = st.number_input("Initial Asset Price (S0)", min_value=1.0, max_value=1000.0, value=st.session_state.S0, step=1.0)
    with col_k:
        st.session_state.K = st.number_input("Strike Price (K)", min_value=1.0, max_value=1000.0, value=st.session_state.K, step=1.0)
    
    col_t, col_r = st.columns(2)
    with col_t:
        st.session_state.T = st.number_input("Time to Maturity (T, years)", min_value=0.01, max_value=5.0, value=st.session_state.T, step=0.1)
    with col_r:
        st.session_state.r = st.number_input("Risk-Free Rate (r)", min_value=0.0, max_value=0.2, value=st.session_state.r, step=0.01)
    
    st.session_state.option_type = st.selectbox("Option Type", ["call", "put"], index=0 if st.session_state.option_type == 'call' else 1)

with st.sidebar.expander("Heston Model Parameters"):
    col_v0, col_kappa = st.columns(2)
    with col_v0:
        st.session_state.v0 = st.number_input("Initial Variance (v0)", min_value=0.0, max_value=1.0, value=st.session_state.v0, step=0.01)
    with col_kappa:
        st.session_state.kappa = st.number_input("Mean Reversion Rate (kappa)", min_value=0.01, max_value=5.0, value=st.session_state.kappa, step=0.1)
    
    col_theta, col_sigma_v = st.columns(2)
    with col_theta:
        st.session_state.theta = st.number_input("Long-Term Variance (theta)", min_value=0.0, max_value=1.0, value=st.session_state.theta, step=0.01)
    with col_sigma_v:
        st.session_state.sigma_v = st.number_input("Volatility of Variance (sigma_v)", min_value=0.0, max_value=1.0, value=st.session_state.sigma_v, step=0.01)
    
    st.session_state.rho = st.number_input("Correlation (rho)", min_value=-1.0, max_value=1.0, value=st.session_state.rho, step=0.1)

with st.sidebar.expander("Simulation Parameters"):
    col_steps, col_paths, col_jobs = st.columns(3)
    with col_steps:
        st.session_state.num_steps = st.number_input("Time Steps", min_value=10, max_value=1000, value=st.session_state.num_steps, step=10)
    with col_paths:
        num_paths_input = st.number_input("Number of Paths", min_value=1000, max_value=100000, value=st.session_state.num_paths, step=1000)
        if num_paths_input % 2 != 0:
            st.error("Number of paths must be even for antithetic variates.")
            num_paths_input += 1
        st.session_state.num_paths = num_paths_input
    with col_jobs:
        st.session_state.n_jobs = st.number_input("Parallel Jobs", min_value=1, max_value=8, value=st.session_state.n_jobs, step=1)

st.title("🚀 Advanced Heston Model Option Pricing Dashboard")
st.write("Professional-grade interactive tool for pricing European options using the Heston stochastic volatility model with advanced analytics.")

# Generate paths once for reuse
@st.cache_data
def get_simulation_data():
    return simulate_heston_paths(
        st.session_state.S0, st.session_state.v0, st.session_state.kappa,
        st.session_state.theta, st.session_state.sigma_v, st.session_state.rho,
        st.session_state.r, st.session_state.T, st.session_state.num_steps,
        st.session_state.num_paths, st.session_state.n_jobs
    )

S_paths, v_paths = get_simulation_data()

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
    "Option Price & Greeks", "Price Paths", "Convergence Analysis",
    "Sensitivity Analysis", "Risk Metrics", "MC Diagnostics", 
    "Greeks vs Time", "3D Volatility Surface", "Payoff Distribution", 
    "Volatility Smile", "Model Comparison", "Model Comparison Heatmap"
])

with tab1:
    st.subheader("Option Price and Greeks")
    progress_bar = st.progress(0)
    try:
        price_mc, std_err, conf_interval = calculate_option_price_monte_carlo(
            S_paths, st.session_state.K, st.session_state.r, st.session_state.T,
            is_call=1 if st.session_state.option_type == 'call' else 0
        )
        bs_price = black_scholes_price(
            st.session_state.S0, st.session_state.K, st.session_state.T,
            st.session_state.r, np.sqrt(st.session_state.theta), st.session_state.option_type
        )
        implied_vol_mc = cached_calculate_implied_volatility(
            price_mc, st.session_state.S0, st.session_state.K, st.session_state.T,
            st.session_state.r, st.session_state.option_type
        )
        greeks = calculate_heston_greeks(
            st.session_state.S0, st.session_state.K, st.session_state.T,
            st.session_state.r, st.session_state.option_type,
            st.session_state.v0, st.session_state.kappa, st.session_state.theta,
            st.session_state.sigma_v, st.session_state.rho, st.session_state.num_steps,
            st.session_state.num_paths, 0.01, st.session_state.n_jobs
        )
        progress_bar.progress(100)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Heston MC Price", f"{price_mc:.4f}")
            st.metric("Standard Error", f"{std_err:.4f}")
            st.metric("95% CI", f"[{conf_interval[0]:.4f}, {conf_interval[1]:.4f}]")
        with col2:
            st.metric("Black-Scholes Price", f"{bs_price:.4f}")
            st.metric("Implied Volatility (MC)", f"{implied_vol_mc:.4f}")

        st.subheader("Greeks")
        df_greeks = pd.DataFrame(greeks.items(), columns=['Greek', 'Value'])
        st.table(df_greeks)
    except Exception as e:
        st.error(f"Error: {str(e)}")
    progress_bar.empty()

with tab2:
    st.subheader("Simulated Price and Variance Paths")
    fig = plot_heston_paths(S_paths, v_paths, num_paths_to_plot=5)
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    st.subheader("Convergence Analysis")
    num_paths_range = np.logspace(2, 5, 10, dtype=int) // 2 * 2
    prices_mc = []
    for n in num_paths_range:
        s_paths, _ = cached_simulate_heston_paths(
            st.session_state.S0, st.session_state.v0, st.session_state.kappa,
            st.session_state.theta, st.session_state.sigma_v, st.session_state.rho,
            st.session_state.r, st.session_state.T, st.session_state.num_steps,
            n, st.session_state.n_jobs
        )
        price_mc, _, _ = calculate_option_price_monte_carlo(
            s_paths, st.session_state.K, st.session_state.r, st.session_state.T,
            is_call=1 if st.session_state.option_type == 'call' else 0
        )
        prices_mc.append(price_mc)
    fig = plot_convergence_analysis(num_paths_range, prices_mc)
    st.pyplot(fig)
    plt.close(fig)

with tab4:
    st.subheader("Sensitivity Analysis")
    param_name = st.selectbox("Parameter for Sensitivity", ["kappa", "rho", "sigma_v", "theta", "v0"], index=0)
    
    param_ranges = {
        "kappa": np.linspace(0.5, 5.0, 10),
        "rho": np.linspace(-0.9, 0.9, 10),
        "sigma_v": np.linspace(0.1, 0.5, 10),
        "theta": np.linspace(0.01, 0.2, 10),
        "v0": np.linspace(0.01, 0.2, 10)
    }
    
    param_range = param_ranges[param_name]
    prices_mc = []
    
    progress_bar_sens = st.progress(0)
    
    for i, param in enumerate(param_range):
        kwargs = {
            'S0': st.session_state.S0, 'v0': st.session_state.v0, 'kappa': st.session_state.kappa,
            'theta': st.session_state.theta, 'sigma_v': st.session_state.sigma_v, 'rho': st.session_state.rho,
            'r': st.session_state.r, 'T': st.session_state.T, 'num_steps': st.session_state.num_steps,
            'num_paths': st.session_state.num_paths, 'n_jobs': st.session_state.n_jobs
        }
        kwargs[param_name] = param
        
        if param_name == 'theta' and kwargs['theta'] <= 0:
            kwargs['theta'] = 0.001
        if param_name == 'v0' and kwargs['v0'] <= 0:
            kwargs['v0'] = 0.001
        if param_name == 'kappa' and kwargs['kappa'] <= 0:
            kwargs['kappa'] = 0.1
        if param_name == 'sigma_v' and kwargs['sigma_v'] <= 0:
            kwargs['sigma_v'] = 0.01
            
        try:
            s_paths, _ = cached_simulate_heston_paths(**kwargs)
            price_mc, _, _ = calculate_option_price_monte_carlo(
                s_paths, st.session_state.K, st.session_state.r, st.session_state.T,
                is_call=1 if st.session_state.option_type == 'call' else 0
            )
            prices_mc.append(price_mc)
        except Exception as e:
            st.warning(f"Error calculating for {param_name}={param}: {str(e)}")
            prices_mc.append(np.nan)
        
        progress_bar_sens.progress((i + 1) / len(param_range))
    
    progress_bar_sens.empty()
    
    valid_indices = ~np.isnan(prices_mc)
    param_range_valid = param_range[valid_indices]
    prices_mc_valid = np.array(prices_mc)[valid_indices]
    
    if len(prices_mc_valid) > 0:
        fig = plot_sensitivity_analysis(param_range_valid, prices_mc_valid, param_name)
        st.pyplot(fig)
        plt.close(fig)
        
        st.subheader(f"Sensitivity Statistics for {param_name}")
        sensitivity_stats = {
            'Min Value': f"{np.min(prices_mc_valid):.4f}",
            'Max Value': f"{np.max(prices_mc_valid):.4f}",
            'Range': f"{np.max(prices_mc_valid) - np.min(prices_mc_valid):.4f}",
            'Standard Deviation': f"{np.std(prices_mc_valid):.4f}",
            'Relative Volatility': f"{np.std(prices_mc_valid) / np.mean(prices_mc_valid) * 100:.2f}%"
        }
        st.write(pd.DataFrame(sensitivity_stats.items(), columns=['Statistic', 'Value']))
    else:
        st.error(f"No valid calculations for parameter {param_name}")

with tab5:
    st.subheader("🎯 Advanced Risk Metrics Dashboard")
    
    if st.button("Calculate Risk Metrics"):
        with st.spinner("Calculating advanced risk metrics..."):
            risk_metrics = calculate_risk_metrics(S_paths, st.session_state.K, st.session_state.r, st.session_state.T, st.session_state.option_type)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Value at Risk (95%)", f"{risk_metrics['VaR_95']:.4f}")
            st.metric("Value at Risk (99%)", f"{risk_metrics['VaR_99']:.4f}")
            st.metric("Max Drawdown", f"{risk_metrics['Max_Drawdown']:.4f}")
        
        with col2:
            st.metric("Expected Shortfall (95%)", f"{risk_metrics['CVaR_95']:.4f}")
            st.metric("Expected Shortfall (99%)", f"{risk_metrics['CVaR_99']:.4f}")
            st.metric("Sharpe Ratio", f"{risk_metrics['Sharpe_Ratio']:.4f}")
        
        with col3:
            st.metric("Skewness", f"{risk_metrics['Skewness']:.4f}")
            st.metric("Kurtosis", f"{risk_metrics['Kurtosis']:.4f}")
            st.metric("Annualized Volatility", f"{risk_metrics['Volatility_Annualized']:.4f}")
        
        # Risk metrics interpretation
        st.subheader("Risk Interpretation")
        if risk_metrics['VaR_95'] < 0:
            st.warning(f"🔴 High Risk: 5% chance of losses exceeding {abs(risk_metrics['VaR_95']):.2f}")
        else:
            st.success("🟢 Low Risk: Positive VaR indicates consistent profits")
        
        if abs(risk_metrics['Skewness']) > 1:
            st.info(f"📊 High Skewness ({risk_metrics['Skewness']:.2f}): Returns are heavily skewed")
        
        if risk_metrics['Kurtosis'] > 3:
            st.info(f"📈 Fat Tails: Kurtosis ({risk_metrics['Kurtosis']:.2f}) indicates higher tail risk")

with tab6:
    st.subheader("🔬 Monte Carlo Diagnostics")
    
    if st.button("Run MC Diagnostics"):
        with st.spinner("Running Monte Carlo diagnostic tests..."):
            diagnostics = monte_carlo_diagnostics(S_paths)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Statistical Tests")
            st.metric("Autocorr Lag-1", f"{diagnostics['autocorr_lag1']:.4f}")
            st.metric("Autocorr Lag-2", f"{diagnostics['autocorr_lag2']:.4f}")
            st.metric("Autocorr Lag-5", f"{diagnostics['autocorr_lag5']:.4f}")
            st.metric("Effective Sample Size", f"{diagnostics['effective_sample_size']:.0f}")
        
        with col2:
            st.subheader("Normality Tests")
            st.metric("Jarque-Bera Statistic", f"{diagnostics['jarque_bera_stat']:.4f}")
            st.metric("JB P-value", f"{diagnostics['jarque_bera_pvalue']:.4f}")
            st.metric("KS Statistic", f"{diagnostics['ks_stat']:.4f}")
            st.metric("KS P-value", f"{diagnostics['ks_pvalue']:.4f}")
        
        st.subheader("Variance Ratio Tests")
        col3, col4 = st.columns(2)
        with col3:
            st.metric("VR Test (k=2)", f"{diagnostics['variance_ratio_2']:.4f}")
        with col4:
            st.metric("VR Test (k=4)", f"{diagnostics['variance_ratio_4']:.4f}")
        
        # QQ Plot
        st.subheader("Q-Q Plot vs Normal Distribution")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Q-Q Plot
        stats.probplot(diagnostics['final_returns'], dist="norm", plot=ax1)
        ax1.set_title("Q-Q Plot vs Normal Distribution")
        ax1.grid(True)
        
        # Autocorrelation Plot
        from statsmodels.tsa.stattools import acf
        autocorr_full = acf(diagnostics['final_returns'], nlags=20, fft=True)
        ax2.bar(range(len(autocorr_full)), autocorr_full)
        ax2.set_title("Autocorrelation Function")
        ax2.set_xlabel("Lag")
        ax2.set_ylabel("Autocorrelation")
        ax2.grid(True)
        
        st.pyplot(fig)
        plt.close(fig)
        
        # Diagnostic interpretation
        st.subheader("Diagnostic Interpretation")
        if abs(diagnostics['autocorr_lag1']) > 0.1:
            st.warning(f"High autocorrelation detected: {diagnostics['autocorr_lag1']:.3f} - paths may not be independent")
        else:
            st.success("Low autocorrelation: Good path independence")
        
        if diagnostics['jarque_bera_pvalue'] < 0.05:
            st.info("Returns deviate significantly from normality (JB test)")
        else:
            st.success("Returns are approximately normal (JB test)")
        
        efficiency = (diagnostics['effective_sample_size'] / len(S_paths)) * 100
        st.info(f"Monte Carlo efficiency: {efficiency:.1f}% of theoretical maximum")

with tab7:
    st.subheader("📈 Greeks Evolution Over Time")
    
    if st.button("Calculate Greeks vs Time-to-Expiry"):
        with st.spinner("Calculating Greeks across different expiration times..."):
            greeks_df = calculate_greeks_time_series(
                st.session_state.S0, st.session_state.K, st.session_state.r, 
                st.session_state.option_type, st.session_state.v0, st.session_state.kappa,
                st.session_state.theta, st.session_state.sigma_v, st.session_state.rho,
                st.session_state.num_steps, st.session_state.num_paths, st.session_state.n_jobs
            )
        
        if not greeks_df.empty:
            # Create interactive plotly charts for Greeks
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=['Delta vs Time', 'Gamma vs Time', 'Vega vs Time', 
                              'Theta vs Time', 'Rho vs Time', 'Greeks Summary'],
                specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
            )
            
            strikes = greeks_df['Strike'].unique()
            colors = ['blue', 'red', 'green']
            
            for i, strike in enumerate(strikes):
                df_strike = greeks_df[greeks_df['Strike'] == strike]
                color = colors[i % len(colors)]
                
                # Delta
                fig.add_trace(go.Scatter(x=df_strike['Time'], y=df_strike['Delta'], 
                                       mode='lines+markers', name=f'Delta K={strike}',
                                       line=dict(color=color)), row=1, col=1)
                
                # Gamma  
                fig.add_trace(go.Scatter(x=df_strike['Time'], y=df_strike['Gamma'],
                                       mode='lines+markers', name=f'Gamma K={strike}',
                                       line=dict(color=color, dash='dash'), showlegend=False), row=1, col=2)
                
                # Vega
                fig.add_trace(go.Scatter(x=df_strike['Time'], y=df_strike['Vega'],
                                       mode='lines+markers', name=f'Vega K={strike}',
                                       line=dict(color=color, dash='dot'), showlegend=False), row=1, col=3)
                
                # Theta
                fig.add_trace(go.Scatter(x=df_strike['Time'], y=df_strike['Theta'],
                                       mode='lines+markers', name=f'Theta K={strike}',
                                       line=dict(color=color, dash='dashdot'), showlegend=False), row=2, col=1)
                
                # Rho
                fig.add_trace(go.Scatter(x=df_strike['Time'], y=df_strike['Rho'],
                                       mode='lines+markers', name=f'Rho K={strike}',
                                       line=dict(color=color, dash='longdash'), showlegend=False), row=2, col=2)
            
            fig.update_layout(height=800, title_text="Greeks Evolution Across Time-to-Expiry")
            fig.update_xaxes(title_text="Time to Expiry (Years)")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary statistics
            st.subheader("Greeks Statistics Summary")
            summary_stats = greeks_df.groupby('Strike').agg({
                'Delta': ['mean', 'std', 'min', 'max'],
                'Gamma': ['mean', 'std', 'min', 'max'],
                'Vega': ['mean', 'std', 'min', 'max'],
                'Theta': ['mean', 'std', 'min', 'max'],
                'Rho': ['mean', 'std', 'min', 'max']
            }).round(4)
            
            st.write(summary_stats)
        else:
            st.error("Failed to calculate Greeks time series")

with tab8:
    st.subheader("🌊 Interactive 3D Volatility Surface")
    
    if st.button("Generate 3D Volatility Surface"):
        with st.spinner("Creating interactive 3D volatility surface..."):
            fig_3d = plot_3d_volatility_surface_plotly(
                st.session_state.S0, st.session_state.r, st.session_state.option_type,
                st.session_state.v0, st.session_state.kappa, st.session_state.theta,
                st.session_state.sigma_v, st.session_state.rho, st.session_state.num_steps,
                st.session_state.num_paths, st.session_state.n_jobs
            )
            st.plotly_chart(fig_3d, use_container_width=True)
        
        # Performance metrics
        st.subheader("Performance Analytics")
        
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Memory Usage", f"{memory_info.rss / 1024 / 1024:.1f} MB")
        with col2:
            st.metric("CPU Usage", f"{cpu_percent:.1f}%")
        with col3:
            st.metric("Num Paths", f"{st.session_state.num_paths:,}")
        with col4:
            st.metric("Parallel Jobs", f"{st.session_state.n_jobs}")
        
        # Computational efficiency analysis
        paths_per_second = st.session_state.num_paths / max(1, cpu_percent / 100)
        st.info(f"Computational Efficiency: ~{paths_per_second:,.0f} paths/second equivalent")

with tab9:
    st.subheader("Payoff Distribution")
    fig = plot_payoff_distribution(S_paths[:, -1], st.session_state.K, st.session_state.option_type)
    st.pyplot(fig)
    plt.close(fig)

with tab10:
    st.subheader("Implied Volatility Smile")
    strike_range = np.linspace(st.session_state.min_spot, st.session_state.max_spot, 10)
    heston_mc_prices = []
    bs_prices = []
    implied_vols_mc = []
    
    progress_bar_smile = st.progress(0)
    
    for i, k in enumerate(strike_range):
        price_mc, _, _ = calculate_option_price_monte_carlo(
            S_paths, k, st.session_state.r, st.session_state.T,
            is_call=1 if st.session_state.option_type == 'call' else 0
        )
        bs_price = black_scholes_price(
            st.session_state.S0, k, st.session_state.T, st.session_state.r,
            np.sqrt(st.session_state.theta), st.session_state.option_type
        )
        heston_mc_prices.append(price_mc)
        bs_prices.append(bs_price)
        implied_vols_mc.append(cached_calculate_implied_volatility(
            price_mc, st.session_state.S0, k, st.session_state.T, st.session_state.r, st.session_state.option_type
        ))
        progress_bar_smile.progress((i + 1) / len(strike_range))
    
    progress_bar_smile.empty()
    
    fig = plot_price_vs_strike(strike_range, heston_mc_prices, bs_prices)
    st.pyplot(fig)
    plt.close(fig)
    fig = plot_implied_volatility_smile(strike_range, implied_vols_mc, np.sqrt(st.session_state.theta))
    st.pyplot(fig)
    plt.close(fig)

with tab11:
    st.subheader("Model Comparison")
    maturity_range = np.linspace(0.1, 2.0, 10)
    fig = plot_volatility_surface(
        st.session_state.S0, st.session_state.T, st.session_state.r, st.session_state.option_type, 
        st.session_state.v0, st.session_state.kappa, st.session_state.theta, 
        st.session_state.sigma_v, st.session_state.rho, strike_range, maturity_range, 
        is_monte_carlo=True, num_steps=st.session_state.num_steps, 
        num_paths=st.session_state.num_paths, n_jobs=st.session_state.n_jobs
    )
    st.pyplot(fig)
    plt.close(fig)

with tab12:
    st.subheader("Model Comparison Heatmap")
    compute_heatmap = st.button("Compute Heatmap")
    if compute_heatmap:
        progress_bar = st.progress(0)
        try:
            strike_range = np.linspace(st.session_state.min_spot, st.session_state.max_spot, 5)
            maturity_range = np.linspace(0.1, 2.0, 5)
            heston_call_prices = np.zeros((len(maturity_range), len(strike_range)))
            heston_put_prices = np.zeros((len(maturity_range), len(strike_range)))
            bs_call_prices = np.zeros((len(maturity_range), len(strike_range)))
            bs_put_prices = np.zeros((len(maturity_range), len(strike_range)))
            
            for i, T in enumerate(maturity_range):
                S_paths_temp, _ = cached_simulate_heston_paths(
                    st.session_state.S0, st.session_state.v0, st.session_state.kappa,
                    st.session_state.theta, st.session_state.sigma_v, st.session_state.rho,
                    st.session_state.r, T, st.session_state.num_steps, st.session_state.num_paths,
                    st.session_state.n_jobs
                )
                for j, K in enumerate(strike_range):
                    call_price_mc, _, _ = calculate_option_price_monte_carlo(S_paths_temp, K, st.session_state.r, T, is_call=1)
                    put_price_mc, _, _ = calculate_option_price_monte_carlo(S_paths_temp, K, st.session_state.r, T, is_call=0)
                    heston_call_prices[i, j] = call_price_mc
                    heston_put_prices[i, j] = put_price_mc
                    bs_call_prices[i, j] = black_scholes_price(st.session_state.S0, K, T, st.session_state.r, np.sqrt(st.session_state.theta), 'call')
                    bs_put_prices[i, j] = black_scholes_price(st.session_state.S0, K, T, st.session_state.r, np.sqrt(st.session_state.theta), 'put')
                progress_bar.progress((i + 1) / len(maturity_range))
            
            # Calculate differences BEFORE using them
            call_diff = heston_call_prices - bs_call_prices
            put_diff = heston_put_prices - bs_put_prices
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Heston vs Black-Scholes Option Prices', fontsize=14, fontweight='bold')
            X, Y = np.meshgrid(strike_range, maturity_range)
            
            im1 = axes[0, 0].pcolormesh(X, Y, heston_call_prices, cmap='viridis', shading='auto')
            axes[0, 0].set_title('Heston Model - Call Options', fontweight='bold')
            axes[0, 0].set_xlabel('Strike Price (K)')
            axes[0, 0].set_ylabel('Time to Maturity (T)')
            axes[0, 0].set_xticks(strike_range)
            axes[0, 0].set_xticklabels([f'{k:.0f}' for k in strike_range], rotation=45)
            axes[0, 0].set_yticks(maturity_range)
            axes[0, 0].set_yticklabels([f'{t:.2f}' for t in maturity_range])
            plt.colorbar(im1, ax=axes[0, 0], label='Option Price')
            
            for i in range(len(maturity_range)):
                for j in range(len(strike_range)):
                    axes[0, 0].text(strike_range[j], maturity_range[i], f'{heston_call_prices[i, j]:.2f}',
                                    ha="center", va="center", color="white", fontsize=8, fontweight='bold')
            
            im2 = axes[0, 1].pcolormesh(X, Y, heston_put_prices, cmap='plasma', shading='auto')
            axes[0, 1].set_title('Heston Model - Put Options', fontweight='bold')
            axes[0, 1].set_xlabel('Strike Price (K)')
            axes[0, 1].set_ylabel('Time to Maturity (T)')
            axes[0, 1].set_xticks(strike_range)
            axes[0, 1].set_xticklabels([f'{k:.0f}' for k in strike_range], rotation=45)
            axes[0, 1].set_yticks(maturity_range)
            axes[0, 1].set_yticklabels([f'{t:.2f}' for t in maturity_range])
            plt.colorbar(im2, ax=axes[0, 1], label='Option Price')
            
            for i in range(len(maturity_range)):
                for j in range(len(strike_range)):
                    axes[0, 1].text(strike_range[j], maturity_range[i], f'{heston_put_prices[i, j]:.2f}',
                                    ha="center", va="center", color="white", fontsize=8, fontweight='bold')
            
            im3 = axes[1, 0].pcolormesh(X, Y, bs_call_prices, cmap='viridis', shading='auto')
            axes[1, 0].set_title('Black-Scholes Model - Call Options', fontweight='bold')
            axes[1, 0].set_xlabel('Strike Price (K)')
            axes[1, 0].set_ylabel('Time to Maturity (T)')
            axes[1, 0].set_xticks(strike_range)
            axes[1, 0].set_xticklabels([f'{k:.0f}' for k in strike_range], rotation=45)
            axes[1, 0].set_yticks(maturity_range)
            axes[1, 0].set_yticklabels([f'{t:.2f}' for t in maturity_range])
            plt.colorbar(im3, ax=axes[1, 0], label='Option Price')
            
            for i in range(len(maturity_range)):
                for j in range(len(strike_range)):
                    axes[1, 0].text(strike_range[j], maturity_range[i], f'{bs_call_prices[i, j]:.2f}',
                                    ha="center", va="center", color="white", fontsize=8, fontweight='bold')
            
            im4 = axes[1, 1].pcolormesh(X, Y, bs_put_prices, cmap='plasma', shading='auto')
            axes[1, 1].set_title('Black-Scholes Model - Put Options', fontweight='bold')
            axes[1, 1].set_xlabel('Strike Price (K)')
            axes[1, 1].set_ylabel('Time to Maturity (T)')
            axes[1, 1].set_xticks(strike_range)
            axes[1, 1].set_xticklabels([f'{k:.0f}' for k in strike_range], rotation=45)
            axes[1, 1].set_yticks(maturity_range)
            axes[1, 1].set_yticklabels([f'{t:.2f}' for t in maturity_range])
            plt.colorbar(im4, ax=axes[1, 1], label='Option Price')
            
            for i in range(len(maturity_range)):
                for j in range(len(strike_range)):
                    axes[1, 1].text(strike_range[j], maturity_range[i], f'{bs_put_prices[i, j]:.2f}',
                                    ha="center", va="center", color="white", fontsize=8, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.subheader("Price Differences Analysis")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max Call Difference", f"{np.max(call_diff):.4f}")
                st.metric("Min Call Difference", f"{np.min(call_diff):.4f}")
            with col2:
                st.metric("Max Put Difference", f"{np.max(put_diff):.4f}")
                st.metric("Min Put Difference", f"{np.min(put_diff):.4f}")
            with col3:
                st.metric("Avg Call Difference", f"{np.mean(call_diff):.4f}")
                st.metric("Avg Put Difference", f"{np.mean(put_diff):.4f}")
            
            fig_diff, axes_diff = plt.subplots(1, 2, figsize=(16, 6))
            fig_diff.suptitle('Price Differences: Heston - Black-Scholes', fontsize=14, fontweight='bold')
            
            im_diff1 = axes_diff[0].pcolormesh(X, Y, call_diff, cmap='RdBu_r', shading='auto')
            axes_diff[0].set_title('Call Options Difference', fontweight='bold')
            axes_diff[0].set_xlabel('Strike Price (K)')
            axes_diff[0].set_ylabel('Time to Maturity (T)')
            axes_diff[0].set_xticks(strike_range)
            axes_diff[0].set_xticklabels([f'{k:.0f}' for k in strike_range], rotation=45)
            axes_diff[0].set_yticks(maturity_range)
            axes_diff[0].set_yticklabels([f'{t:.2f}' for t in maturity_range])
            plt.colorbar(im_diff1, ax=axes_diff[0], label='Price Difference')
            
            for i in range(len(maturity_range)):
                for j in range(len(strike_range)):
                    axes_diff[0].text(strike_range[j], maturity_range[i], f'{call_diff[i, j]:.3f}',
                                    ha="center", va="center", color="white", fontsize=8, fontweight='bold')
            
            im_diff2 = axes_diff[1].pcolormesh(X, Y, put_diff, cmap='RdBu_r', shading='auto')
            axes_diff[1].set_title('Put Options Difference', fontweight='bold')
            axes_diff[1].set_xlabel('Strike Price (K)')
            axes_diff[1].set_ylabel('Time to Maturity (T)')
            axes_diff[1].set_xticks(strike_range)
            axes_diff[1].set_xticklabels([f'{k:.0f}' for k in strike_range], rotation=45)
            axes_diff[1].set_yticks(maturity_range)
            axes_diff[1].set_yticklabels([f'{t:.2f}' for t in maturity_range])
            plt.colorbar(im_diff2, ax=axes_diff[1], label='Price Difference')
            
            for i in range(len(maturity_range)):
                for j in range(len(strike_range)):
                    axes_diff[1].text(strike_range[j], maturity_range[i], f'{put_diff[i, j]:.3f}',
                                    ha="center", va="center", color="white", fontsize=8, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig_diff)
            plt.close(fig_diff)
            
            st.subheader("Summary Statistics")
            summary_data = {
                'Metric': ['Max Call Difference', 'Min Call Difference', 'Mean Call Difference', 'Std Call Difference',
                          'Max Put Difference', 'Min Put Difference', 'Mean Put Difference', 'Std Put Difference'],
                'Value': [
                    f"{np.max(call_diff):.4f}", f"{np.min(call_diff):.4f}", 
                    f"{np.mean(call_diff):.4f}", f"{np.std(call_diff):.4f}",
                    f"{np.max(put_diff):.4f}", f"{np.min(put_diff):.4f}", 
                    f"{np.mean(put_diff):.4f}", f"{np.std(put_diff):.4f}"
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            st.table(summary_df)
        except Exception as e:
            st.error(f"Error generating heatmap: {str(e)}")
        progress_bar.empty()

# Sidebar footer with system info
st.sidebar.markdown("---")
st.sidebar.subheader("System Performance")
process = psutil.Process()
memory_mb = process.memory_info().rss / 1024 / 1024
st.sidebar.text(f"Memory: {memory_mb:.1f} MB")
st.sidebar.text(f"CPU Cores: {psutil.cpu_count()}")
st.sidebar.text(f"Python: {sys.version.split()[0]}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Advanced Heston Model Implementation**")
st.sidebar.markdown("Professional-grade quantitative finance tool")
st.sidebar.markdown("Compatible with Python 3.8-3.12")