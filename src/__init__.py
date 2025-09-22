"""
Advanced Heston Model Option Pricing Package

A comprehensive implementation of the Heston stochastic volatility model
for pricing European options using Monte Carlo simulation.
"""

__version__ = "1.0.0"
__author__ = "Jorge Lucas González"
__email__ = "jorge.lucas.glez@gmail.com"

from .heston import (
    simulate_heston_paths,
    calculate_option_price_monte_carlo,
    black_scholes_price,
    calculate_heston_greeks
)

from .plots import (
    plot_heston_paths,
    plot_convergence_analysis,
    plot_sensitivity_analysis,
    plot_payoff_distribution,
    plot_price_vs_strike,
    plot_implied_volatility_smile,
    plot_volatility_surface
)

__all__ = [
    'simulate_heston_paths',
    'calculate_option_price_monte_carlo', 
    'black_scholes_price',
    'calculate_heston_greeks',
    'plot_heston_paths',
    'plot_convergence_analysis',
    'plot_sensitivity_analysis',
    'plot_payoff_distribution',
    'plot_price_vs_strike',
    'plot_implied_volatility_smile',
    'plot_volatility_surface'
]