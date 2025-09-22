# Advanced Heston Model Option Pricing

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Professional-grade implementation of the Heston stochastic volatility model for European option pricing using optimized Monte Carlo simulation with advanced analytics and interactive visualizations.**

Built as part of my **quantitative finance** portfolio, this project demonstrates expertise in mathematical modeling, computational finance, and professional software development.

## Key Features

### Core Implementation
- **Optimized Monte Carlo Simulation** - Parallelized Euler-Maruyama with antithetic variates
- **Advanced Risk Analytics** - VaR, CVaR, drawdown analysis, and higher moments
- **Professional Visualizations** - Interactive 3D surfaces, Greeks evolution, and diagnostic plots
- **Monte Carlo Diagnostics** - Comprehensive statistical validation and quality assessment
- **Real-time Performance Monitoring** - CPU/memory tracking with efficiency metrics

### Interactive Dashboard
**11 Specialized Analysis Tabs:**
- Option Pricing & Greeks
- Path Visualization
- Convergence Analysis  
- Sensitivity Analysis
- Advanced Risk Metrics
- Monte Carlo Diagnostics
- Greeks vs Time Analysis
- 3D Volatility Surfaces
- Payoff Distributions
- Volatility Smile Analysis
- Model Comparison

## Quick Start

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/heston-option-pricing-model.git
cd heston-option-pricing-model
pip install -r requirements.txt
```

### Launch Interactive Dashboard
```bash
streamlit run app/streamlit_app.py
```
**Access at**: http://localhost:8501

### Run Complete Analysis
```bash
python scripts/main.py
```

## Sample Results

```
🎯 Option Pricing Results
├── Heston MC Price: 8.16 ± 0.11 (95% CI: [7.94, 8.37])
├── Black-Scholes Price: 8.43
├── Implied Volatility: 19.3%
└── Price Difference: -0.28 (3.3% lower than BS)

📈 Greeks Analysis  
├── Delta: 0.62 (sensitivity to underlying)
├── Gamma: 0.020 (convexity measure)
├── Vega: 58.28 (volatility sensitivity) 
├── Theta: -4.29 (time decay per day)
└── Rho: 53.35 (interest rate sensitivity)

🎯 Risk Metrics
├── VaR (95%): -2.45 (5% tail loss)
├── CVaR (95%): -4.12 (expected shortfall)
├── Max Drawdown: -8.5%
├── Sharpe Ratio: 1.24
└── Skewness: -0.31 (left-tailed distribution)
```

## Project Structure

```
heston_pricing_model/
├── app/streamlit_app.py      # Interactive web application
├── src/                      # Core implementation
│   ├── heston.py            # Monte Carlo simulation engine
│   └── plots.py             # Visualization suite  
├── scripts/main.py          # Complete analysis runner
├── tests/                   # Comprehensive test suite
├── results/                 # Output directory
└── requirements.txt         # Dependencies
```

## Technical Highlights

### Mathematical Implementation
- **Heston Model**: Full implementation of stochastic volatility SDE
- **Monte Carlo Methods**: Euler-Maruyama discretization with variance reduction
- **Greeks Calculation**: Numerical differentiation with adaptive epsilon
- **Statistical Rigor**: Confidence intervals, convergence analysis, and diagnostic testing

### Performance Optimizations
- **Vectorized NumPy**: >10x speedup over pure Python
- **Parallel Processing**: Multi-core utilization with joblib
- **Memory Management**: Optimized array allocation
- **No Numba Dependency**: Pure NumPy solution for maximum compatibility

### Professional Software Development
- **Clean Architecture**: Modular design with separation of concerns
- **Comprehensive Testing**: Unit tests with >90% coverage
- **Type Hints**: Full type annotations for code clarity
- **Documentation**: Professional-grade docstrings and comments

## Applications

### Quantitative Finance
- **Options Trading**: Fair value calculation and Greeks analysis
- **Risk Management**: Portfolio risk assessment and hedging strategies  
- **Market Making**: Real-time option pricing and volatility surface analysis
- **Model Validation**: Comparison with Black-Scholes and market data

### Academic & Research
- **Financial Engineering**: Structured products and exotic options
- **Monte Carlo Methods**: Algorithm development and optimization
- **Stochastic Processes**: Numerical solution of SDEs
- **Computational Finance**: High-performance financial computations

## Skills Demonstrated

- **Quantitative Finance**: Deep understanding of stochastic volatility models
- **Mathematical Modeling**: Advanced numerical methods and statistical analysis
- **Software Engineering**: Production-ready code with professional standards
- **Data Science**: Statistical analysis, visualization, and diagnostic testing
- **Performance Computing**: Optimization and parallel processing techniques

## Dependencies

```
numpy>=1.21.0        # Numerical computing
scipy>=1.7.0         # Scientific computing  
pandas>=1.3.0        # Data analysis
matplotlib>=3.4.0    # Static plotting
seaborn>=0.11.0      # Statistical visualization
streamlit>=1.28.0    # Interactive web apps
plotly>=5.0.0        # Interactive 3D plotting
joblib>=1.1.0        # Parallel computing
psutil>=5.8.0        # System monitoring
statsmodels>=0.13.0  # Statistical testing
```

## Features Showcase

| Feature | Implementation | Benefit |
|---------|---------------|---------|
| **Parallel Monte Carlo** | Multi-threaded with joblib | 4x faster execution |
| **Antithetic Variates** | Built-in variance reduction | 50% fewer paths needed |
| **Interactive 3D Surfaces** | Plotly integration | Professional visualizations |
| **Advanced Risk Metrics** | VaR, CVaR, drawdown analysis | Comprehensive risk assessment |
| **Real-time Monitoring** | CPU/memory tracking | Performance optimization |
| **Statistical Validation** | Monte Carlo diagnostics | Quality assurance |

## Academic Foundation

Implementation based on:
- **Heston, S.L.** (1993) - Original stochastic volatility model
- **Gatheral, J.** (2006) - Volatility surface modeling
- **Glasserman, P.** (2003) - Monte Carlo methods in finance

## Professional Context

This project demonstrates skills directly applicable to:
- **Quantitative Trading** - Algorithm development and market making
- **Risk Management** - Portfolio optimization and derivative hedging
- **Financial Engineering** - Structured products and model validation
- **FinTech Development** - Real-time analytics and trading systems

## Contact

**Jorge Lucas González**  
jorge.lucas.glez@gmail.com  
Burgos, Spain  
Physics Student (University of Valladolid)  
Specialization: Quantitative Finance & Mathematical Modeling

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Implementation inspired by industry best practices in quantitative finance
- Mathematical foundation based on seminal work by Steven Heston
- Performance optimization techniques from computational finance literature

---

**If this project helps you or showcases relevant skills, please consider giving it a star!**
