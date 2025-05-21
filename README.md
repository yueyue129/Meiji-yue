# Meiji-yue
this is a package about asset allocation

# Asset Optimizer

**Asset Optimizer** is a Python-based portfolio optimization package that supports multiple classical optimization strategies. It offers an end-to-end pipeline that integrates data preprocessing, portfolio optimization, performance evaluation, and rich visualizations.

This package is designed for quants, financial engineers, students, and researchers who aim to build and evaluate multi-asset portfolios efficiently.

---

## Features

- Support for multiple portfolio optimization methods (e.g., minimum variance, risk parity, maximum Sharpe)
- Built-in data preprocessing: automatic return calculation, covariance matrix, and expected returns
- One-line API: `run_pipeline()` for full workflow
- Visualization and HTML report generation
- CLI support for automation and scripting
- Modular and extensible architecture

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yueyue129/portfolio allocation.git
cd portfolio allocation

2. Install dependencies and the package (editable mode)

pip install -e .


⸻

Quick Start

Run in Python:

from asset_optimizer import run_pipeline

result = run_pipeline(
    data_path='./data',
    result_path='./result',
    method='risk parity',  # Options: min variance, max sharpe, risk parity, etc.
    Rf=0.02,
    freq='D'
)

print(result)  # Display performance metrics

Run from command line:

asset-optimize --data_path ./data --result_path ./result --method "risk parity"


⸻

Input Data Format

Place the following two Excel files in your data_path directory:
 • 标的净值.xlsx: Asset net value data (rows = dates, columns = asset names)
 • 产品信息.xlsx: Asset metadata (must include a column named “产品简称” with asset names)

⸻

Output

Results are saved to the result_path directory and include:
 • Portfolio weight bar chart and pie chart
 • Cumulative return plot
 • Automatically generated HTML report
 • Performance metrics: annual return, Sharpe ratio, max drawdown, etc.

⸻

Package Structure

asset_optimizer/
├── datapre.py           # Data loading and preprocessing
├── optimizer.py         # Portfolio optimization methods
├── evaluator.py         # Performance evaluation
├── visualizer.py        # Plotting and report generation
├── __init__.py          # Unified entry point: run_pipeline()


Author


Quantitative Researcher / Data Scientist
Email: 


