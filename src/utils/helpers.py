# src/utils/helpers.py

import pandas as pd
import numpy as np
from typing import Dict, List, Union
import re

def validate_portfolio_weights(portfolio: Dict[str, float], tolerance: float = 0.01) -> bool:
    return abs(sum(portfolio.values()) - 1.0) <= tolerance

def format_currency(value: float) -> str:
    if abs(value) >= 1e9:
        return f"${value/1e9:.1f}B"
    elif abs(value) >= 1e6:
        return f"${value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"${value/1e3:.1f}K"
    else:
        return f"${value:.2f}"

def format_percentage(value: float, decimals: int = 1) -> str:
    sign = "+" if value > 0 else ""
    return f"{sign}{value*100:.{decimals}f}%"

def calculate_portfolio_value(weights: Dict[str, float], prices: Dict[str, float], 
                            initial_value: float = 100000) -> float:
    return sum(weight * prices.get(symbol, 0) * initial_value / prices.get(symbol, 1) 
               for symbol, weight in weights.items())

def clean_symbol(symbol: str) -> str:
    symbol = re.sub(r'[^A-Z]', '', symbol.upper())
    return symbol if len(symbol) <= 5 else symbol[:5]

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator != 0 else default