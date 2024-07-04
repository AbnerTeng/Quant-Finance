"""
Logic utilities for trading strategies
"""
from typing import Any

def crossover(cross: Any, crossed: Any) -> bool:
    """
    Up crossover, mainly used for moving average strategies
    
    Scenarios
    ---------
    1. Close[i] > MA[i]
    2. FastMA[i] > SlowMA[i]
    """
    return cross > crossed

def crossdown(cross: Any, crossed: Any) -> bool:
    """
    Down crossover, mainly used for moving average strategies
    
    Scenarios
    ---------
    1. Close[i] < MA[i]
    2. FastMA[i] < SlowMA[i]
    """
    return cross <= crossed