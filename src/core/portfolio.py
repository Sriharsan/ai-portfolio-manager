from typing import Dict, List, Optional
import pandas as pd
from dataclasses import dataclass

@dataclass
class Portfolio:
    """Core Portfolio entity with validation"""
    
    def __init__(self, weights: Dict[str, float], name: str = "Portfolio"):
        self.weights = weights
        self.name = name
        self.created_at = pd.Timestamp.now()
        self.validate()
    
    def validate(self) -> bool:
        """Validate portfolio weights"""
        if not self.weights:
            raise ValueError("Portfolio cannot be empty")
        
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Weights sum to {total_weight:.3f}, not 1.0")
        
        for symbol, weight in self.weights.items():
            if weight < 0:
                raise ValueError(f"Negative weight for {symbol}: {weight}")
            if weight > 0.5:
                raise ValueError(f"Weight too high for {symbol}: {weight}")
        
        return True
    
    def rebalance(self, new_weights: Dict[str, float]) -> 'Portfolio':
        """Create rebalanced portfolio"""
        return Portfolio(new_weights, f"{self.name}_rebalanced")
    
    def get_concentration_risk(self) -> float:
        """Calculate concentration risk (Herfindahl index)"""
        return sum(w**2 for w in self.weights.values())
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'weights': self.weights,
            'created_at': self.created_at,
            'concentration_risk': self.get_concentration_risk()
        }