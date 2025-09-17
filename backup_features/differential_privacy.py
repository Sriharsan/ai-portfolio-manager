import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings

class DifferentialPrivacyEngine:
    """
    Implements differential privacy mechanisms for financial data protection.
    Enables aggregate analytics while preserving individual privacy.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        """
        Initialize DP engine with privacy parameters.
        
        Args:
            epsilon: Privacy budget (smaller = more private)
            delta: Probability of privacy breach (smaller = more private)
        """
        self.epsilon = epsilon
        self.delta = delta
        self.privacy_spent = 0.0
    
    def laplace_mechanism(self, true_value: float, sensitivity: float) -> float:
        """
        Add Laplace noise for differential privacy.
        Used for numerical aggregates like average returns.
        """
        if self.privacy_spent + (sensitivity / self.epsilon) > 1.0:
            warnings.warn("Privacy budget nearly exhausted!")
        
        # Add Laplace noise
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        self.privacy_spent += sensitivity / self.epsilon
        
        return true_value + noise
    
    def exponential_mechanism(self, candidates: List, utility_scores: List[float], 
                            sensitivity: float) -> any:
        """
        Select from candidates based on utility while preserving privacy.
        Used for selecting portfolio strategies or asset classes.
        """
        # Calculate probabilities using exponential mechanism
        scaled_utilities = np.array(utility_scores) * self.epsilon / (2 * sensitivity)
        max_utility = np.max(scaled_utilities)
        
        # Numerical stability: subtract max
        stable_utilities = scaled_utilities - max_utility
        probabilities = np.exp(stable_utilities)
        probabilities = probabilities / np.sum(probabilities)
        
        # Sample according to probabilities
        selected_idx = np.random.choice(len(candidates), p=probabilities)
        self.privacy_spent += sensitivity / self.epsilon
        
        return candidates[selected_idx]
    
    def private_portfolio_analytics(self, user_portfolios: List[Dict[str, float]], 
                                  query_type: str) -> Dict:
        """
        Compute differentially private aggregate statistics across user portfolios.
        Enables market insights without revealing individual positions.
        """
        if not user_portfolios:
            return {}
        
        # Convert to DataFrame for easier manipulation
        portfolio_df = pd.DataFrame(user_portfolios).fillna(0.0)
        
        results = {}
        
        if query_type == "average_allocation":
            # Average asset allocation across all users
            for asset in portfolio_df.columns:
                true_average = portfolio_df[asset].mean()
                # Sensitivity: max change if one user changes allocation by 100%
                sensitivity = 1.0 / len(user_portfolios)
                private_average = self.laplace_mechanism(true_average, sensitivity)
                results[f"avg_{asset}"] = max(0.0, private_average)  # Ensure non-negative
        
        elif query_type == "portfolio_diversity":
            # Average number of assets held per portfolio
            diversity_scores = [(portfolio != 0).sum() for portfolio in user_portfolios]
            true_avg_diversity = np.mean(diversity_scores)
            sensitivity = 1.0  # One user can change diversity by at most 1
            results["avg_diversity"] = self.laplace_mechanism(true_avg_diversity, sensitivity)
        
        elif query_type == "risk_distribution":
            # Distribution of risk preferences (assuming risk score per portfolio)
            # This would need risk scores as input in practice
            risk_buckets = ["conservative", "moderate", "aggressive"]
            # Simulate risk categorization (in practice, computed from portfolio metrics)
            risk_categories = np.random.choice(risk_buckets, len(user_portfolios))
            
            for bucket in risk_buckets:
                true_count = np.sum(risk_categories == bucket)
                sensitivity = 1.0  # One user can change count by at most 1
                private_count = self.laplace_mechanism(true_count, sensitivity)
                results[f"{bucket}_count"] = max(0, int(round(private_count)))
        
        return results
    
    def check_privacy_budget(self) -> Dict:
        """Return current privacy budget status."""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "privacy_spent": self.privacy_spent,
            "budget_remaining": max(0, 1.0 - self.privacy_spent)
        }

# Example usage for interview demonstration
def demonstrate_differential_privacy():
    """Demonstrate privacy-preserving portfolio analytics."""
    
    # Initialize DP engine
    dp_engine = DifferentialPrivacyEngine(epsilon=1.0)
    
    # Simulate user portfolios (in practice, these would be real user data)
    sample_portfolios = [
        {"AAPL": 0.3, "MSFT": 0.2, "GOOGL": 0.15, "TSLA": 0.1, "SPY": 0.25},
        {"AAPL": 0.25, "MSFT": 0.25, "BND": 0.3, "VTI": 0.2},
        {"TSLA": 0.4, "NVDA": 0.3, "AMD": 0.2, "ARKK": 0.1},
        {"SPY": 0.5, "BND": 0.3, "VEA": 0.2},
        {"AAPL": 0.2, "MSFT": 0.2, "GOOGL": 0.2, "AMZN": 0.2, "META": 0.2}
    ]
    
    print("Differential Privacy Demonstration:")
    
    # Compute private aggregate statistics
    avg_allocations = dp_engine.private_portfolio_analytics(sample_portfolios, "average_allocation")
    print("\nPrivate Average Allocations:")
    for asset, allocation in avg_allocations.items():
        print(f"{asset}: {allocation:.3f}")
    
    # Check privacy budget
    budget_status = dp_engine.check_privacy_budget()
    print(f"\nPrivacy Budget Status:")
    print(f"Remaining: {budget_status['budget_remaining']:.3f}")
