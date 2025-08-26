"""
Institutional Portfolio Builder - BlackRock-Level Portfolio Construction
Advanced portfolio templates and asset allocation strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class InstitutionalPortfolioBuilder:
    """Build institutional-grade diversified portfolios"""
    
    def __init__(self, market_data_provider):
        self.market_data = market_data_provider
        self.asset_universe = market_data_provider.asset_universe
        
        # Portfolio templates
        self.portfolio_templates = self._create_portfolio_templates()
        
    def _create_portfolio_templates(self) -> Dict[str, Dict]:
        """Create institutional portfolio templates"""
        
        templates = {
            'conservative': {
                'name': 'Conservative Balanced',
                'description': 'Capital preservation with moderate growth',
                'target_return': 0.06,
                'target_volatility': 0.08,
                'allocation': {
                    'bonds': 0.60,
                    'us_equity': 0.25,
                    'international_equity': 0.10,
                    'alternatives': 0.05
                }
            },
            'balanced': {
                'name': 'Balanced Growth',
                'description': 'Balanced growth and income',
                'target_return': 0.08,
                'target_volatility': 0.12,
                'allocation': {
                    'bonds': 0.40,
                    'us_equity': 0.35,
                    'international_equity': 0.15,
                    'alternatives': 0.10
                }
            },
            'growth': {
                'name': 'Growth Portfolio',
                'description': 'Long-term capital appreciation',
                'target_return': 0.10,
                'target_volatility': 0.16,
                'allocation': {
                    'bonds': 0.20,
                    'us_equity': 0.50,
                    'international_equity': 0.20,
                    'alternatives': 0.10
                }
            },
            'aggressive': {
                'name': 'Aggressive Growth',
                'description': 'Maximum growth potential',
                'target_return': 0.12,
                'target_volatility': 0.20,
                'allocation': {
                    'bonds': 0.10,
                    'us_equity': 0.60,
                    'international_equity': 0.25,
                    'alternatives': 0.05
                }
            },
            'institutional_endowment': {
                'name': 'Endowment Model',
                'description': 'Yale/Harvard endowment style',
                'target_return': 0.09,
                'target_volatility': 0.14,
                'allocation': {
                    'us_equity': 0.30,
                    'international_equity': 0.20,
                    'emerging_markets': 0.10,
                    'bonds': 0.15,
                    'real_estate': 0.10,
                    'commodities': 0.10,
                    'alternatives': 0.05
                }
            },
            'risk_parity': {
                'name': 'Risk Parity',
                'description': 'Equal risk contribution strategy',
                'target_return': 0.08,
                'target_volatility': 0.10,
                'allocation': {
                    'us_equity': 0.25,
                    'international_equity': 0.15,
                    'bonds': 0.35,
                    'commodities': 0.15,
                    'real_estate': 0.10
                }
            },
            'tactical_allocation': {
                'name': 'Tactical Asset Allocation',
                'description': 'Dynamic sector rotation',
                'target_return': 0.11,
                'target_volatility': 0.18,
                'allocation': {
                    'technology': 0.20,
                    'healthcare': 0.15,
                    'financials': 0.15,
                    'consumer_discretionary': 0.10,
                    'industrials': 0.10,
                    'energy': 0.08,
                    'utilities': 0.07,
                    'materials': 0.07,
                    'real_estate': 0.08
                }
            },
            'income_focused': {
                'name': 'Income Generation',
                'description': 'High dividend and bond income',
                'target_return': 0.07,
                'target_volatility': 0.10,
                'allocation': {
                    'dividend_stocks': 0.30,
                    'reits': 0.15,
                    'corporate_bonds': 0.25,
                    'government_bonds': 0.20,
                    'utilities': 0.10
                }
            }
        }
        
        return templates
    
    def build_portfolio_from_template(self, template_name: str, 
                                    customizations: Optional[Dict] = None) -> Dict[str, float]:
        """Build portfolio from institutional template"""
        
        if template_name not in self.portfolio_templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        template = self.portfolio_templates[template_name].copy()
        
        # Apply customizations if provided
        if customizations:
            for key, value in customizations.items():
                if key in template:
                    template[key] = value
        
        # Convert allocation categories to specific symbols
        portfolio = self._map_allocation_to_symbols(template['allocation'])
        
        return portfolio
    
    def _map_allocation_to_symbols(self, allocation: Dict[str, float]) -> Dict[str, float]:
        """Map allocation categories to specific symbols"""
        
        category_mappings = {
            'bonds': {
                'TLT': 0.30,  # Long-term Treasury
                'IEF': 0.25,  # Intermediate Treasury
                'LQD': 0.25,  # Investment Grade Corporate
                'TIP': 0.20   # TIPS
            },
            'government_bonds': {
                'TLT': 0.40,
                'IEF': 0.35,
                'SHY': 0.25
            },
            'corporate_bonds': {
                'LQD': 0.60,
                'HYG': 0.40
            },
            'us_equity': {
                'SPY': 0.40,   # Large Cap
                'QQQ': 0.25,   # Tech-heavy
                'IWM': 0.15,   # Small Cap
                'VTI': 0.20    # Total Market
            },
            'international_equity': {
                'VEA': 0.60,   # Developed Markets
                'VWO': 0.40    # Emerging Markets
            },
            'emerging_markets': {
                'VWO': 1.0
            },
            'real_estate': {
                'VNQ': 0.60,
                'IYR': 0.40
            },
            'reits': {
                'VNQ': 0.70,
                'XLRE': 0.30
            },
            'commodities': {
                'GLD': 0.50,
                'SLV': 0.25,
                'USO': 0.25
            },
            'alternatives': {
                'GLD': 0.40,
                'VNQ': 0.35,
                'VIXY': 0.25
            },
            'technology': {
                'XLK': 0.40,
                'QQQ': 0.30,
                'AAPL': 0.08,
                'MSFT': 0.07,
                'GOOGL': 0.07,
                'NVDA': 0.05,
                'META': 0.03
            },
            'healthcare': {
                'XLV': 0.50,
                'JNJ': 0.15,
                'UNH': 0.15,
                'PFE': 0.10,
                'ABBV': 0.10
            },
            'financials': {
                'XLF': 0.40,
                'JPM': 0.15,
                'BRK-B': 0.15,
                'V': 0.10,
                'MA': 0.10,
                'BAC': 0.10
            },
            'consumer_discretionary': {
                'XLY': 0.40,
                'AMZN': 0.15,
                'TSLA': 0.15,
                'HD': 0.15,
                'NKE': 0.15
            },
            'consumer_staples': {
                'XLP': 0.50,
                'PG': 0.20,
                'KO': 0.15,
                'PEP': 0.15
            },
            'industrials': {
                'XLI': 0.60,
                'BA': 0.15,
                'CAT': 0.15,
                'GE': 0.10
            },
            'energy': {
                'XLE': 0.50,
                'XOM': 0.20,
                'CVX': 0.20,
                'COP': 0.10
            },
            'utilities': {
                'XLU': 0.70,
                'NEE': 0.15,
                'DUK': 0.15
            },
            'materials': {
                'XLB': 0.70,
                'LIN': 0.15,
                'APD': 0.15
            },
            'dividend_stocks': {
                'VYM': 0.30,  # High Dividend Yield ETF
                'SCHD': 0.30, # Dividend Appreciation
                'JNJ': 0.10,
                'PG': 0.10,
                'KO': 0.10,
                'PFE': 0.10
            }
        }
        
        portfolio = {}
        
        for category, weight in allocation.items():
            if category in category_mappings:
                category_symbols = category_mappings[category]
                for symbol, symbol_weight in category_symbols.items():
                    if symbol in portfolio:
                        portfolio[symbol] += weight * symbol_weight
                    else:
                        portfolio[symbol] = weight * symbol_weight
        
        # Normalize to ensure weights sum to 1
        total_weight = sum(portfolio.values())
        if total_weight > 0:
            portfolio = {symbol: weight/total_weight for symbol, weight in portfolio.items()}
        
        return portfolio
    
    def create_sector_rotation_portfolio(self, momentum_lookback: int = 60) -> Dict[str, float]:
        """Create momentum-based sector rotation portfolio"""
        
        sector_etfs = ['XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLP', 'XLI', 'XLU', 'XLRE', 'XLB']
        
        sector_momentum = {}
        
        for etf in sector_etfs:
            try:
                data = self.market_data.get_stock_data_premium(etf, '6mo')
                if not data.empty and len(data) >= momentum_lookback:
                    # Calculate momentum score
                    returns = data['Daily_Return'].tail(momentum_lookback)
                    momentum_score = returns.mean() * np.sqrt(len(returns))
                    sector_momentum[etf] = momentum_score
            except Exception as e:
                logger.warning(f"Momentum calculation failed for {etf}: {e}")
        
        if not sector_momentum:
            # Fallback to equal weight
            return {etf: 1.0/len(sector_etfs) for etf in sector_etfs}
        
        # Rank sectors by momentum and allocate more to top performers
        sorted_sectors = sorted(sector_momentum.items(), key=lambda x: x[1], reverse=True)
        
        # Create allocation with bias toward top momentum sectors
        total_sectors = len(sorted_sectors)
        portfolio = {}
        
        for i, (sector, momentum) in enumerate(sorted_sectors):
            # Higher allocation for higher-ranked sectors
            rank_weight = (total_sectors - i) / sum(range(1, total_sectors + 1))
            portfolio[sector] = rank_weight
        
        return portfolio
    
    def create_risk_parity_portfolio(self, symbols: List[str]) -> Dict[str, float]:
        """Create risk parity portfolio (equal risk contribution)"""
        
        if len(symbols) < 2:
            return {symbol: 1.0/len(symbols) for symbol in symbols}
        
        # Get return data for all symbols
        returns_data = {}
        for symbol in symbols:
            try:
                data = self.market_data.get_stock_data_premium(symbol, '1y')
                if not data.empty and 'Daily_Return' in data.columns:
                    returns_data[symbol] = data['Daily_Return'].dropna()
            except:
                continue
        
        if len(returns_data) < 2:
            return {symbol: 1.0/len(symbols) for symbol in symbols}
        
        # Calculate risk-based weights
        volatilities = {}
        for symbol, returns in returns_data.items():
            volatilities[symbol] = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Inverse volatility weighting as approximation for risk parity
        inv_vol_weights = {}
        total_inv_vol = 0
        
        for symbol, vol in volatilities.items():
            if vol > 0:
                inv_vol = 1.0 / vol
                inv_vol_weights[symbol] = inv_vol
                total_inv_vol += inv_vol
        
        # Normalize weights
        risk_parity_weights = {}
        for symbol, inv_vol in inv_vol_weights.items():
            risk_parity_weights[symbol] = inv_vol / total_inv_vol
        
        # Add any missing symbols with equal weight
        remaining_symbols = set(symbols) - set(risk_parity_weights.keys())
        if remaining_symbols:
            remaining_weight = 0.1  # Reserve 10% for missing symbols
            individual_weight = remaining_weight / len(remaining_symbols)
            
            # Reduce existing weights proportionally
            adjustment_factor = (1.0 - remaining_weight)
            risk_parity_weights = {symbol: weight * adjustment_factor 
                                 for symbol, weight in risk_parity_weights.items()}
            
            # Add missing symbols
            for symbol in remaining_symbols:
                risk_parity_weights[symbol] = individual_weight
        
        return risk_parity_weights
    
    def create_smart_beta_portfolio(self, universe_filter: Optional[Dict] = None) -> Dict[str, float]:
        """Create smart beta portfolio using factor tilts"""
        
        # Get equity universe
        equity_symbols = self.market_data.get_available_assets(
            filter_by=universe_filter or {'type': 'equity', 'region': 'US', 'cap': 'large'}
        )
        
        if not equity_symbols:
            # Fallback to major stocks
            equity_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V']
        
        # Calculate factor scores
        factor_scores = {}
        
        for symbol in equity_symbols[:20]:  # Limit to top 20 for performance
            try:
                data = self.market_data.get_stock_data_premium(symbol, '1y')
                if not data.empty and len(data) > 100:
                    score = self._calculate_smart_beta_score(data)
                    if score is not None:
                        factor_scores[symbol] = score
            except:
                continue
        
        if not factor_scores:
            return {symbol: 1.0/len(equity_symbols[:10]) for symbol in equity_symbols[:10]}
        
        # Create weights based on factor scores
        sorted_stocks = sorted(factor_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Top 10 stocks with factor-based weighting
        top_stocks = sorted_stocks[:10]
        total_score = sum(score for _, score in top_stocks)
        
        portfolio = {}
        for symbol, score in top_stocks:
            portfolio[symbol] = score / total_score
        
        return portfolio
    
    def _calculate_smart_beta_score(self, data: pd.DataFrame) -> Optional[float]:
        """Calculate smart beta factor score"""
        
        try:
            returns = data['Daily_Return'].dropna()
            if len(returns) < 60:
                return None
            
            # Quality factor (lower volatility is better)
            volatility = returns.std() * np.sqrt(252)
            quality_score = max(0, 1 - volatility)  # Inverse relationship
            
            # Momentum factor (recent performance)
            momentum_score = returns.tail(60).mean() * 252  # Annualized recent return
            
            # Value factor (simplified using price trend)
            price_trend = (data['Close'].iloc[-1] / data['Close'].iloc[-252] - 1) if len(data) >= 252 else 0
            value_score = max(0, 0.1 - price_trend)  # Contrarian approach
            
            # Composite score
            composite_score = 0.4 * quality_score + 0.4 * momentum_score + 0.2 * value_score
            
            return max(0.01, composite_score)  # Minimum score
            
        except Exception:
            return None
    
    def get_portfolio_recommendations(self, risk_tolerance: str, 
                                   investment_horizon: str,
                                   income_focus: bool = False) -> List[Dict]:
        """Get portfolio recommendations based on preferences"""
        
        recommendations = []
        
        # Map preferences to templates
        if risk_tolerance.lower() == 'conservative':
            if income_focus:
                recommendations.append({
                    'name': 'Conservative Income',
                    'template': 'income_focused',
                    'score': 0.95
                })
            recommendations.append({
                'name': 'Conservative Balanced', 
                'template': 'conservative',
                'score': 0.90
            })
        
        elif risk_tolerance.lower() == 'moderate':
            recommendations.append({
                'name': 'Balanced Growth',
                'template': 'balanced', 
                'score': 0.95
            })
            recommendations.append({
                'name': 'Risk Parity',
                'template': 'risk_parity',
                'score': 0.85
            })
        
        elif risk_tolerance.lower() == 'aggressive':
            recommendations.append({
                'name': 'Growth Portfolio',
                'template': 'growth',
                'score': 0.90
            })
            if investment_horizon == 'long':
                recommendations.append({
                    'name': 'Endowment Model',
                    'template': 'institutional_endowment',
                    'score': 0.95
                })
        
        # Always include tactical allocation for active investors
        recommendations.append({
            'name': 'Tactical Allocation',
            'template': 'tactical_allocation',
            'score': 0.80
        })
        
        # Sort by score
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:3]  # Top 3 recommendations
    
    def get_available_templates(self) -> Dict[str, Dict]:
        """Get all available portfolio templates"""
        return self.portfolio_templates.copy()

# Usage example function
def create_institutional_portfolio(template_name: str = 'balanced',
                                 market_data_provider=None) -> Dict[str, float]:
    """Create an institutional-grade portfolio"""
    
    if market_data_provider is None:
        from enhanced_market_data import market_data_provider
    
    builder = InstitutionalPortfolioBuilder(market_data_provider)
    portfolio = builder.build_portfolio_from_template(template_name)
    
    return portfolio


optimizer = InstitutionalPortfolioBuilder(market_data_provider)
