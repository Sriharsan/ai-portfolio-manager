# src/models/portfolio_optimizer.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple
import logging

class InstitutionalPortfolioBuilder:
    """BlackRock-level portfolio construction integrating ALL advanced optimization methods"""
    
    def __init__(self, market_data_provider):
        self.market_data = market_data_provider
        self.logger = logging.getLogger(__name__)
        self.templates = self._initialize_templates()
    
        try:
            from src.models.advanced_optimization import institutional_optimizer
            from src.models.risk_manager import create_risk_manager 
            from src.models.ml_engine import ml_engine
            from src.analytics.backtesting import professional_backtester
            from src.data.alternative_data import alternative_data_processor
        
            self.advanced_optimizer = institutional_optimizer
            self.risk_manager = create_risk_manager()  
            self.ml_engine = ml_engine
            self.backtester = professional_backtester
            self.alt_data = alternative_data_processor
        
            self.advanced_features_available = True
            self.logger.info("✅ ALL Advanced optimization engines loaded successfully")
        
        except ImportError as e:
            self.logger.warning(f"Advanced features unavailable: {e}")
            # Fallback to basic risk manager
            from .risk_manager import create_risk_manager
            self.risk_manager = create_risk_manager()
            self.advanced_features_available = False
    
    def _initialize_templates(self) -> Dict:
        """Initialize institutional portfolio templates"""
        return {
            'conservative': {
                'name': 'Conservative Growth',
                'description': 'Capital preservation with modest growth',
                'target_return': 0.06,
                'target_volatility': 0.08,
                'allocation': {
                    'TLT': 0.30,    # Long-term Treasury
                    'IEF': 0.20,    # Intermediate Treasury
                    'LQD': 0.15,    # Investment Grade Corporate
                    'SPY': 0.25,    # US Large Cap
                    'VEA': 0.10     # International Developed
                }
            },
            'balanced': {
                'name': 'Balanced Growth',
                'description': 'Balanced approach between growth and stability',
                'target_return': 0.08,
                'target_volatility': 0.12,
                'allocation': {
                    'SPY': 0.30,    # US Large Cap
                    'QQQ': 0.15,    # Tech-heavy NASDAQ
                    'VEA': 0.15,    # International Developed
                    'VWO': 0.05,    # Emerging Markets
                    'TLT': 0.15,    # Long-term Treasury
                    'LQD': 0.10,    # Investment Grade Corporate
                    'VNQ': 0.10     # REITs
                }
            },
            'growth': {
                'name': 'Growth Focused',
                'description': 'Maximum growth potential with higher volatility',
                'target_return': 0.10,
                'target_volatility': 0.16,
                'allocation': {
                    'SPY': 0.25,    # US Large Cap
                    'QQQ': 0.20,    # NASDAQ 100
                    'VEA': 0.20,    # International Developed
                    'VWO': 0.15,    # Emerging Markets
                    'IWM': 0.10,    # Small Cap
                    'TLT': 0.05,    # Minimal bonds
                    'GLD': 0.05     # Gold hedge
                }
            },
            'income_focused': {
                'name': 'Income Generation',
                'description': 'High dividend yield with moderate growth',
                'target_return': 0.07,
                'target_volatility': 0.10,
                'allocation': {
                    'SPY': 0.20,    # US Large Cap
                    'VYM': 0.15,    # High Dividend Yield (use SPY if not available)
                    'VNQ': 0.20,    # REITs
                    'LQD': 0.20,    # Investment Grade Corporate
                    'HYG': 0.15,    # High Yield Corporate
                    'TIP': 0.10     # TIPS
                }
            },
            'institutional_endowment': {
                'name': 'Endowment Model',
                'description': 'Yale/Harvard endowment-style diversification',
                'target_return': 0.09,
                'target_volatility': 0.14,
                'allocation': {
                    'SPY': 0.20,    # US Equity
                    'VEA': 0.15,    # International Equity
                    'VWO': 0.10,    # Emerging Markets
                    'VNQ': 0.15,    # Real Estate
                    'GLD': 0.10,    # Commodities/Gold
                    'TLT': 0.15,    # Long-term Bonds
                    'HYG': 0.10,    # High Yield
                    'USO': 0.05     # Oil/Energy
                }
            }
        }
    
    def get_available_templates(self) -> Dict:
        """Return available portfolio templates"""
        return self.templates
    
    # =============================================================================
    # TEMPLATE-BASED PORTFOLIO BUILDING
    # =============================================================================
    
    def build_portfolio_from_template(self, template_name: str) -> Dict[str, float]:
        """Build portfolio from template"""
        try:
            if template_name not in self.templates:
                self.logger.warning(f"Template {template_name} not found, using balanced")
                template_name = 'balanced'
            
            template = self.templates[template_name]
            allocation = template['allocation'].copy()
            
            # Verify assets are available and adjust if needed
            available_assets = self.market_data.get_available_assets()
            adjusted_allocation = {}
            
            for symbol, weight in allocation.items():
                if symbol in available_assets:
                    adjusted_allocation[symbol] = weight
                else:
                    # Find substitute
                    substitute = self._find_substitute(symbol, available_assets)
                    if substitute:
                        adjusted_allocation[substitute] = weight
                        self.logger.info(f"Substituted {symbol} with {substitute}")
            
            # Normalize weights to sum to 1
            total_weight = sum(adjusted_allocation.values())
            if total_weight > 0:
                adjusted_allocation = {k: v/total_weight for k, v in adjusted_allocation.items()}
            
            self.logger.info(f"Built {template_name} portfolio with {len(adjusted_allocation)} assets")
            return adjusted_allocation
            
        except Exception as e:
            self.logger.error(f"Portfolio building failed: {e}")
            return self._fallback_portfolio()
    
    # =============================================================================
    # ADVANCED OPTIMIZATION METHODS (MARKOWITZ, BLACK-LITTERMAN, RISK PARITY)
    # =============================================================================
    
    def optimize_portfolio_advanced(self, symbols: List[str], 
                                  method: str = 'markowitz',
                                  target_return: Optional[float] = None,
                                  market_views: Optional[Dict[str, float]] = None,
                                  **kwargs) -> Dict:
        """Advanced portfolio optimization using institutional-grade methods"""
        
        if not self.advanced_features_available:
            self.logger.warning("Advanced optimization not available, using basic methods")
            return self._basic_optimization(symbols)
        
        try:
            # Get returns data for optimization
            returns_data = self._get_returns_matrix(symbols)
            
            if returns_data.empty:
                return {'error': 'Insufficient data for optimization'}
            
            # Apply selected optimization method
            if method == 'markowitz':
                result = self.advanced_optimizer.markowitz_optimization(
                    returns_data, target_return=target_return, **kwargs
                )
            
            elif method == 'black_litterman':
                # Get market cap data (simplified for now)
                market_caps = {symbol: 1000000000 for symbol in symbols}  # Equal caps
                views = market_views or {}
                
                result = self.advanced_optimizer.black_litterman_optimization(
                    returns_data, market_caps, views, **kwargs
                )
            
            elif method == 'risk_parity':
                result = self.advanced_optimizer.risk_parity_optimization(returns_data)
            
            elif method == 'ml_enhanced':
                result = self._ml_enhanced_optimization(returns_data, symbols)
            
            else:
                return {'error': f'Unknown optimization method: {method}'}
            
            # Add comprehensive risk analysis if optimization successful
            if 'weights' in result and result.get('optimization_status') == 'optimal':
                portfolio_weights = result['weights']
                
                # Calculate comprehensive risk metrics
                result['risk_analysis'] = self._comprehensive_risk_analysis(
                    returns_data, portfolio_weights
                )
                
                # Add alternative data insights
                result['alternative_insights'] = self._get_alternative_insights(symbols)
                
            return result
            
        except Exception as e:
            self.logger.error(f"Advanced optimization failed: {e}")
            return self._basic_optimization(symbols)
    
    def markowitz_optimization(self, symbols: List[str], **kwargs) -> Dict:
        """Markowitz mean-variance optimization"""
        return self.optimize_portfolio_advanced(symbols, method='markowitz', **kwargs)
    
    def black_litterman_optimization(self, symbols: List[str], 
                                   market_views: Dict[str, float], **kwargs) -> Dict:
        """Black-Litterman optimization with investor views"""
        return self.optimize_portfolio_advanced(
            symbols, method='black_litterman', 
            market_views=market_views, **kwargs
        )
    
    def risk_parity_optimization(self, symbols: List[str], **kwargs) -> Dict:
        """Risk parity optimization"""
        return self.optimize_portfolio_advanced(symbols, method='risk_parity', **kwargs)
    
    def ml_enhanced_optimization(self, symbols: List[str], **kwargs) -> Dict:
        """Machine learning enhanced optimization"""
        return self.optimize_portfolio_advanced(symbols, method='ml_enhanced', **kwargs)
    
    # =============================================================================
    # ML-ENHANCED OPTIMIZATION
    # =============================================================================
    
    def _ml_enhanced_optimization(self, returns_data: pd.DataFrame, symbols: List[str]) -> Dict:
        """ML-enhanced optimization combining LSTM + Ensemble predictions with Markowitz"""
        try:
            # Get ML predictions for each asset
            ml_predictions = {}
            
            for symbol in symbols:
                # Create price series for ML analysis
                symbol_data = self.market_data.get_stock_data(symbol, '1y')
                if not symbol_data.empty:
                    
                    # LSTM prediction
                    lstm_result = self.ml_engine.lstm_return_prediction(
                        symbol_data, symbol, prediction_days=5
                    )
                    
                    # Ensemble prediction  
                    ensemble_result = self.ml_engine.ensemble_forecasting(
                        symbol_data, symbol
                    )
                    
                    # Combine predictions (60% LSTM, 40% Ensemble)
                    if 'prediction' in lstm_result and 'prediction' in ensemble_result:
                        combined_pred = (lstm_result['prediction'] * 0.6 + 
                                       ensemble_result['prediction'] * 0.4)
                        ml_predictions[symbol] = combined_pred
            
            if not ml_predictions:
                # Fallback to traditional Markowitz
                return self.advanced_optimizer.markowitz_optimization(returns_data)
            
            # Adjust expected returns based on ML predictions
            adjusted_returns = returns_data.copy()
            
            for symbol, prediction in ml_predictions.items():
                if symbol in adjusted_returns.columns:
                    historical_mean = adjusted_returns[symbol].mean()
                    # Blend historical with ML prediction (70% historical, 30% ML)
                    blended_return = 0.7 * historical_mean + 0.3 * prediction
                    
                    # Adjust the return series
                    adjustment = blended_return - historical_mean
                    adjusted_returns[symbol] = adjusted_returns[symbol] + adjustment
            
            # Run Markowitz optimization with ML-adjusted returns
            ml_result = self.advanced_optimizer.markowitz_optimization(adjusted_returns)
            
            # Add ML information to result
            if 'weights' in ml_result:
                ml_result['ml_predictions'] = ml_predictions
                ml_result['optimization_method'] = 'ml_enhanced_markowitz'
                ml_result['enhancement'] = 'LSTM + Ensemble predictions integrated'
            
            return ml_result
            
        except Exception as e:
            self.logger.error(f"ML-enhanced optimization failed: {e}")
            return self.advanced_optimizer.markowitz_optimization(returns_data)
    
    # =============================================================================
    # COMPREHENSIVE INSTITUTIONAL RISK ANALYSIS
    # =============================================================================
        
    def _comprehensive_risk_analysis(self, returns_data: pd.DataFrame, 
                                 weights: Dict[str, float]) -> Dict:
        """Comprehensive institutional-grade risk analysis"""
        try:
            # Calculate portfolio returns
            portfolio_returns = []
            for date in returns_data.index:
                daily_return = sum(
                    weights.get(symbol, 0) * returns_data.loc[date, symbol]
                    for symbol in returns_data.columns
                    if not pd.isna(returns_data.loc[date, symbol])
                )   
                portfolio_returns.append(daily_return)

            portfolio_returns_series = pd.Series(portfolio_returns, index=returns_data.index)

            # === Institutional Risk Manager Calls ===
            try:
                institutional_var = self.risk_manager.monte_carlo_var(portfolio_returns_series)
            except Exception as e:
                self.logger.warning(f"Monte Carlo VaR failed: {e}")
                institutional_var = {}

            try:
                garch_forecast = self.risk_manager.garch_volatility_forecast(portfolio_returns_series)
            except Exception as e:
                self.logger.warning(f"GARCH volatility forecast failed: {e}")
                garch_forecast = {}

            try:
                stress_test = self.risk_manager.stress_test_scenarios(weights, returns_data)
            except Exception as e:
                self.logger.warning(f"Stress test failed: {e}")
                stress_test = {}

            try:
                corr_analysis = self.risk_manager.correlation_breakdown_analysis(returns_data)
            except Exception as e:
                self.logger.warning(f"Correlation breakdown failed: {e}")
                corr_analysis = {}

            # === Build Risk Report ===
            risk_report = {
                "VaR (1-day, 95%)": institutional_var.get("var_95", 0),
                "CVaR (1-day, 95%)": institutional_var.get("cvar_95", 0),
                "Maximum Drawdown": self.risk_manager.max_drawdown(portfolio_returns_series),
                "GARCH Forecast": garch_forecast,
                "Stress Test": stress_test,
                "Correlation Breakdown": corr_analysis
            }

            return risk_report

        except Exception as e:
            self.logger.error(f"Risk analysis failed: {e}")
            return {"error": str(e)}

    
    # =============================================================================
    # ALTERNATIVE DATA INTEGRATION
    # =============================================================================
    
    def _get_alternative_insights(self, symbols: List[str]) -> Dict:
        """Get alternative data insights for portfolio symbols"""
        try:
            insights = {}
            
            # Social sentiment aggregation
            sentiment_result = self.alt_data.social_sentiment_aggregation(symbols)
            insights['social_sentiment'] = sentiment_result
            
            # Individual symbol analysis
            symbol_insights = {}
            for symbol in symbols[:5]:  # Limit to top 5 for performance
                
                # Earnings call NLP analysis
                earnings_analysis = self.alt_data.earnings_call_nlp_analysis(symbol)
                
                # Supply chain analytics
                supply_chain = self.alt_data.supply_chain_analysis(symbol)
                
                symbol_insights[symbol] = {
                    'earnings_sentiment': earnings_analysis,
                    'supply_chain': supply_chain
                }
            
            insights['individual_symbols'] = symbol_insights
            
            # Satellite economic indicators
            satellite_data = self.alt_data.satellite_economic_indicators()
            insights['economic_indicators'] = satellite_data
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Alternative insights failed: {e}")
            return {'error': str(e)}
    
    # =============================================================================
    # PROFESSIONAL BACKTESTING
    # =============================================================================
    
    def run_backtest(self, optimization_method: str, symbols: List[str], 
                    test_type: str = 'walk_forward') -> Dict:
        """Run comprehensive backtesting analysis"""
        
        if not self.advanced_features_available:
            return {'error': 'Backtesting requires advanced features'}
        
        try:
            # Get historical data
            returns_data = self._get_returns_matrix(symbols, period='2y')
            
            if returns_data.empty:
                return {'error': 'Insufficient historical data for backtesting'}
            
            # Define optimization function
            def optimization_func(train_data):
                return self.optimize_portfolio_advanced(
                    train_data.columns.tolist(), 
                    method=optimization_method
                )
            
            if test_type == 'walk_forward':
                # Walk-forward analysis with transaction costs
                results = self.backtester.walk_forward_analysis(
                    returns_data, optimization_func, 
                    window_size=252, rebalance_freq=63
                )
            
            elif test_type == 'out_of_sample':
                # Out-of-sample testing
                results = self.backtester.out_of_sample_testing(
                    returns_data, optimization_func
                )
            
            else:
                return {'error': f'Unknown test type: {test_type}'}
            
            return results
            
        except Exception as e:
            self.logger.error(f"Backtesting failed: {e}")
            return {'error': str(e)}
    
    # =============================================================================
    # SMART BETA AND FACTOR-BASED STRATEGIES
    # =============================================================================
    
    def create_smart_beta_portfolio(self) -> Dict[str, float]:
        """Create smart beta factor-based portfolio"""
        # Use sector ETFs for factor exposure
        smart_beta_allocation = {
            'XLK': 0.20,  # Technology (Growth factor)
            'XLF': 0.15,  # Financials (Value factor)  
            'XLV': 0.15,  # Healthcare (Quality factor)
            'XLI': 0.10,  # Industrials (Momentum factor)
            'XLE': 0.05,  # Energy (Value factor)
            'XLU': 0.10,  # Utilities (Low volatility)
            'XLP': 0.10,  # Consumer Staples (Quality)
            'XLY': 0.10,  # Consumer Discretionary (Growth)
            'XLRE': 0.05  # Real Estate (Dividend factor)
        }
        
        self.logger.info("Created smart beta factor-based portfolio")
        return smart_beta_allocation
    
    def create_sector_rotation_portfolio(self, lookback_days: int = 60) -> Dict[str, float]:
        """Create sector rotation based on momentum"""
        sector_etfs = ['XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLP', 'XLI', 'XLU', 'XLRE', 'XLB']
        
        try:
            # Get momentum scores for each sector
            momentum_scores = {}
            
            for sector in sector_etfs:
                try:
                    data = self.market_data.get_stock_data(sector, '6mo')
                    if not data.empty and len(data) >= lookback_days:
                        # Calculate momentum as return over lookback period
                        returns = data['Close'].pct_change().dropna()
                        if len(returns) >= lookback_days:
                            momentum = returns.tail(lookback_days).mean()
                            momentum_scores[sector] = momentum
                except Exception as e:
                    self.logger.warning(f"Failed to get data for {sector}: {e}")
            
            if not momentum_scores:
                self.logger.warning("No momentum data available, using smart beta")
                return self.create_smart_beta_portfolio()
            
            # Sort by momentum and select top sectors
            sorted_sectors = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
            top_sectors = sorted_sectors[:5]  # Top 5 momentum sectors
            
            # Create portfolio with momentum-weighted allocation
            total_momentum = sum(max(0, score) for _, score in top_sectors)
            
            if total_momentum > 0:
                portfolio = {}
                for sector, score in top_sectors:
                    if score > 0:
                        portfolio[sector] = max(0.1, score / total_momentum)  # Min 10% allocation
            else:
                # Fallback to equal weights
                equal_weight = 1.0 / len(top_sectors)
                portfolio = {sector: equal_weight for sector, _ in top_sectors}
            
            # Normalize to sum to 1
            total_weight = sum(portfolio.values())
            portfolio = {k: v/total_weight for k, v in portfolio.items()}
            
            self.logger.info(f"Created sector rotation portfolio with {len(portfolio)} sectors")
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Sector rotation failed: {e}")
            return self.create_smart_beta_portfolio()  # Fallback
    
    def create_risk_parity_portfolio(self, symbols: List[str]) -> Dict[str, float]:
        """Create risk parity portfolio where each asset contributes equal risk"""
        try:
            # Get returns data for all symbols
            returns_data = {}
            
            for symbol in symbols:
                try:
                    data = self.market_data.get_stock_data(symbol, '1y')
                    if not data.empty and 'Daily_Return' in data.columns:
                        returns = data['Daily_Return'].dropna()
                        if len(returns) > 50:  # Minimum data requirement
                            returns_data[symbol] = returns
                except Exception as e:
                    self.logger.warning(f"Failed to get data for {symbol}: {e}")
            
            if len(returns_data) < 2:
                self.logger.warning("Insufficient data for risk parity, using equal weights")
                equal_weight = 1.0 / len(symbols)
                return {symbol: equal_weight for symbol in symbols}
            
            # Calculate volatilities (risk proxy)
            volatilities = {}
            for symbol, returns in returns_data.items():
                vol = returns.std() * np.sqrt(252)  # Annualized volatility
                volatilities[symbol] = vol
            
            # Risk parity: weight inversely proportional to volatility
            inv_vol_weights = {}
            for symbol, vol in volatilities.items():
                if vol > 0:
                    inv_vol_weights[symbol] = 1.0 / vol
            
            # Normalize weights
            total_weight = sum(inv_vol_weights.values())
            if total_weight > 0:
                risk_parity_portfolio = {k: v/total_weight for k, v in inv_vol_weights.items()}
            else:
                equal_weight = 1.0 / len(symbols)
                risk_parity_portfolio = {symbol: equal_weight for symbol in symbols}
            
            self.logger.info(f"Created risk parity portfolio with {len(risk_parity_portfolio)} assets")
            return risk_parity_portfolio
            
        except Exception as e:
            self.logger.error(f"Risk parity creation failed: {e}")
            # Fallback to equal weights
            equal_weight = 1.0 / min(len(symbols), 5)
            return {symbol: equal_weight for symbol in symbols}
    
    # =============================================================================
    # PORTFOLIO RECOMMENDATIONS ENGINE
    # =============================================================================
    
    def get_portfolio_recommendations(self, risk_tolerance: str, 
                                   time_horizon: str, 
                                   income_focus: bool) -> List[Dict]:
        """Get portfolio recommendations based on investor profile"""
        recommendations = []
        
        if risk_tolerance == 'conservative':
            recommendations.append({
                'name': 'Conservative Growth',
                'template': 'conservative',
                'score': 90,
                'reason': 'Matches conservative risk profile with capital preservation focus'
            })
            
            if income_focus:
                recommendations.append({
                    'name': 'Income Generation',
                    'template': 'income_focused',
                    'score': 85,
                    'reason': 'High income generation with lower volatility'
                })
        
        elif risk_tolerance == 'moderate':
            recommendations.append({
                'name': 'Balanced Growth',
                'template': 'balanced',
                'score': 95,
                'reason': 'Perfect match for moderate risk tolerance'
            })
            
            if time_horizon == 'long-term':
                recommendations.append({
                    'name': 'Endowment Model',
                    'template': 'institutional_endowment',
                    'score': 80,
                    'reason': 'Long-term institutional diversification approach'
                })
        
        elif risk_tolerance == 'aggressive':
            recommendations.append({
                'name': 'Growth Focused',
                'template': 'growth',
                'score': 95,
                'reason': 'Maximum growth potential for aggressive investors'
            })
            
            recommendations.append({
                'name': 'Endowment Model',
                'template': 'institutional_endowment',
                'score': 85,
                'reason': 'Diversified aggressive approach with alternatives'
            })
        
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)
    
    def get_optimization_methods(self) -> List[str]:
        """Get available optimization methods"""
        basic_methods = ['template_based', 'equal_weight']
        
        if self.advanced_features_available:
            advanced_methods = [
                'markowitz', 'black_litterman', 'risk_parity', 'ml_enhanced',
                'smart_beta', 'sector_rotation'
            ]
            return basic_methods + advanced_methods
        
        return basic_methods
    
    # =============================================================================
    # UTILITY METHODS
    # =============================================================================
    
    def _get_returns_matrix(self, symbols: List[str], period: str = '1y') -> pd.DataFrame:
        """Get aligned returns matrix for optimization"""
        returns_dict = {}

        for symbol in symbols:
            try:
                data = self.market_data.get_stock_data(symbol, period)
                if not data.empty and 'Daily_Return' in data.columns:
                    returns = data['Daily_Return'].dropna()
                    if len(returns) > 50:  # Minimum data requirement
                        # Ensure timezone-naive index
                        returns.index = returns.index.tz_localize(None)
                        returns_dict[symbol] = returns
            except Exception as e:
                self.logger.warning(f"Failed to get data for {symbol}: {e}")

        if not returns_dict:
            return pd.DataFrame()

        # Create aligned returns matrix
        returns_df = pd.DataFrame(returns_dict)
        returns_df.index = returns_df.index.tz_localize(None)  # enforce tz-naive
        returns_df = returns_df.dropna()

        return returns_df

    
    def _find_substitute(self, symbol: str, available_assets: List[str]) -> Optional[str]:
        """Find substitute for unavailable asset"""
        substitutes = {
            'VYM': 'SPY',      # High dividend -> Large cap
            'VXUS': 'VEA',     # Total intl -> Developed markets
            'BND': 'TLT',      # Total bond -> Long treasury
            'BNDX': 'TLT',     # Intl bonds -> US treasury
            'PDBC': 'GLD',     # Commodities -> Gold
            'VNQI': 'VNQ',     # Intl REITs -> US REITs
        }
        
        if symbol in substitutes and substitutes[symbol] in available_assets:
            return substitutes[symbol]
        
        # Generic substitutions by asset class
        if symbol.startswith('VT') or symbol in ['VXUS', 'EFA']:  # International equity
            for candidate in ['VEA', 'EFA', 'VXUS']:
                if candidate in available_assets:
                    return candidate
        
        if symbol in ['BND', 'AGG', 'BNDX']:  # Broad bonds
            for candidate in ['TLT', 'IEF', 'LQD']:
                if candidate in available_assets:
                    return candidate
        
        return None
    
    def _basic_optimization(self, symbols: List[str]) -> Dict:
        """Basic optimization fallback when advanced methods unavailable"""
        try:
            returns_data = self._get_returns_matrix(symbols)
            
            if returns_data.empty:
                # Equal weight fallback
                equal_weight = 1.0 / len(symbols)
                return {
                    'weights': {symbol: equal_weight for symbol in symbols},
                    'optimization_status': 'fallback_equal_weight',
                    'method': 'equal_weight'
                }
            
            # Simple mean-variance optimization
            mean_returns = returns_data.mean() * 252  # Annualized
            cov_matrix = returns_data.cov() * 252  # Annualized
            
            # Minimize variance for target return
            n_assets = len(symbols)
            
            def objective(weights):
                return np.sqrt(weights.T @ cov_matrix.values @ weights)
            
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            bounds = [(0.01, 0.5) for _ in range(n_assets)]  # 1% to 50% bounds
            
            # Initial guess
            x0 = np.array([1.0/n_assets] * n_assets)
            
            result = minimize(objective, x0, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                weights_dict = dict(zip(symbols, result.x))
                portfolio_return = mean_returns @ result.x
                portfolio_vol = objective(result.x)
                
                return {
                    'weights': weights_dict,
                    'expected_return': portfolio_return,
                    'expected_volatility': portfolio_vol,
                    'sharpe_ratio': portfolio_return / portfolio_vol if portfolio_vol > 0 else 0,
                    'optimization_status': 'optimal',
                    'method': 'basic_min_variance'
                }
            else:
                equal_weight = 1.0 / len(symbols)
                return {
                    'weights': {symbol: equal_weight for symbol in symbols},
                    'optimization_status': 'failed_fallback_equal',
                    'method': 'equal_weight'
                }
                
        except Exception as e:
            self.logger.error(f"Basic optimization failed: {e}")
            equal_weight = 1.0 / len(symbols)
            return {
                'weights': {symbol: equal_weight for symbol in symbols},
                'optimization_status': 'error_fallback',
                'method': 'equal_weight',
                'error': str(e)
            }
    
    def _fallback_portfolio(self) -> Dict[str, float]:
        """Fallback portfolio if template building fails"""
        return {
            'SPY': 0.60,   # S&P 500
            'TLT': 0.40    # Long-term Treasury
        }