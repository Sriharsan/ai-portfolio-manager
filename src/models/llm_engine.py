# src/models/llm_engine.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import re
from pathlib import Path

# ✅ Add OpenAI import
import openai


class OptimizedLLMEngine:
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._setup_models()
        self._load_financial_context()
    
    def _setup_models(self):
        try:
            # Try to load transformers for sentiment
            from transformers import pipeline
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1  # CPU only for memory efficiency
            )
            self.use_transformers = True
            self.use_openai = False
            self.logger.info("✅ FinBERT loaded")
        except Exception as e:
            self.logger.warning(f"FinBERT failed, trying OpenAI: {e}")
            self.use_transformers = False
            try:
                from config import config
                openai.api_key = config.OPENAI_API_KEY
                self.use_openai = True
                self.logger.info("✅ OpenAI GPT initialized")
            except Exception as e2:
                self.logger.warning(f"OpenAI not available, using rule-based: {e2}")
                self.use_openai = False
                self._setup_rule_based_models()
    
    def _setup_rule_based_models(self):
        self.positive_words = {
            'bullish', 'growth', 'profit', 'gain', 'strong', 'buy', 'upgrade',
            'beat', 'exceed', 'rally', 'surge', 'momentum', 'breakout'
        }
        self.negative_words = {
            'bearish', 'decline', 'loss', 'weak', 'sell', 'downgrade',
            'miss', 'crash', 'fall', 'pressure', 'breakdown', 'drop'
        }
    
    def _load_financial_context(self):
        self.market_signals = {
            'bull': ['uptrend', 'breakout', 'support', 'momentum'],
            'bear': ['downtrend', 'breakdown', 'resistance', 'selling'],
            'neutral': ['sideways', 'consolidation', 'range-bound']
        }
    
    def analyze_sentiment(self, text: str) -> Dict:
        if not text:
            return {'label': 'neutral', 'score': 0.5}
        
        try:
            if self.use_transformers:
                result = self.sentiment_pipeline(text[:512])[0]  # Limit text length
                return {
                    'label': result['label'].lower(),
                    'score': result['score'],
                    'model': 'finbert'
                }
            elif getattr(self, "use_openai", False):
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": f"Classify the financial sentiment of this text: {text}"}]
                )
                label = response['choices'][0]['message']['content'].strip().lower()
                return {'label': label, 'score': 0.7, 'model': 'openai'}
            else:
                return self._rule_based_sentiment(text)
        except Exception:
            return self._rule_based_sentiment(text)
    
    def _rule_based_sentiment(self, text: str) -> Dict:
        text_lower = text.lower()
        pos_count = sum(1 for word in self.positive_words if word in text_lower)
        neg_count = sum(1 for word in self.negative_words if word in text_lower)
        
        total = pos_count + neg_count
        if total == 0:
            return {'label': 'neutral', 'score': 0.5, 'model': 'rule_based'}
        
        pos_ratio = pos_count / total
        if pos_ratio > 0.6:
            return {'label': 'positive', 'score': 0.5 + pos_ratio/2, 'model': 'rule_based'}
        elif pos_ratio < 0.4:
            return {'label': 'negative', 'score': 0.5 - (1-pos_ratio)/2, 'model': 'rule_based'}
        else:
            return {'label': 'neutral', 'score': 0.5, 'model': 'rule_based'}
    
    def generate_market_insight(self, stock_data: pd.DataFrame, symbol: str) -> str:
        try:
            analysis = self._analyze_stock_metrics(stock_data, symbol)

            # ✅ Use OpenAI if available
            if getattr(self, "use_openai", False):
                prompt = f"Generate a professional financial market insight for {symbol} given this analysis: {analysis}"
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response['choices'][0]['message']['content']

            return self._create_insight_text(analysis, symbol)
        except Exception as e:
            self.logger.error(f"Insight generation failed: {e}")
            return f"Market analysis for {symbol} is currently unavailable."
    
    def _analyze_stock_metrics(self, data: pd.DataFrame, symbol: str) -> Dict:
        if data.empty or len(data) < 2:
            return {'error': 'Insufficient data'}
        
        latest = data.iloc[-1]
        previous = data.iloc[-2]
        
        price_change = (latest['Close'] - previous['Close']) / previous['Close'] * 100
        
        # Simple trend analysis
        if len(data) >= 20:
            sma_20 = data['Close'].tail(20).mean()
            trend = 'bullish' if latest['Close'] > sma_20 else 'bearish'
        else:
            trend = 'neutral'
        
        # Volume analysis
        avg_volume = data['Volume'].tail(10).mean()
        volume_signal = 'high' if latest['Volume'] > avg_volume * 1.5 else 'normal'
        
        return {
            'symbol': symbol,
            'price_change': price_change,
            'current_price': latest['Close'],
            'trend': trend,
            'volume_signal': volume_signal,
            'rsi': latest.get('RSI', 50)
        }
    
    def _create_insight_text(self, analysis: Dict, symbol: str) -> str:
        if 'error' in analysis:
            return f"Unable to analyze {symbol} - insufficient data."
        
        price_change = analysis['price_change']
        trend = analysis['trend']
        
        # Price movement
        direction = "gained" if price_change > 0 else "declined"
        insight = f"{symbol} has {direction} {abs(price_change):.1f}% recently. "
        
        # Trend analysis
        if trend == 'bullish':
            insight += "Technical indicators suggest bullish momentum. "
        elif trend == 'bearish':
            insight += "Technical indicators show bearish pressure. "
        
        # Volume confirmation
        if analysis['volume_signal'] == 'high':
            insight += "Above-average trading volume confirms the move."
        
        # RSI insight
        rsi = analysis.get('rsi', 50)
        if rsi > 70:
            insight += " RSI indicates overbought conditions."
        elif rsi < 30:
            insight += " RSI suggests oversold levels."
        
        return insight.strip()
    
    def generate_portfolio_summary(self, portfolio_data: Dict, weights: Dict) -> str:
        try:
            holdings_perf = []
            
            for symbol, weight in weights.items():
                if symbol in portfolio_data:
                    data = portfolio_data[symbol]
                    if not data.empty:
                        change = data['Daily_Return'].iloc[-1] if 'Daily_Return' in data.columns else 0
                        holdings_perf.append({'symbol': symbol, 'return': change, 'weight': weight})
            
            if not holdings_perf:
                return "Portfolio analysis unavailable due to insufficient data."
            
            # Calculate weighted return
            portfolio_return = sum(h['return'] * h['weight'] for h in holdings_perf)
            
            # ✅ Use OpenAI if available
            if getattr(self, "use_openai", False):
                prompt = f"Summarize this portfolio performance with top and bottom contributors: {holdings_perf}, portfolio return={portfolio_return}"
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response['choices'][0]['message']['content']

            # Create summary
            if portfolio_return > 0.01:
                summary = f"Portfolio is up {portfolio_return*100:.1f}% today with positive momentum. "
            elif portfolio_return < -0.01:
                summary = f"Portfolio is down {abs(portfolio_return)*100:.1f}% today facing headwinds. "
            else:
                summary = "Portfolio is trading relatively flat today. "
            
            # Top performer
            if holdings_perf:
                best = max(holdings_perf, key=lambda x: x['return'])
                worst = min(holdings_perf, key=lambda x: x['return'])
                summary += f"{best['symbol']} leads (+{best['return']*100:.1f}%) "
                summary += f"while {worst['symbol']} lags ({worst['return']*100:+.1f}%)."
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Portfolio summary failed: {e}")
            return "Portfolio summary temporarily unavailable."
    
    def extract_financial_entities(self, text: str) -> List[Dict]:
        entities = []
        
        patterns = {
            'stock_symbol': r'\b[A-Z]{2,5}\b',
            'currency': r'\$[\d,]+\.?\d*',
            'percentage': r'\d+\.?\d*%'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append({
                    'text': match,
                    'label': entity_type,
                    'confidence': 0.8
                })
        
        return entities


# Global instance
llm_engine = OptimizedLLMEngine()
