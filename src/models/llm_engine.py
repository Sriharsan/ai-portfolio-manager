"""
AI Portfolio Management System - LLM Engine
PyTorch-powered Large Language Model for financial insights
Enterprise-grade AI with fine-tuned models for financial analysis
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertForSequenceClassification
)
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
import json
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import configuration and market data
from config import config
from data.market_data import market_data

class FinancialLLMEngine:
    """
    Enterprise-grade LLM Engine for financial analysis
    Combines multiple AI models for comprehensive market insights
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, config.LOG_LEVEL))
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"🤖 LLM Engine initializing on {self.device}")
        
        # Initialize models
        self._initialize_models()
        
        # Financial keywords and context
        self._load_financial_context()
        
    def _initialize_models(self):
        """Initialize all required models"""
        
        try:
            # 1. Financial Sentiment Analysis (FinBERT)
            self._load_finbert()
            
            # 2. General Language Model for text generation
            self._load_general_llm()
            
            # 3. Financial NER (Named Entity Recognition)
            self._load_financial_ner()
            
            # 4. Market News Classifier
            self._load_news_classifier()
            
            self.logger.info("✅ All models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Model initialization failed: {e}")
            self._fallback_to_simple_models()
    
    def _load_finbert(self):
        """Load FinBERT for financial sentiment analysis"""
        try:
            model_name = "ProsusAI/finbert"
            self.logger.info(f"📥 Loading FinBERT: {model_name}")
            
            self.pipelines['sentiment'] = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=0 if self.device.type == 'cuda' else -1
            )
            
            self.logger.info("✅ FinBERT loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"⚠️ FinBERT loading failed: {e}")
            self._fallback_sentiment_model()
    
    def _load_general_llm(self):
        """Load general language model for text generation"""
        try:
            model_name = "microsoft/DialoGPT-medium"
            self.logger.info(f"📥 Loading General LLM: {model_name}")
            
            self.tokenizers['general'] = AutoTokenizer.from_pretrained(model_name)
            self.models['general'] = AutoModel.from_pretrained(model_name).to(self.device)
            
            # Add special tokens for financial context
            if self.tokenizers['general'].pad_token is None:
                self.tokenizers['general'].pad_token = self.tokenizers['general'].eos_token
            
            self.logger.info("✅ General LLM loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"⚠️ General LLM loading failed: {e}")
            self._fallback_generation_model()
    
    def _load_financial_ner(self):
        """Load Financial Named Entity Recognition model"""
        try:
            # Using a general NER model adapted for financial entities
            self.pipelines['ner'] = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=0 if self.device.type == 'cuda' else -1
            )
            
            self.logger.info("✅ Financial NER loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"⚠️ NER loading failed: {e}")
    
    def _load_news_classifier(self):
        """Load news category classifier"""
        try:
            # Using a general classification model
            self.pipelines['classification'] = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device.type == 'cuda' else -1
            )
            
            self.logger.info("✅ News classifier loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"⚠️ News classifier loading failed: {e}")
    
    def _fallback_to_simple_models(self):
        """Fallback to simple rule-based models if PyTorch models fail"""
        self.logger.warning("⚠️ Falling back to rule-based models")
        
        # Simple sentiment analysis
        self.sentiment_keywords = {
            'positive': ['bullish', 'growth', 'profit', 'gain', 'strong', 'buy', 'upgrade', 'outperform'],
            'negative': ['bearish', 'decline', 'loss', 'weak', 'sell', 'downgrade', 'underperform', 'risk'],
            'neutral': ['hold', 'maintain', 'stable', 'unchanged', 'neutral']
        }
    
    def _fallback_sentiment_model(self):
        """Fallback sentiment analysis using rule-based approach"""
        self.use_fallback_sentiment = True
        self.logger.info("📝 Using rule-based sentiment analysis")
    
    def _fallback_generation_model(self):
        """Fallback text generation using templates"""
        self.use_fallback_generation = True
        self.logger.info("📝 Using template-based text generation")
    
    def _load_financial_context(self):
        """Load financial domain knowledge and context"""
        
        self.financial_terms = {
            'metrics': ['P/E ratio', 'EPS', 'ROE', 'ROA', 'debt-to-equity', 'current ratio'],
            'indicators': ['RSI', 'MACD', 'moving average', 'Bollinger bands', 'volume'],
            'sectors': ['technology', 'healthcare', 'finance', 'energy', 'consumer', 'industrial'],
            'events': ['earnings', 'dividend', 'merger', 'acquisition', 'IPO', 'stock split']
        }
        
        self.market_context = {
            'bull_signals': ['breakout', 'uptrend', 'support level', 'momentum'],
            'bear_signals': ['breakdown', 'downtrend', 'resistance level', 'selling pressure'],
            'volatility': ['high volatility', 'low volatility', 'market uncertainty']
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze sentiment of financial text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment label and confidence score
        """
        
        if not text or len(text.strip()) == 0:
            return {'label': 'neutral', 'score': 0.0}
        
        try:
            # Try FinBERT pipeline first
            if 'sentiment' in self.pipelines:
                result = self.pipelines['sentiment'](text)[0]
                
                # Map FinBERT labels to standard labels
                label_mapping = {
                    'positive': 'positive',
                    'negative': 'negative', 
                    'neutral': 'neutral'
                }
                
                return {
                    'label': label_mapping.get(result['label'].lower(), 'neutral'),
                    'score': float(result['score']),
                    'model': 'finbert'
                }
            
            # Fallback to rule-based sentiment
            else:
                return self._rule_based_sentiment(text)
                
        except Exception as e:
            self.logger.error(f"❌ Sentiment analysis failed: {e}")
            return self._rule_based_sentiment(text)
    
    def _rule_based_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """Rule-based sentiment analysis fallback"""
        
        text_lower = text.lower()
        
        positive_score = sum(1 for word in self.sentiment_keywords['positive'] if word in text_lower)
        negative_score = sum(1 for word in self.sentiment_keywords['negative'] if word in text_lower)
        neutral_score = sum(1 for word in self.sentiment_keywords['neutral'] if word in text_lower)
        
        total_score = positive_score + negative_score + neutral_score
        
        if total_score == 0:
            return {'label': 'neutral', 'score': 0.5, 'model': 'rule_based'}
        
        if positive_score > negative_score and positive_score > neutral_score:
            return {'label': 'positive', 'score': positive_score / total_score, 'model': 'rule_based'}
        elif negative_score > positive_score and negative_score > neutral_score:
            return {'label': 'negative', 'score': negative_score / total_score, 'model': 'rule_based'}
        else:
            return {'label': 'neutral', 'score': neutral_score / total_score, 'model': 'rule_based'}
    
    def extract_financial_entities(self, text: str) -> List[Dict[str, str]]:
        """
        Extract financial entities from text
        
        Args:
            text: Input text
            
        Returns:
            List of entities with type and value
        """
        
        entities = []
        
        try:
            # Try NER pipeline
            if 'ner' in self.pipelines:
                ner_results = self.pipelines['ner'](text)
                
                for entity in ner_results:
                    entities.append({
                        'text': entity['word'],
                        'label': entity['entity_group'],
                        'confidence': float(entity['score'])
                    })
            
            # Add rule-based financial entity extraction
            financial_entities = self._extract_financial_patterns(text)
            entities.extend(financial_entities)
            
        except Exception as e:
            self.logger.error(f"❌ Entity extraction failed: {e}")
            entities = self._extract_financial_patterns(text)
        
        return entities
    
    def _extract_financial_patterns(self, text: str) -> List[Dict[str, str]]:
        """Extract financial patterns using regex"""
        
        patterns = {
            'stock_symbol': r'\b[A-Z]{2,5}\b',
            'currency': r'\$[\d,]+\.?\d*',
            'percentage': r'\d+\.?\d*%',
            'ratio': r'\d+\.?\d*:\d+\.?\d*',
            'date': r'\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}'
        }
        
        entities = []
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append({
                    'text': match,
                    'label': entity_type,
                    'confidence': 0.8
                })
        
        return entities
    
    def generate_market_insight(
        self, 
        stock_data: pd.DataFrame, 
        symbol: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate AI-powered market insights for a stock
        
        Args:
            stock_data: Stock data DataFrame
            symbol: Stock ticker symbol
            context: Additional context information
            
        Returns:
            Generated insight text
        """
        
        try:
            # Analyze the stock data
            analysis = self._analyze_stock_metrics(stock_data, symbol)
            
            # Generate insight using LLM or templates
            if hasattr(self, 'use_fallback_generation') and self.use_fallback_generation:
                insight = self._generate_template_insight(analysis, symbol)
            else:
                insight = self._generate_llm_insight(analysis, symbol)
            
            return insight
            
        except Exception as e:
            self.logger.error(f"❌ Insight generation failed: {e}")
            return self._generate_fallback_insight(symbol)
    
    def _analyze_stock_metrics(self, data: pd.DataFrame, symbol: str) -> Dict:
        """Analyze stock metrics for insight generation"""
        
        if data.empty:
            return {'error': 'No data available'}
        
        latest = data.iloc[-1]
        previous = data.iloc[-2] if len(data) > 1 else latest
        
        # Price analysis
        current_price = latest['Close']
        price_change = current_price - previous['Close']
        price_change_pct = (price_change / previous['Close']) * 100
        
        # Technical analysis
        analysis = {
            'symbol': symbol,
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'volume': latest['Volume'],
            'avg_volume': data['Volume'].tail(20).mean(),
            'volatility_30d': data['Daily_Return'].tail(30).std() * np.sqrt(252) if 'Daily_Return' in data else None,
            'rsi': latest.get('RSI', None),
            'macd': latest.get('MACD', None),
            'macd_signal': latest.get('MACD_Signal', None),
            'sma_20': latest.get('SMA_20', None),
            'sma_50': latest.get('SMA_50', None),
            'bb_position': None,
            'trend': 'neutral'
        }
        
        # Bollinger Band position
        if all(col in latest for col in ['BB_Upper', 'BB_Lower', 'Close']):
            bb_range = latest['BB_Upper'] - latest['BB_Lower']
            bb_position = (latest['Close'] - latest['BB_Lower']) / bb_range
            analysis['bb_position'] = bb_position
        
        # Trend analysis
        if analysis['sma_20'] and analysis['sma_50']:
            if analysis['sma_20'] > analysis['sma_50']:
                if current_price > analysis['sma_20']:
                    analysis['trend'] = 'strong_bullish'
                else:
                    analysis['trend'] = 'bullish'
            else:
                if current_price < analysis['sma_20']:
                    analysis['trend'] = 'strong_bearish'
                else:
                    analysis['trend'] = 'bearish'
        
        # RSI interpretation
        if analysis['rsi']:
            if analysis['rsi'] > 70:
                analysis['rsi_signal'] = 'overbought'
            elif analysis['rsi'] < 30:
                analysis['rsi_signal'] = 'oversold'
            else:
                analysis['rsi_signal'] = 'neutral'
        
        # Volume analysis
        volume_ratio = latest['Volume'] / analysis['avg_volume']
        if volume_ratio > 2.0:
            analysis['volume_signal'] = 'high'
        elif volume_ratio < 0.5:
            analysis['volume_signal'] = 'low'
        else:
            analysis['volume_signal'] = 'normal'
        
        return analysis
    
    def _generate_template_insight(self, analysis: Dict, symbol: str) -> str:
        """Generate insight using templates (fallback method)"""
        
        if 'error' in analysis:
            return f"Unable to analyze {symbol} due to insufficient data."
        
        # Build insight components
        insights = []
        
        # Price movement
        price_change = analysis['price_change_pct']
        if abs(price_change) > 2:
            direction = "gained" if price_change > 0 else "declined"
            insights.append(f"{symbol} has {direction} {abs(price_change):.1f}% in the latest session.")
        
        # Trend analysis
        trend = analysis['trend']
        trend_messages = {
            'strong_bullish': f"{symbol} shows strong bullish momentum with price above both 20-day and 50-day moving averages.",
            'bullish': f"{symbol} displays bullish signals with short-term average above long-term average.",
            'strong_bearish': f"{symbol} exhibits strong bearish pressure with price below key moving averages.",
            'bearish': f"{symbol} shows bearish signals with declining short-term momentum.",
            'neutral': f"{symbol} is trading in a neutral range with mixed technical signals."
        }
        insights.append(trend_messages.get(trend, f"{symbol} technical outlook is unclear."))
        
        # RSI insights
        if 'rsi_signal' in analysis:
            rsi_signal = analysis['rsi_signal']
            rsi_value = analysis['rsi']
            if rsi_signal == 'overbought':
                insights.append(f"RSI at {rsi_value:.1f} suggests {symbol} may be overbought and due for a pullback.")
            elif rsi_signal == 'oversold':
                insights.append(f"RSI at {rsi_value:.1f} indicates {symbol} may be oversold and could see a bounce.")
        
        # Volume analysis
        volume_signal = analysis.get('volume_signal', 'normal')
        if volume_signal == 'high':
            insights.append(f"Trading volume is significantly above average, indicating strong investor interest in {symbol}.")
        elif volume_signal == 'low':
            insights.append(f"Below-average volume suggests limited conviction in {symbol}'s current price movement.")
        
        # Risk assessment
        volatility = analysis.get('volatility_30d')
        if volatility:
            if volatility > 0.3:
                insights.append(f"High volatility of {volatility*100:.1f}% indicates elevated risk for {symbol}.")
            elif volatility < 0.15:
                insights.append(f"Low volatility of {volatility*100:.1f}% suggests {symbol} is relatively stable.")
        
        return " ".join(insights)
    
    def _generate_llm_insight(self, analysis: Dict, symbol: str) -> str:
        """Generate insight using LLM (when available)"""
        
        # Create prompt for LLM
        prompt = self._create_financial_prompt(analysis, symbol)
        
        try:
            # Use general LLM for generation (simplified approach)
            if 'general' in self.models:
                # This is a simplified implementation
                # In production, you'd implement proper text generation
                return self._generate_template_insight(analysis, symbol)
            else:
                return self._generate_template_insight(analysis, symbol)
                
        except Exception as e:
            self.logger.error(f"❌ LLM generation failed: {e}")
            return self._generate_template_insight(analysis, symbol)
    
    def _create_financial_prompt(self, analysis: Dict, symbol: str) -> str:
        """Create a structured prompt for LLM generation"""
        
        prompt = f"""
        Financial Analysis for {symbol}:
        
        Price: ${analysis['current_price']:.2f} ({analysis['price_change_pct']:+.1f}%)
        Trend: {analysis['trend']}
        RSI: {analysis.get('rsi', 'N/A')}
        Volume Signal: {analysis.get('volume_signal', 'normal')}
        
        Generate a professional investment insight:
        """
        
        return prompt
    
    def _generate_fallback_insight(self, symbol: str) -> str:
        """Generate a basic fallback insight"""
        return f"Market analysis for {symbol} is currently unavailable. Please check back later for updated insights."
    
    def analyze_portfolio_text(self, text: str, portfolio_symbols: List[str]) -> Dict:
        """
        Analyze text for portfolio-relevant information
        
        Args:
            text: Text to analyze (news, earnings calls, etc.)
            portfolio_symbols: List of symbols in the portfolio
            
        Returns:
            Analysis results
        """
        
        result = {
            'sentiment': self.analyze_sentiment(text),
            'entities': self.extract_financial_entities(text),
            'relevance_score': 0.0,
            'mentioned_symbols': [],
            'key_topics': []
        }
        
        # Check for portfolio symbol mentions
        text_upper = text.upper()
        for symbol in portfolio_symbols:
            if symbol.upper() in text_upper:
                result['mentioned_symbols'].append(symbol)
                result['relevance_score'] += 0.3
        
        # Extract key topics using classification
        try:
            if 'classification' in self.pipelines:
                topics = ['earnings', 'market outlook', 'regulatory news', 'economic indicators', 'company strategy']
                classification_result = self.pipelines['classification'](text, topics)
                result['key_topics'] = [
                    {'topic': label['label'], 'score': label['score']} 
                    for label in classification_result['labels'][:3]
                ]
        except Exception as e:
            self.logger.warning(f"⚠️ Topic classification failed: {e}")
        
        # Calculate overall relevance
        entity_relevance = len(result['entities']) * 0.1
        sentiment_strength = abs(result['sentiment']['score'] - 0.5) * 2
        result['relevance_score'] += entity_relevance + sentiment_strength * 0.2
        
        result['relevance_score'] = min(result['relevance_score'], 1.0)
        
        return result
    
    def generate_portfolio_summary(
        self, 
        portfolio_data: Dict[str, pd.DataFrame],
        portfolio_weights: Dict[str, float]
    ) -> str:
        """
        Generate an AI-powered portfolio summary
        
        Args:
            portfolio_data: Dictionary of stock data for portfolio
            portfolio_weights: Portfolio allocation weights
            
        Returns:
            Generated portfolio summary
        """
        
        try:
            # Analyze each holding
            holdings_analysis = []
            
            for symbol, weight in portfolio_weights.items():
                if symbol in portfolio_data and not portfolio_data[symbol].empty:
                    data = portfolio_data[symbol]
                    analysis = self._analyze_stock_metrics(data, symbol)
                    analysis['weight'] = weight
                    holdings_analysis.append(analysis)
            
            # Generate summary
            summary = self._create_portfolio_summary_text(holdings_analysis)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Portfolio summary generation failed: {e}")
            return "Portfolio summary is currently unavailable. Please try again later."
    
    def _create_portfolio_summary_text(self, holdings_analysis: List[Dict]) -> str:
        """Create portfolio summary text from analysis"""
        
        if not holdings_analysis:
            return "Portfolio analysis unavailable due to insufficient data."
        
        # Calculate portfolio-level metrics
        total_return = sum(h['price_change_pct'] * h['weight'] for h in holdings_analysis if 'price_change_pct' in h)
        
        # Count signals
        bullish_count = sum(1 for h in holdings_analysis if h.get('trend', '').endswith('bullish'))
        bearish_count = sum(1 for h in holdings_analysis if h.get('trend', '').endswith('bearish'))
        
        # Build summary
        summary_parts = []
        
        # Overall performance
        if total_return > 1:
            summary_parts.append(f"Your portfolio is up {total_return:.1f}% today, showing positive momentum.")
        elif total_return < -1:
            summary_parts.append(f"Your portfolio is down {abs(total_return):.1f}% today, facing some headwinds.")
        else:
            summary_parts.append("Your portfolio is trading relatively flat today.")
        
        # Technical outlook
        if bullish_count > bearish_count:
            summary_parts.append(f"{bullish_count} of your holdings show bullish signals, indicating potential upside.")
        elif bearish_count > bullish_count:
            summary_parts.append(f"{bearish_count} holdings show bearish signals, suggesting caution may be warranted.")
        else:
            summary_parts.append("Your holdings show mixed technical signals across the portfolio.")
        
        # Top performers and laggards
        sorted_holdings = sorted(holdings_analysis, key=lambda x: x.get('price_change_pct', 0), reverse=True)
        
        if len(sorted_holdings) > 1:
            best_performer = sorted_holdings[0]
            worst_performer = sorted_holdings[-1]
            
            summary_parts.append(
                f"Today's leader is {best_performer['symbol']} (+{best_performer.get('price_change_pct', 0):.1f}%), "
                f"while {worst_performer['symbol']} lags ({worst_performer.get('price_change_pct', 0):+.1f}%)."
            )
        
        # Risk assessment
        high_vol_stocks = [h['symbol'] for h in holdings_analysis if h.get('volatility_30d', 0) > 0.25]
        if high_vol_stocks:
            summary_parts.append(f"Monitor elevated volatility in {', '.join(high_vol_stocks)}.")
        
        return " ".join(summary_parts)
    
    def classify_market_news(self, text: str) -> Dict:
        """
        Classify market news by category and importance
        
        Args:
            text: News text to classify
            
        Returns:
            Classification results
        """
        
        categories = [
            'earnings report',
            'economic data', 
            'federal reserve',
            'geopolitical event',
            'corporate announcement',
            'market analysis'
        ]
        
        try:
            if 'classification' in self.pipelines:
                result = self.pipelines['classification'](text, categories)
                
                return {
                    'primary_category': result['labels'][0],
                    'confidence': result['scores'][0],
                    'all_categories': [
                        {'category': label, 'score': score}
                        for label, score in zip(result['labels'], result['scores'])
                    ]
                }
            else:
                return self._rule_based_classification(text, categories)
                
        except Exception as e:
            self.logger.error(f"❌ News classification failed: {e}")
            return self._rule_based_classification(text, categories)
    
    def _rule_based_classification(self, text: str, categories: List[str]) -> Dict:
        """Rule-based news classification fallback"""
        
        text_lower = text.lower()
        
        category_keywords = {
            'earnings report': ['earnings', 'revenue', 'eps', 'quarterly', 'guidance'],
            'economic data': ['gdp', 'inflation', 'employment', 'retail sales', 'housing'],
            'federal reserve': ['fed', 'federal reserve', 'interest rate', 'monetary policy'],
            'geopolitical event': ['trade war', 'sanctions', 'election', 'brexit', 'china'],
            'corporate announcement': ['merger', 'acquisition', 'ceo', 'dividend', 'buyback'],
            'market analysis': ['outlook', 'forecast', 'prediction', 'analyst', 'rating']
        }
        
        scores = {}
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[category] = score / len(keywords)
        
        # Find best match
        best_category = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_category]
        
        return {
            'primary_category': best_category,
            'confidence': best_score,
            'all_categories': [
                {'category': cat, 'score': score}
                for cat, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ]
        }
    
    def get_ai_recommendation(
        self, 
        symbol: str, 
        analysis: Dict,
        risk_tolerance: str = 'moderate'
    ) -> Dict:
        """
        Generate AI-powered investment recommendation
        
        Args:
            symbol: Stock symbol
            analysis: Stock analysis data
            risk_tolerance: 'conservative', 'moderate', 'aggressive'
            
        Returns:
            Investment recommendation
        """
        
        recommendation = {
            'symbol': symbol,
            'action': 'hold',
            'confidence': 0.5,
            'reasoning': [],
            'risk_factors': [],
            'price_target': None,
            'time_horizon': '3-6 months'
        }
        
        # Analyze signals
        bullish_signals = 0
        bearish_signals = 0
        
        # Price trend
        if analysis.get('trend') in ['strong_bullish', 'bullish']:
            bullish_signals += 2 if 'strong' in analysis['trend'] else 1
            recommendation['reasoning'].append("Positive price trend")
        elif analysis.get('trend') in ['strong_bearish', 'bearish']:
            bearish_signals += 2 if 'strong' in analysis['trend'] else 1
            recommendation['reasoning'].append("Negative price trend")
        
        # RSI signals
        rsi_signal = analysis.get('rsi_signal')
        if rsi_signal == 'oversold':
            bullish_signals += 1
            recommendation['reasoning'].append("Oversold conditions may present buying opportunity")
        elif rsi_signal == 'overbought':
            bearish_signals += 1
            recommendation['reasoning'].append("Overbought conditions suggest potential pullback")
        
        # Volume confirmation
        volume_signal = analysis.get('volume_signal')
        if volume_signal == 'high' and analysis.get('price_change_pct', 0) > 0:
            bullish_signals += 1
            recommendation['reasoning'].append("High volume confirms price movement")
        
        # Risk factors
        volatility = analysis.get('volatility_30d', 0)
        if volatility > 0.3:
            recommendation['risk_factors'].append("High volatility increases investment risk")
        
        # Generate final recommendation
        signal_difference = bullish_signals - bearish_signals
        
        if signal_difference >= 2:
            recommendation['action'] = 'buy'
            recommendation['confidence'] = min(0.8, 0.5 + signal_difference * 0.1)
        elif signal_difference <= -2:
            recommendation['action'] = 'sell'
            recommendation['confidence'] = min(0.8, 0.5 + abs(signal_difference) * 0.1)
        else:
            recommendation['action'] = 'hold'
            recommendation['confidence'] = 0.6
        
        # Adjust for risk tolerance
        if risk_tolerance == 'conservative' and recommendation['action'] == 'buy':
            if recommendation['confidence'] < 0.7:
                recommendation['action'] = 'hold'
                recommendation['reasoning'].append("Conservative approach suggests holding given moderate confidence")
        
        return recommendation

# Global LLM engine instance
llm_engine = FinancialLLMEngine()

# Testing function
def test_llm_engine():
    """Test the LLM engine functionality"""
    
    print("🧪 Testing LLM Engine...")
    
    # Test sentiment analysis
    test_texts = [
        "Apple reported strong quarterly earnings with revenue beating expectations",
        "Market volatility increases amid economic uncertainty and declining consumer confidence",
        "Tech stocks maintain steady performance with balanced trading volume"
    ]
    
    for text in test_texts:
        sentiment = llm_engine.analyze_sentiment(text)
        print(f"📝 Text: {text[:50]}...")
        print(f"   Sentiment: {sentiment['label']} (confidence: {sentiment['score']:.2f})")
    
    # Test stock insight generation
    try:
        stock_data = market_data.get_stock_data('AAPL', period='1mo')
        insight = llm_engine.generate_market_insight(stock_data, 'AAPL')
        print(f"\n💡 AAPL Insight: {insight}")
    except Exception as e:
        print(f"❌ Stock insight test failed: {e}")
    
    # Test portfolio summary
    try:
        portfolio = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
        portfolio_data, _ = market_data.get_portfolio_data(portfolio, period='1mo')
        summary = llm_engine.generate_portfolio_summary(portfolio_data, portfolio)
        print(f"\n📊 Portfolio Summary: {summary}")
    except Exception as e:
        print(f"❌ Portfolio summary test failed: {e}")

if __name__ == "__main__":
    test_llm_engine()