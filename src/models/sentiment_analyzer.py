# src/models/sentiment_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict, List, Union
import re
import logging

class FinancialSentimentAnalyzer:
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._load_sentiment_lexicons()
    
    def _load_sentiment_lexicons(self):
        
        self.positive_words = {
            'bullish', 'growth', 'profit', 'gain', 'strong', 'buy', 'upgrade', 
            'outperform', 'beat', 'exceed', 'optimistic', 'rally', 'surge',
            'momentum', 'breakout', 'bull', 'rise', 'upside', 'positive'
        }
        
        self.negative_words = {
            'bearish', 'decline', 'loss', 'weak', 'sell', 'downgrade', 
            'underperform', 'miss', 'below', 'pessimistic', 'crash', 'fall',
            'pressure', 'breakdown', 'bear', 'drop', 'downside', 'negative'
        }
        
        # Financial intensifiers
        self.intensifiers = {
            'very', 'extremely', 'highly', 'significantly', 'dramatically',
            'substantially', 'considerably', 'strongly', 'deeply'
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        
        if not text or len(text.strip()) == 0:
            return {'label': 'neutral', 'score': 0.5, 'confidence': 0.0}
        
        # Clean and tokenize text
        words = self._preprocess_text(text)
        
        # Calculate sentiment scores
        positive_score = self._calculate_sentiment_score(words, self.positive_words)
        negative_score = self._calculate_sentiment_score(words, self.negative_words)
        
        # Determine overall sentiment
        total_score = positive_score + negative_score
        
        if total_score == 0:
            return {'label': 'neutral', 'score': 0.5, 'confidence': 0.3}
        
        positive_ratio = positive_score / total_score
        
        if positive_ratio > 0.6:
            label = 'positive'
            score = 0.5 + (positive_ratio - 0.5)
        elif positive_ratio < 0.4:
            label = 'negative' 
            score = 0.5 - (0.5 - positive_ratio)
        else:
            label = 'neutral'
            score = 0.5
        
        confidence = min(1.0, total_score / 10.0)  # Normalize confidence
        
        return {
            'label': label,
            'score': score,
            'confidence': confidence,
            'word_count': len(words),
            'positive_words': positive_score,
            'negative_words': negative_score
        }
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Clean and tokenize text"""
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Remove common stop words (basic list)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        return words
    
    def _calculate_sentiment_score(self, words: List[str], sentiment_words: set) -> float:
        
        score = 0
        for i, word in enumerate(words):
            if word in sentiment_words:
                base_score = 1
                
                # Check for intensifiers
                if i > 0 and words[i-1] in self.intensifiers:
                    base_score *= 1.5
                
                # Check for negations
                if i > 0 and words[i-1] in {'not', 'no', 'never', 'none'}:
                    base_score *= -1
                
                score += base_score
        
        return score
    
    def batch_analyze(self, texts: List[str]) -> List[Dict]:
        
        return [self.analyze_sentiment(text) for text in texts]
    
    def market_sentiment_score(self, news_texts: List[str]) -> Dict:
        
        if not news_texts:
            return {'overall_sentiment': 'neutral', 'confidence': 0.0}
        
        sentiments = self.batch_analyze(news_texts)
        
        # Weight sentiments by confidence
        weighted_scores = []
        total_confidence = 0
        
        for sentiment in sentiments:
            if sentiment['confidence'] > 0.3:  # Only use confident predictions
                weighted_scores.append(sentiment['score'] * sentiment['confidence'])
                total_confidence += sentiment['confidence']
        
        if total_confidence == 0:
            return {'overall_sentiment': 'neutral', 'confidence': 0.0}
        
        average_score = sum(weighted_scores) / total_confidence
        
        if average_score > 0.6:
            label = 'positive'
        elif average_score < 0.4:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'overall_sentiment': label,
            'score': average_score,
            'confidence': total_confidence / len(sentiments),
            'articles_analyzed': len(news_texts)
        }

# Global instance
sentiment_analyzer = FinancialSentimentAnalyzer()