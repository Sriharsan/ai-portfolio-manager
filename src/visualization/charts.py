"""
Charts Module - Financial Visualization Components
Optimized charting functions using Plotly
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

class ChartGenerator:
    """Optimized financial chart generation"""
    
    def __init__(self):
        self.color_scheme = {
            'positive': '#00C851',
            'negative': '#FF4444', 
            'neutral': '#33b5e5',
            'background': '#f8f9fa'
        }
    
    def create_performance_chart(self, data: pd.DataFrame, title: str = "Portfolio Performance") -> go.Figure:
        """Create portfolio performance line chart"""
        
        fig = go.Figure()
        
        if 'Cumulative_Return' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Cumulative_Return'],
                mode='lines',
                name='Portfolio',
                line=dict(color=self.color_scheme['neutral'], width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def create_allocation_pie(self, weights: Dict[str, float]) -> go.Figure:
        """Create portfolio allocation pie chart"""
        
        fig = px.pie(
            values=list(weights.values()),
            names=list(weights.keys()),
            title="Portfolio Allocation"
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        return fig
    
    def create_risk_metrics_bar(self, metrics: Dict) -> go.Figure:
        """Create risk metrics bar chart"""
        
        risk_data = {
            'Metric': ['VaR (1-day)', 'CVaR (1-day)', 'Max Drawdown', 'Volatility'],
            'Value': [
                metrics.get('var_1d_95', 0) / 1000,  # Convert to thousands
                metrics.get('cvar_1d_95', 0) / 1000,
                metrics.get('maximum_drawdown', 0) * 100,  # Convert to percentage
                metrics.get('volatility_annual', 0) * 100
            ]
        }
        
        fig = px.bar(
            x=risk_data['Metric'],
            y=risk_data['Value'],
            title="Risk Metrics Overview"
        )
        
        return fig
    
    def create_correlation_heatmap(self, returns_data: Dict[str, pd.Series]) -> go.Figure:
        """Create correlation heatmap"""
        
        # Create correlation matrix
        df = pd.DataFrame(returns_data)
        corr_matrix = df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdYlBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="Asset Correlation Matrix",
            height=400
        )
        
        return fig

# Global instance
chart_generator = ChartGenerator()