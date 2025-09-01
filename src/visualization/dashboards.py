# src/visualization/dashboards.py

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List

class AdvancedDashboard:
    
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'positive': '#2ca02c', 
            'negative': '#d62728',
            'neutral': '#ff7f0e'
        }
    
    def portfolio_comparison_chart(self, portfolios: Dict[str, pd.DataFrame]) -> go.Figure:
        fig = go.Figure()
        
        for name, data in portfolios.items():
            if 'Cumulative_Return' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Cumulative_Return'],
                    mode='lines',
                    name=name,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title="Portfolio Performance Comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            template="plotly_white",
            height=500
        )
        
        return fig
    
    def risk_return_scatter(self, data: List[Dict]) -> go.Figure:
        if not data:
            return go.Figure()
        
        returns = [d.get('return', 0) for d in data]
        risks = [d.get('risk', 0) for d in data]
        names = [d.get('name', f'Asset {i}') for i, d in enumerate(data)]
        
        fig = px.scatter(
            x=risks, y=returns, text=names,
            labels={'x': 'Risk (Volatility)', 'y': 'Expected Return'},
            title="Risk-Return Profile"
        )
        
        fig.update_traces(textposition="top center")
        fig.update_layout(height=400)
        
        return fig
    
    def asset_correlation_matrix(self, returns_data: pd.DataFrame) -> go.Figure:
        corr = returns_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Asset Correlation Matrix",
            height=400
        )
        
        return fig
    
    def drawdown_chart(self, returns: pd.Series) -> go.Figure:
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            fill='tonexty',
            mode='lines',
            name='Drawdown',
            line=dict(color=self.colors['negative'])
        ))
        
        fig.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_white",
            height=300
        )
        
        return fig
    
    def sector_allocation_chart(self, allocations: Dict[str, float]) -> go.Figure:
        fig = px.sunburst(
            values=list(allocations.values()),
            names=list(allocations.keys()),
            title="Portfolio Allocation"
        )
        
        fig.update_layout(height=400)
        return fig
    
    def performance_metrics_table(self, metrics: Dict) -> None:
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Return",
                f"{metrics.get('total_return', 0)*100:.1f}%",
                delta=f"{metrics.get('excess_return', 0)*100:.1f}%" if 'excess_return' in metrics else None
            )
            st.metric(
                "Sharpe Ratio",
                f"{metrics.get('sharpe_ratio', 0):.2f}"
            )
        
        with col2:
            st.metric(
                "Max Drawdown",
                f"{metrics.get('max_drawdown', 0)*100:.1f}%"
            )
            st.metric(
                "Volatility",
                f"{metrics.get('volatility', 0)*100:.1f}%"
            )
        
        with col3:
            st.metric(
                "Win Rate",
                f"{metrics.get('win_rate', 0)*100:.0f}%"
            )
            st.metric(
                "Calmar Ratio",
                f"{metrics.get('calmar_ratio', 0):.2f}"
            )
    
    def ai_insights_card(self, insights: str) -> None:
        st.markdown("""
        <div style="
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            margin: 1rem 0;
        ">
            <h4 style="margin-top: 0; color: #1f77b4;">🤖 AI Market Insights</h4>
            <p style="margin-bottom: 0; line-height: 1.6;">{}</p>
        </div>
        """.format(insights), unsafe_allow_html=True)
    
    def create_gauge_chart(self, value: float, title: str, 
                          min_val: float = 0, max_val: float = 100) -> go.Figure:
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = value,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
            gauge = {
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': self.colors['primary']},
                'steps': [
                    {'range': [min_val, max_val*0.5], 'color': "lightgray"},
                    {'range': [max_val*0.5, max_val*0.8], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_val*0.9
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig

# Global instance
dashboard = AdvancedDashboard()