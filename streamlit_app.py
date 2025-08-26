"""
AI Portfolio Management System - Optimized Streamlit App
Memory-efficient dashboard with real AI integration
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import config
from data.data_loader import data_loader
from visualization.charts import chart_generator

# Page config
st.set_page_config(
    page_title="AI Portfolio Manager",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {'AAPL': 0.3, 'MSFT': 0.3, 'GOOGL': 0.4}

def main():
    st.title("AI Portfolio Management System")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox("Select Page:", [
            "Dashboard", "Portfolio Analysis", "AI Insights", "Risk Management"
        ])
        
        # Portfolio input
        st.header("Portfolio Settings")
        symbols = st.text_input("Symbols (comma-separated):", "AAPL,MSFT,GOOGL").split(',')
        symbols = [s.strip().upper() for s in symbols if s.strip()]
        
        # Equal weights by default
        weights = [1.0/len(symbols) for _ in symbols]
        st.session_state.portfolio = dict(zip(symbols, weights))
        
        st.write(f"Portfolio: {st.session_state.portfolio}")
    
    # Main content
    if page == "Dashboard":
        show_dashboard()
    elif page == "Portfolio Analysis":
        show_portfolio_analysis()
    elif page == "AI Insights":
        show_ai_insights()
    elif page == "Risk Management":
        show_risk_management()

def show_dashboard():
    """Main dashboard with key metrics"""
    
    st.header("Portfolio Dashboard")
    
    # Get portfolio analysis
    with st.spinner("Loading portfolio data..."):
        analysis = data_loader.get_portfolio_analysis(st.session_state.portfolio, '6mo')
    
    if 'error' in analysis:
        st.error(f"Error loading data: {analysis['error']}")
        return
    
    # Key metrics
    metrics = analysis.get('performance_metrics', {})
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{metrics.get('total_return', 0)*100:.1f}%")
    with col2:
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    with col3:
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0)*100:.1f}%")
    with col4:
        st.metric("Volatility", f"{metrics.get('volatility', 0)*100:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'portfolio_data' in analysis:
            chart = chart_generator.create_performance_chart(analysis['portfolio_data'])
            st.plotly_chart(chart, use_container_width=True)
    
    with col2:
        chart = chart_generator.create_allocation_pie(st.session_state.portfolio)
        st.plotly_chart(chart, use_container_width=True)
    
    # AI Insights
    st.subheader("AI Market Analysis")
    if 'ai_insights' in analysis:
        st.write(analysis['ai_insights'])
    else:
        st.info("AI insights loading...")

def show_portfolio_analysis():
    """Portfolio optimization and analysis"""
    
    st.header("Portfolio Analysis")
    
    analysis = data_loader.get_portfolio_analysis(st.session_state.portfolio, '1y')
    
    if 'error' in analysis:
        st.error(f"Analysis unavailable: {analysis['error']}")
        return
    
    # Optimization results
    if 'optimization' in analysis and analysis['optimization']:
        opt = analysis['optimization']
        
        st.subheader("Optimization Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Current Allocation:")
            for symbol, weight in st.session_state.portfolio.items():
                st.write(f"{symbol}: {weight*100:.1f}%")
        
        with col2:
            st.write("Suggested Allocation:")
            for symbol, weight in opt['weights'].items():
                st.write(f"{symbol}: {weight*100:.1f}%")
        
        st.metric("Expected Return", f"{opt['expected_return']*100:.1f}%")
        st.metric("Expected Volatility", f"{opt['volatility']*100:.1f}%")

def show_ai_insights():
    """AI-powered market insights"""
    
    st.header("AI Market Insights")
    
    # Individual stock analysis
    st.subheader("Stock Analysis")
    
    selected_symbol = st.selectbox("Select Stock:", list(st.session_state.portfolio.keys()))
    
    if st.button("Analyze"):
        with st.spinner(f"Analyzing {selected_symbol}..."):
            stock_analysis = data_loader.get_stock_analysis(selected_symbol)
        
        if 'error' not in stock_analysis:
            st.write(f"**Current Price:** ${stock_analysis['current_price']:.2f}")
            st.write(f"**Daily Change:** {stock_analysis['price_change']:.2f}%")
            
            st.subheader("AI Insight")
            st.write(stock_analysis['ai_insight'])

def show_risk_management():
    """Risk management dashboard"""
    
    st.header("Risk Management")
    
    analysis = data_loader.get_portfolio_analysis(st.session_state.portfolio, '6mo')
    
    if 'error' in analysis:
        st.error("Risk analysis unavailable")
        return
    
    # Risk metrics
    risk_metrics = analysis.get('risk_metrics', {})
    
    if 'error' not in risk_metrics:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("VaR (1-day, 95%)", f"${risk_metrics.get('var_1d_95', 0):,.0f}")
            st.metric("CVaR (1-day, 95%)", f"${risk_metrics.get('cvar_1d_95', 0):,.0f}")
        
        with col2:
            st.metric("Annual Volatility", f"{risk_metrics.get('volatility_annual', 0)*100:.1f}%")
            st.metric("Max Drawdown", f"{risk_metrics.get('maximum_drawdown', 0)*100:.1f}%")

if __name__ == "__main__":
    main()