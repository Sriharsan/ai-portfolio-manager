"""
AI Portfolio Management System - Streamlit App
Fixed import issues and error handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import logging

# Configure logging first
logging.basicConfig(level=logging.INFO)

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Set page config
st.set_page_config(
    page_title="AI Portfolio Manager",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {'AAPL': 0.3, 'MSFT': 0.3, 'GOOGL': 0.4}

def safe_import():
    """Safely import modules with error handling"""
    try:
        from config import config
        from data.data_loader import data_loader
        from visualization.charts import chart_generator
        return config, data_loader, chart_generator
    except Exception as e:
        st.error(f"Import error: {str(e)}")
        st.info("Please ensure all dependencies are installed: `pip install -r requirements.txt`")
        return None, None, None

def main():
    """Main application"""
    
    # Safe imports
    config, data_loader, chart_generator = safe_import()
    
    if not all([config, data_loader, chart_generator]):
        st.stop()
    
    st.title("🤖 AI Portfolio Management System")
    st.markdown("*Democratizing institutional-grade investment analytics*")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Portfolio Configuration")
        
        # Portfolio input
        symbols_input = st.text_input(
            "Enter symbols (comma-separated):", 
            value="AAPL,MSFT,GOOGL",
            help="Enter valid US stock tickers"
        )
        
        if symbols_input:
            symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
            
            if symbols:
                # Equal weighting
                weight = 1.0 / len(symbols)
                st.session_state.portfolio = {symbol: weight for symbol in symbols}
                
                st.write("**Current Portfolio:**")
                for symbol, weight in st.session_state.portfolio.items():
                    st.write(f"• {symbol}: {weight*100:.1f}%")
        
        # Analysis period
        period = st.selectbox(
            "Analysis Period:",
            options=['1mo', '3mo', '6mo', '1y'],
            index=2
        )
        
        # Action buttons
        analyze_button = st.button("🔄 Update Analysis", type="primary")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Dashboard", 
        "⚡ Portfolio Analysis", 
        "🤖 AI Insights", 
        "⚠️ Risk Management"
    ])
    
    with tab1:
        show_dashboard(data_loader, chart_generator, period)
    
    with tab2:
        show_portfolio_analysis(data_loader, period)
    
    with tab3:
        show_ai_insights(data_loader)
    
    with tab4:
        show_risk_management(data_loader, period)

def show_dashboard(data_loader, chart_generator, period):
    """Portfolio dashboard"""
    
    st.subheader("Portfolio Dashboard")
    
    if not st.session_state.portfolio:
        st.warning("Please configure your portfolio in the sidebar.")
        return
    
    # Load data with error handling
    with st.spinner("Loading portfolio data..."):
        try:
            analysis = data_loader.get_portfolio_analysis(st.session_state.portfolio, period)
        except Exception as e:
            st.error(f"Data loading failed: {str(e)}")
            st.info("This might be due to API rate limits or network issues. Please try again.")
            return
    
    if 'error' in analysis:
        st.error(f"Analysis failed: {analysis['error']}")
        return
    
    # Key metrics
    metrics = analysis.get('performance_metrics', {})
    
    if metrics and 'error' not in metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_return = metrics.get('total_return', 0) * 100
            st.metric("Total Return", f"{total_return:.1f}%")
        
        with col2:
            sharpe = metrics.get('sharpe_ratio', 0)
            st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        with col3:
            drawdown = metrics.get('max_drawdown', 0) * 100
            st.metric("Max Drawdown", f"{drawdown:.1f}%")
        
        with col4:
            volatility = metrics.get('volatility', 0) * 100
            st.metric("Volatility", f"{volatility:.1f}%")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'portfolio_data' in analysis and not analysis['portfolio_data'].empty:
            try:
                chart = chart_generator.create_performance_chart(
                    analysis['portfolio_data'], 
                    "Portfolio Performance"
                )
                st.plotly_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"Chart generation failed: {str(e)}")
        else:
            st.info("Portfolio performance chart unavailable")
    
    with col2:
        try:
            pie_chart = chart_generator.create_allocation_pie(st.session_state.portfolio)
            st.plotly_chart(pie_chart, use_container_width=True)
        except Exception as e:
            st.error(f"Allocation chart failed: {str(e)}")
    
    # AI Insights
    st.subheader("🤖 AI Market Analysis")
    if 'ai_insights' in analysis and analysis['ai_insights']:
        st.info(analysis['ai_insights'])
    else:
        st.write("AI analysis is processing your portfolio...")

def show_portfolio_analysis(data_loader, period):
    """Portfolio optimization analysis"""
    
    st.subheader("Portfolio Analysis & Optimization")
    
    if not st.session_state.portfolio:
        st.warning("Please configure your portfolio first.")
        return
    
    try:
        analysis = data_loader.get_portfolio_analysis(st.session_state.portfolio, period)
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return
    
    if 'error' in analysis:
        st.error(analysis['error'])
        return
    
    # Current allocation
    st.write("**Current Allocation:**")
    allocation_df = pd.DataFrame([
        {'Symbol': symbol, 'Weight': f"{weight*100:.1f}%", 'Allocation': weight}
        for symbol, weight in st.session_state.portfolio.items()
    ])
    st.dataframe(allocation_df, hide_index=True)
    
    # Optimization results
    if 'optimization' in analysis and analysis['optimization']:
        opt = analysis['optimization']
        
        if 'weights' in opt:
            st.write("**Optimized Allocation:**")
            opt_df = pd.DataFrame([
                {'Symbol': symbol, 'Current': f"{st.session_state.portfolio.get(symbol, 0)*100:.1f}%", 
                 'Suggested': f"{weight*100:.1f}%"}
                for symbol, weight in opt['weights'].items()
            ])
            st.dataframe(opt_df, hide_index=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Expected Return", f"{opt.get('expected_return', 0)*100:.1f}%")
            with col2:
                st.metric("Expected Risk", f"{opt.get('volatility', 0)*100:.1f}%")
    else:
        st.info("Optimization analysis unavailable - may require more data.")

def show_ai_insights(data_loader):
    """AI-powered insights"""
    
    st.subheader("AI Market Insights")
    
    if not st.session_state.portfolio:
        st.warning("Please configure your portfolio first.")
        return
    
    # Stock selector
    selected_stock = st.selectbox("Select stock for detailed analysis:", 
                                 list(st.session_state.portfolio.keys()))
    
    if st.button("Generate AI Analysis"):
        with st.spinner(f"Analyzing {selected_stock}..."):
            try:
                stock_analysis = data_loader.get_stock_analysis(selected_stock, '6mo')
                
                if 'error' not in stock_analysis:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Current Price", f"${stock_analysis.get('current_price', 0):.2f}")
                    
                    with col2:
                        change = stock_analysis.get('price_change', 0)
                        st.metric("Daily Change", f"{change:.2f}%", delta=f"{change:.2f}%")
                    
                    st.subheader("AI Analysis")
                    st.write(stock_analysis.get('ai_insight', 'Analysis unavailable'))
                else:
                    st.error(f"Analysis failed: {stock_analysis['error']}")
                    
            except Exception as e:
                st.error(f"AI analysis failed: {str(e)}")

def show_risk_management(data_loader, period):
    """Risk management dashboard"""
    
    st.subheader("Risk Management")
    
    if not st.session_state.portfolio:
        st.warning("Please configure your portfolio first.")
        return
    
    try:
        analysis = data_loader.get_portfolio_analysis(st.session_state.portfolio, period)
    except Exception as e:
        st.error(f"Risk analysis failed: {str(e)}")
        return
    
    if 'error' in analysis:
        st.error(analysis['error'])
        return
    
    risk_metrics = analysis.get('risk_metrics', {})
    
    if risk_metrics and 'error' not in risk_metrics:
        col1, col2 = st.columns(2)
        
        with col1:
            var_value = risk_metrics.get('var_1d_95', 0)
            st.metric("VaR (1-day, 95%)", f"${var_value:,.0f}")
            
            cvar_value = risk_metrics.get('cvar_1d_95', 0)
            st.metric("CVaR (1-day, 95%)", f"${cvar_value:,.0f}")
        
        with col2:
            vol = risk_metrics.get('volatility_annual', 0) * 100
            st.metric("Annual Volatility", f"{vol:.1f}%")
            
            drawdown = risk_metrics.get('maximum_drawdown', 0) * 100
            st.metric("Max Drawdown", f"{drawdown:.1f}%")
        
        # Risk interpretation
        st.subheader("Risk Assessment")
        if var_value < -5000:
            st.warning("⚠️ High risk portfolio - consider diversification")
        elif var_value < -2000:
            st.info("📊 Moderate risk level")
        else:
            st.success("✅ Conservative risk profile")
    else:
        st.info("Risk metrics unavailable - need more historical data")

if __name__ == "__main__":
    main()