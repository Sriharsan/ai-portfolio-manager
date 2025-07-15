"""
AI Portfolio Management System - Main Streamlit Application
Enterprise-grade portfolio management with PyTorch LLMs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import configuration
from config import config

# Page configuration
st.set_page_config(
    page_title=config.STREAMLIT_CONFIG["page_title"],
    page_icon=config.STREAMLIT_CONFIG["page_icon"],
    layout=config.STREAMLIT_CONFIG["layout"],
    initial_sidebar_state=config.STREAMLIT_CONFIG["initial_sidebar_state"]
)

def main():
    """
    Main application function
    """
    
    # Header
    st.title("🚀 AI Portfolio Management System")
    st.markdown("*Enterprise-grade portfolio management with PyTorch LLMs*")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Navigation")
        
        # Navigation menu
        page = st.selectbox(
            "Choose a page:",
            [
                "🏠 Dashboard",
                "📈 Portfolio Analysis", 
                "🤖 AI Insights",
                "⚖️ Risk Management",
                "📊 Performance Analytics",
                "⚙️ Settings"
            ]
        )
        
        # Configuration status
        st.header("🔧 System Status")
        validation = config.validate_config()
        
        if validation["warnings"]:
            st.warning(f"⚠️ {len(validation['warnings'])} warnings")
            with st.expander("View warnings"):
                for warning in validation["warnings"]:
                    st.write(f"• {warning}")
        else:
            st.success("✅ All systems operational")
    
    # Main content area
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📈 Portfolio Analysis":
        show_portfolio_analysis()
    elif page == "🤖 AI Insights":
        show_ai_insights()
    elif page == "⚖️ Risk Management":
        show_risk_management()
    elif page == "📊 Performance Analytics":
        show_performance_analytics()
    elif page == "⚙️ Settings":
        show_settings()

def show_dashboard():
    """
    Main dashboard page
    """
    st.header("📊 Portfolio Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Portfolio Value",
            value="$1,234,567",
            delta="$12,345 (1.2%)"
        )
    
    with col2:
        st.metric(
            label="Total Return",
            value="15.3%",
            delta="2.1% vs benchmark"
        )
    
    with col3:
        st.metric(
            label="Sharpe Ratio",
            value="1.45",
            delta="0.12"
        )
    
    with col4:
        st.metric(
            label="Max Drawdown",
            value="-8.2%",
            delta="-1.1%"
        )
    
    # Portfolio allocation chart
    st.subheader("🥧 Portfolio Allocation")
    
    # Sample data
    allocation_data = {
        'Asset': ['Technology', 'Healthcare', 'Financial', 'Energy', 'Consumer', 'Industrial'],
        'Allocation': [25, 20, 15, 10, 15, 15],
        'Value': [312500, 250000, 187500, 125000, 187500, 187500]
    }
    
    df_allocation = pd.DataFrame(allocation_data)
    
    # Create pie chart
    fig_pie = px.pie(
        df_allocation, 
        values='Allocation', 
        names='Asset',
        title="Asset Allocation",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Performance chart
    st.subheader("📈 Performance Overview")
    
    # Generate sample performance data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))
    cumulative_returns = (1 + returns).cumprod()
    
    performance_df = pd.DataFrame({
        'Date': dates,
        'Portfolio': cumulative_returns,
        'Benchmark': (1 + np.random.normal(0.0008, 0.015, len(dates))).cumprod()
    })
    
    fig_performance = go.Figure()
    
    fig_performance.add_trace(go.Scatter(
        x=performance_df['Date'],
        y=performance_df['Portfolio'],
        mode='lines',
        name='Portfolio',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig_performance.add_trace(go.Scatter(
        x=performance_df['Date'],
        y=performance_df['Benchmark'],
        mode='lines',
        name='Benchmark',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    fig_performance.update_layout(
        title="Portfolio vs Benchmark Performance",
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_performance, use_container_width=True)

def show_portfolio_analysis():
    """
    Portfolio analysis page
    """
    st.header("📈 Portfolio Analysis")
    
    st.info("🚧 Portfolio analysis features coming soon! This will include:")
    st.write("""
    - **Modern Portfolio Theory** optimization
    - **Risk-Return** analysis
    - **Correlation** matrices
    - **Asset allocation** recommendations
    - **Rebalancing** suggestions
    """)
    
    # Placeholder for portfolio analysis
    st.subheader("🎯 Optimization Results")
    st.write("Portfolio optimization engine will be integrated here.")

def show_ai_insights():
    """
    AI insights page with LLM integration
    """
    st.header("🤖 AI-Powered Market Insights")
    
    st.info("🚧 AI insights powered by PyTorch LLMs coming soon!")
    
    # Placeholder for AI insights
    st.subheader("💡 Market Intelligence")
    st.write("This section will feature:")
    st.write("""
    - **Sentiment Analysis** of market news
    - **Earnings Call** summaries
    - **Economic Indicator** interpretation
    - **Investment** recommendations
    - **Risk Assessment** insights
    """)
    
    # Sample AI insight
    st.subheader("📰 Latest Market Sentiment")
    st.write("*AI-generated insight based on recent market data:*")
    st.markdown("""
    > **Market Outlook:** Current sentiment analysis indicates cautious optimism 
    > in the technology sector, with strong earnings reports offsetting concerns 
    > about regulatory changes. Recommendation: Maintain current tech allocation 
    > while monitoring policy developments.
    """)

def show_risk_management():
    """
    Risk management page
    """
    st.header("⚖️ Risk Management")
    
    st.info("🚧 Advanced risk management features coming soon!")
    
    # Risk metrics placeholder
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Risk Metrics")
        st.write("""
        - **Value at Risk (VaR):** $45,000 (1-day, 95%)
        - **Conditional VaR:** $67,500
        - **Beta:** 1.15
        - **Standard Deviation:** 16.3%
        """)
    
    with col2:
        st.subheader("🚨 Risk Alerts")
        st.warning("⚠️ High correlation detected between Tech holdings")
        st.info("ℹ️ Portfolio beta above target range")

def show_performance_analytics():
    """
    Performance analytics page
    """
    st.header("📊 Performance Analytics")
    
    st.info("🚧 Advanced performance analytics coming soon!")
    
    # Performance metrics
    st.subheader("📈 Key Performance Indicators")
    
    metrics_data = {
        'Metric': ['Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio', 'Max Drawdown'],
        'Portfolio': ['15.3%', '12.8%', '16.1%', '1.45', '-8.2%'],
        'Benchmark': ['11.2%', '9.4%', '14.8%', '1.12', '-12.1%']
    }
    
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

def show_settings():
    """
    Settings page
    """
    st.header("⚙️ System Settings")
    
    # Configuration display
    st.subheader("🔧 Current Configuration")
    
    # Show configuration status
    validation = config.validate_config()
    
    if validation["warnings"]:
        st.warning("⚠️ Configuration Issues Detected")
        for warning in validation["warnings"]:
            st.write(f"• {warning}")
    
    # Configuration form
    st.subheader("🔑 API Configuration")
    
    with st.form("api_config"):
        st.write("Configure your API keys:")
        
        alpha_vantage_key = st.text_input(
            "Alpha Vantage API Key",
            type="password",
            help="Get your free key at https://www.alphavantage.co/support/#api-key"
        )
        
        fred_key = st.text_input(
            "FRED API Key", 
            type="password",
            help="Get your free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
        
        hf_key = st.text_input(
            "Hugging Face API Key",
            type="password", 
            help="Get your free key at https://huggingface.co/settings/tokens"
        )
        
        submitted = st.form_submit_button("Save Configuration")
        
        if submitted:
            st.success("✅ Configuration saved! Please restart the application.")
    
    # System information
    st.subheader("ℹ️ System Information")
    st.write(f"**Application:** {config.APP_NAME}")
    st.write(f"**Version:** {config.APP_VERSION}")
    st.write(f"**Debug Mode:** {config.DEBUG}")
    st.write(f"**Log Level:** {config.LOG_LEVEL}")

if __name__ == "__main__":
    main()
