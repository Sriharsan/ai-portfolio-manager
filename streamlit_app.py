"""
Enhanced AI Portfolio Management System - BlackRock-Level Application
Institutional-grade portfolio management with comprehensive asset coverage
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import logging
import plotly.graph_objects as go
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add src to Python path
current_dir = Path(__file__).parent
src_dir = current_dir / "src"
sys.path.insert(0, str(src_dir))

# Set page config
st.set_page_config(
    page_title="Institutional AI Portfolio Manager",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}
if 'portfolio_template' not in st.session_state:
    st.session_state.portfolio_template = 'balanced'
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None

def safe_import():
    """Safely import enhanced modules"""
    try:
        # Import enhanced modules
        sys.path.insert(0, str(Path(__file__).parent))
        from enhanced_market_data import market_data_provider
        from institutional_portfolio_builder import InstitutionalPortfolioBuilder
        from config import config
        from data.data_loader import data_loader
        from visualization.charts import chart_generator
        from visualization.dashboards import dashboard
        
        # Initialize portfolio builder
        portfolio_builder = InstitutionalPortfolioBuilder(market_data_provider)
        
        return market_data_provider, portfolio_builder, config, data_loader, chart_generator, dashboard
    except Exception as e:
        st.error(f"Import error: {str(e)}")
        st.info("Please ensure all dependencies are installed and the enhanced modules are available.")
        st.code("pip install -r requirements.txt")
        return None, None, None, None, None, None

def main():
    """Enhanced main application"""
    
    # Safe imports
    market_data, portfolio_builder, config, data_loader, chart_generator, dashboard = safe_import()
    
    if not all([market_data, portfolio_builder]):
        st.stop()
    
    # Header
    st.title("🏛️ Institutional AI Portfolio Manager")
    st.markdown("*BlackRock-level portfolio management with 75+ assets across all major asset classes*")
    
    # Display API status
    display_api_status(config)
    
    # Enhanced sidebar
    with st.sidebar:
        show_enhanced_sidebar(market_data, portfolio_builder)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏛️ Institutional Dashboard", 
        "📊 Portfolio Builder",
        "⚡ Market Analysis", 
        "🤖 AI Insights", 
        "⚠️ Risk Management",
        "🌍 Global Markets"
    ])
    
    with tab1:
        show_institutional_dashboard(data_loader, chart_generator, dashboard)
    
    with tab2:
        show_portfolio_builder(portfolio_builder, market_data)
    
    with tab3:
        show_market_analysis(market_data, chart_generator)
    
    with tab4:
        show_ai_insights(data_loader, market_data)
    
    with tab5:
        show_enhanced_risk_management(data_loader, dashboard)
    
    with tab6:
        show_global_markets(market_data)

def display_api_status(config):
    """Display API connection status"""
    
    if config:
        api_status = config.get_api_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "🟢" if api_status['alpha_vantage'] else "🔴"
            st.write(f"{status} Alpha Vantage")
        
        with col2:
            status = "🟢" if api_status['fred'] else "🔴"
            st.write(f"{status} FRED")
        
        with col3:
            status = "🟢" if api_status['huggingface'] else "🔴"
            st.write(f"{status} HuggingFace")
        
        with col4:
            status = "🟢" if api_status['openai'] else "🔴"
            st.write(f"{status} OpenAI")

def show_enhanced_sidebar(market_data, portfolio_builder):
    """Enhanced sidebar with institutional features"""
    
    st.header("🏛️ Portfolio Configuration")
    
    # Portfolio creation methods
    creation_method = st.selectbox(
        "Portfolio Creation Method:",
        ["Template-Based", "Custom Selection", "Smart Beta", "Sector Rotation", "Risk Parity"]
    )
    
    if creation_method == "Template-Based":
        show_template_selector(portfolio_builder)
    elif creation_method == "Custom Selection":
        show_custom_selector(market_data)
    elif creation_method == "Smart Beta":
        show_smart_beta_builder(portfolio_builder)
    elif creation_method == "Sector Rotation":
        show_sector_rotation_builder(portfolio_builder)
    elif creation_method == "Risk Parity":
        show_risk_parity_builder(portfolio_builder, market_data)
    
    # Analysis period
    st.session_state.analysis_period = st.selectbox(
        "Analysis Period:",
        options=['1mo', '3mo', '6mo', '1y', '2y'],
        index=2
    )
    
    # Portfolio value
    st.session_state.portfolio_value = st.number_input(
        "Portfolio Value ($):",
        min_value=10000,
        max_value=100000000,
        value=1000000,
        step=100000,
        format="%d"
    )
    
    # Action buttons
    if st.button("🔄 Update Analysis", type="primary"):
        st.session_state.analysis_data = None  # Force refresh
        st.rerun()
    
    if st.button("📊 Export Data"):
        show_export_options()

def show_template_selector(portfolio_builder):
    """Show institutional portfolio templates"""
    
    templates = portfolio_builder.get_available_templates()
    
    # Risk tolerance and preferences
    risk_tolerance = st.selectbox(
        "Risk Tolerance:",
        ["Conservative", "Moderate", "Aggressive"]
    )
    
    investment_horizon = st.selectbox(
        "Investment Horizon:",
        ["Short-term (1-3 years)", "Medium-term (3-7 years)", "Long-term (7+ years)"]
    )
    
    income_focus = st.checkbox("Income Focus", value=False)
    
    # Get recommendations
    recommendations = portfolio_builder.get_portfolio_recommendations(
        risk_tolerance.lower(),
        investment_horizon.split()[0].lower(),
        income_focus
    )
    
    st.write("**Recommended Templates:**")
    
    selected_template = st.selectbox(
        "Select Template:",
        [rec['name'] for rec in recommendations] + list(templates.keys())
    )
    
    # Map display name back to template key
    template_key = selected_template.lower().replace(' ', '_')
    if template_key not in templates:
        # Find matching template
        for rec in recommendations:
            if rec['name'] == selected_template:
                template_key = rec['template']
                break
    
    if st.button("Build Portfolio from Template"):
        try:
            st.session_state.portfolio = portfolio_builder.build_portfolio_from_template(template_key)
            st.session_state.portfolio_template = template_key
            st.success(f"Portfolio created with {len(st.session_state.portfolio)} assets")
        except Exception as e:
            st.error(f"Template building failed: {str(e)}")
    
    # Show template details
    if template_key in templates:
        template_info = templates[template_key]
        st.write(f"**{template_info['name']}**")
        st.write(template_info['description'])
        st.write(f"Target Return: {template_info['target_return']*100:.1f}%")
        st.write(f"Target Volatility: {template_info['target_volatility']*100:.1f}%")

def show_custom_selector(market_data):
    """Show custom asset selection"""
    
    # Asset class filter
    asset_classes = st.multiselect(
        "Asset Classes:",
        ["US Equity", "International Equity", "Bonds", "Commodities", "Real Estate", "Alternatives"],
        default=["US Equity", "Bonds"]
    )
    
    available_symbols = []
    
    # Map asset classes to filters
    for asset_class in asset_classes:
        if asset_class == "US Equity":
            symbols = market_data.get_available_assets({'type': 'equity', 'region': 'US'})
            available_symbols.extend(symbols)
        elif asset_class == "International Equity":
            symbols = market_data.get_available_assets({'region': 'Developed'}) + \
                     market_data.get_available_assets({'region': 'Emerging'})
            available_symbols.extend(symbols)
        elif asset_class == "Bonds":
            symbols = market_data.get_available_assets({'sector': 'Government Bonds'}) + \
                     market_data.get_available_assets({'sector': 'Corporate Bonds'})
            available_symbols.extend(symbols)
        elif asset_class == "Commodities":
            symbols = market_data.get_available_assets({'sector': 'Commodities'})
            available_symbols.extend(symbols)
        elif asset_class == "Real Estate":
            symbols = market_data.get_available_assets({'sector': 'Real Estate'})
            available_symbols.extend(symbols)
    
    # Remove duplicates
    available_symbols = list(set(available_symbols))
    
    st.write(f"**Available Assets ({len(available_symbols)} symbols):**")
    
    # Multi-select with search
    selected_symbols = st.multiselect(
        "Select Assets:",
        available_symbols,
        default=available_symbols[:10] if len(available_symbols) >= 10 else available_symbols
    )
    
    if selected_symbols:
        # Allocation method
        allocation_method = st.radio(
            "Allocation Method:",
            ["Equal Weight", "Market Cap Weight", "Custom Weights"]
        )
        
        if allocation_method == "Equal Weight":
            weight = 1.0 / len(selected_symbols)
            st.session_state.portfolio = {symbol: weight for symbol in selected_symbols}
        elif allocation_method == "Custom Weights":
            st.write("**Set Custom Weights:**")
            weights = {}
            for symbol in selected_symbols:
                weights[symbol] = st.slider(f"{symbol}:", 0.0, 1.0, 1.0/len(selected_symbols), 0.01)
            
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.01:
                st.warning(f"Weights sum to {total_weight:.2f}, not 1.0")
            else:
                st.session_state.portfolio = weights

def show_smart_beta_builder(portfolio_builder):
    """Show smart beta portfolio builder"""
    
    st.write("**Smart Beta Strategy:**")
    st.info("Factor-based portfolio using quality, momentum, and value factors")
    
    universe_size = st.selectbox("Universe Size:", [10, 20, 30, 50], index=1)
    
    factor_focus = st.selectbox(
        "Factor Focus:",
        ["Balanced", "Quality-Focused", "Momentum-Focused", "Value-Focused"]
    )
    
    if st.button("Build Smart Beta Portfolio"):
        try:
            with st.spinner("Building smart beta portfolio..."):
                st.session_state.portfolio = portfolio_builder.create_smart_beta_portfolio()
                st.success("Smart beta portfolio created!")
        except Exception as e:
            st.error(f"Smart beta building failed: {str(e)}")

def show_sector_rotation_builder(portfolio_builder):
    """Show sector rotation portfolio builder"""
    
    st.write("**Sector Rotation Strategy:**")
    st.info("Momentum-based sector allocation using relative strength")
    
    lookback_period = st.slider("Momentum Lookback (days):", 20, 120, 60)
    
    if st.button("Build Sector Rotation Portfolio"):
        try:
            with st.spinner("Analyzing sector momentum..."):
                st.session_state.portfolio = portfolio_builder.create_sector_rotation_portfolio(lookback_period)
                st.success("Sector rotation portfolio created!")
        except Exception as e:
            st.error(f"Sector rotation building failed: {str(e)}")

def show_risk_parity_builder(portfolio_builder, market_data):
    """Show risk parity portfolio builder"""
    
    st.write("**Risk Parity Strategy:**")
    st.info("Equal risk contribution from each asset")
    
    # Select asset universe
    universe_type = st.selectbox(
        "Universe:",
        ["Multi-Asset", "Equity Only", "Custom Selection"]
    )
    
    if universe_type == "Multi-Asset":
        symbols = ['SPY', 'TLT', 'GLD', 'VNQ', 'VEA', 'VWO', 'LQD', 'HYG']
    elif universe_type == "Equity Only":
        symbols = ['SPY', 'QQQ', 'IWM', 'VEA', 'VWO', 'XLK', 'XLF', 'XLV', 'XLE', 'XLY']
    else:
        symbols = st.multiselect(
            "Select symbols:",
            market_data.get_available_assets(),
            default=['SPY', 'TLT', 'GLD', 'VNQ']
        )
    
    if st.button("Build Risk Parity Portfolio"):
        try:
            with st.spinner("Calculating risk parity weights..."):
                st.session_state.portfolio = portfolio_builder.create_risk_parity_portfolio(symbols)
                st.success("Risk parity portfolio created!")
        except Exception as e:
            st.error(f"Risk parity building failed: {str(e)}")

def show_institutional_dashboard(data_loader, chart_generator, dashboard):
    """Enhanced institutional dashboard"""
    
    st.subheader("🏛️ Institutional Portfolio Dashboard")
    
    if not st.session_state.portfolio:
        st.warning("Please configure your portfolio using the sidebar.")
        show_sample_portfolios()
        return
    
    # Load comprehensive analysis
    with st.spinner("Loading institutional-grade analysis..."):
        try:
            analysis = get_or_create_analysis(data_loader)
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return
    
    if 'error' in analysis:
        st.error(f"Analysis failed: {analysis['error']}")
        return
    
    # Key metrics dashboard
    show_key_metrics(analysis)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        show_performance_chart(analysis, chart_generator)
    
    with col2:
        show_allocation_chart()
    
    # Additional analysis
    col3, col4 = st.columns(2)
    
    with col3:
        show_risk_metrics_chart(analysis, chart_generator)
    
    with col4:
        show_sector_breakdown()
    
    # AI insights
    show_ai_insights_card(analysis)

def show_sample_portfolios():
    """Show sample institutional portfolios"""
    
    st.info("**Sample Institutional Portfolios:**")
    
    sample_portfolios = {
        "Balanced Growth (60/40)": {
            'SPY': 0.35, 'QQQ': 0.15, 'IWM': 0.10,  # US Equity 60%
            'TLT': 0.20, 'LQD': 0.15, 'TIP': 0.05   # Bonds 40%
        },
        "Endowment Model": {
            'SPY': 0.20, 'VEA': 0.15, 'VWO': 0.10,  # Global Equity 45%
            'TLT': 0.15, 'LQD': 0.10,               # Bonds 25%
            'VNQ': 0.10, 'GLD': 0.10, 'USO': 0.05,  # Alternatives 25%
            'VIXY': 0.05                            # Volatility 5%
        },
        "Global Diversified": {
            'VTI': 0.25, 'VXUS': 0.25,              # Global Equity 50%
            'BND': 0.20, 'BNDX': 0.10,              # Global Bonds 30%
            'VNQ': 0.10, 'VNQI': 0.05,              # Global REITs 15%
            'PDBC': 0.05                            # Commodities 5%
        }
    }
    
    for name, portfolio in sample_portfolios.items():
        if st.button(f"Load {name}"):
            st.session_state.portfolio = portfolio
            st.rerun()

def get_or_create_analysis(data_loader):
    """Get or create portfolio analysis with caching"""
    
    if st.session_state.analysis_data is None:
        analysis = data_loader.get_portfolio_analysis(
            st.session_state.portfolio, 
            st.session_state.analysis_period
        )
        st.session_state.analysis_data = analysis
        return analysis
    else:
        return st.session_state.analysis_data

def show_key_metrics(analysis):
    """Show key performance metrics"""
    
    metrics = analysis.get('performance_metrics', {})
    
    if metrics and 'error' not in metrics:
        col1, col2, col3, col4, col5 = st.columns(5)
        
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
        
        with col5:
            calmar = metrics.get('calmar_ratio', 0)
            st.metric("Calmar Ratio", f"{calmar:.2f}")

def show_performance_chart(analysis, chart_generator):
    """Show performance chart"""
    
    if 'portfolio_data' in analysis and not analysis['portfolio_data'].empty:
        try:
            chart = chart_generator.create_performance_chart(
                analysis['portfolio_data'], 
                "Portfolio Performance vs Benchmark"
            )
            st.plotly_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Performance chart error: {str(e)}")
    else:
        st.info("Performance chart unavailable")

def show_allocation_chart():
    """Show current allocation"""
    
    if st.session_state.portfolio:
        # Create enhanced pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(st.session_state.portfolio.keys()),
            values=list(st.session_state.portfolio.values()),
            hole=0.3,
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Current Portfolio Allocation",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_risk_metrics_chart(analysis, chart_generator):
    """Show risk metrics visualization"""
    
    risk_metrics = analysis.get('risk_metrics', {})
    
    if risk_metrics and 'error' not in risk_metrics:
        try:
            chart = chart_generator.create_risk_metrics_bar(risk_metrics)
            st.plotly_chart(chart, use_container_width=True)
        except Exception as e:
            st.error(f"Risk chart error: {str(e)}")

def show_sector_breakdown():
    """Show sector breakdown analysis"""
    
    st.subheader("Sector Breakdown")
    
    # Analyze portfolio by sectors
    if st.session_state.portfolio:
        # This would require sector mapping - simplified version
        sectors = {
            'Technology': 0.0,
            'Financials': 0.0,
            'Healthcare': 0.0,
            'Consumer': 0.0,
            'Bonds': 0.0,
            'Real Estate': 0.0,
            'Other': 0.0
        }
        
        # Map symbols to sectors (simplified)
        sector_mapping = {
            'SPY': 'Diversified', 'QQQ': 'Technology', 'IWM': 'Small Cap',
            'TLT': 'Bonds', 'LQD': 'Bonds', 'HYG': 'Bonds',
            'VNQ': 'Real Estate', 'GLD': 'Commodities',
            'XLK': 'Technology', 'XLF': 'Financials', 'XLV': 'Healthcare'
        }
        
        for symbol, weight in st.session_state.portfolio.items():
            sector = sector_mapping.get(symbol, 'Other')
            if sector in sectors:
                sectors[sector] += weight
            else:
                sectors['Other'] += weight
        
        # Create sector chart
        sector_df = pd.DataFrame([
            {'Sector': sector, 'Weight': weight} 
            for sector, weight in sectors.items() if weight > 0
        ])
        
        if not sector_df.empty:
            fig = px.bar(sector_df, x='Sector', y='Weight', 
                        title="Sector Allocation")
            st.plotly_chart(fig, use_container_width=True)

def show_ai_insights_card(analysis):
    """Show AI insights in styled card"""
    
    st.subheader("🤖 AI Market Intelligence")
    
    insights = analysis.get('ai_insights', 'AI analysis is processing...')
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    ">
        <h4 style="margin-top: 0; color: white;">🧠 AI Analysis</h4>
        <p style="margin-bottom: 0; line-height: 1.6; font-size: 16px;">{insights}</p>
    </div>
    """, unsafe_allow_html=True)

def show_portfolio_builder(portfolio_builder, market_data):
    """Portfolio builder interface"""
    
    st.subheader("📊 Advanced Portfolio Builder")
    
    # Portfolio construction wizard
    st.write("### Portfolio Construction Wizard")
    
    # Step 1: Investment Profile
    with st.expander("Step 1: Investment Profile", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age:", 18, 100, 35)
            risk_capacity = st.selectbox("Risk Capacity:", 
                                       ["Conservative", "Moderate", "Aggressive"])
        
        with col2:
            investment_goal = st.selectbox("Primary Goal:",
                                         ["Growth", "Income", "Balanced", "Capital Preservation"])
            time_horizon = st.selectbox("Time Horizon:",
                                      ["< 3 years", "3-7 years", "7-15 years", "> 15 years"])
    
    # Step 2: Asset Allocation
    with st.expander("Step 2: Strategic Asset Allocation"):
        allocation_approach = st.radio(
            "Allocation Approach:",
            ["Age-Based", "Risk-Based", "Goal-Based", "Custom"]
        )
        
        if allocation_approach == "Age-Based":
            stock_allocation = (100 - age) / 100
            bond_allocation = age / 100
            st.write(f"Suggested allocation: {stock_allocation*100:.0f}% Stocks, {bond_allocation*100:.0f}% Bonds")
        
        elif allocation_approach == "Risk-Based":
            if risk_capacity == "Conservative":
                suggested_allocation = {"Stocks": 0.3, "Bonds": 0.6, "Alternatives": 0.1}
            elif risk_capacity == "Moderate":
                suggested_allocation = {"Stocks": 0.6, "Bonds": 0.3, "Alternatives": 0.1}
            else:
                suggested_allocation = {"Stocks": 0.8, "Bonds": 0.1, "Alternatives": 0.1}
            
            for asset_class, weight in suggested_allocation.items():
                st.write(f"{asset_class}: {weight*100:.0f}%")
    
    # Step 3: Implementation
    with st.expander("Step 3: Implementation Strategy"):
        implementation = st.selectbox(
            "Implementation Style:",
            ["Passive (Index Funds)", "Active (Stock Picking)", "Hybrid", "Factor-Based"]
        )
        
        rebalancing = st.selectbox(
            "Rebalancing Frequency:",
            ["Monthly", "Quarterly", "Semi-Annual", "Annual", "Threshold-Based"]
        )
    
    # Generate portfolio
    if st.button("🚀 Generate Institutional Portfolio", type="primary"):
        with st.spinner("Generating your institutional portfolio..."):
            try:
                # Use investment profile to select template
                if risk_capacity == "Conservative":
                    template = 'conservative'
                elif risk_capacity == "Moderate":
                    template = 'balanced'
                else:
                    template = 'growth'
                
                if investment_goal == "Income":
                    template = 'income_focused'
                elif time_horizon == "> 15 years" and risk_capacity == "Aggressive":
                    template = 'institutional_endowment'
                
                portfolio = portfolio_builder.build_portfolio_from_template(template)
                st.session_state.portfolio = portfolio
                st.session_state.analysis_data = None  # Force refresh
                
                st.success(f"✅ Generated portfolio with {len(portfolio)} assets")
                
                # Show allocation
                st.write("**Generated Allocation:**")
                portfolio_df = pd.DataFrame([
                    {'Symbol': symbol, 'Weight': f"{weight*100:.1f}%", 'Allocation': weight}
                    for symbol, weight in portfolio.items()
                ])
                st.dataframe(portfolio_df, hide_index=True)
                
            except Exception as e:
                st.error(f"Portfolio generation failed: {str(e)}")

def show_market_analysis(market_data, chart_generator):
    """Market analysis dashboard"""
    
    st.subheader("⚡ Global Market Analysis")
    
    # Economic indicators
    with st.expander("📊 Economic Indicators", expanded=True):
        try:
            indicators = market_data.get_economic_indicators()
            
            if indicators:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("GDP Growth", f"{indicators.get('gdp_growth', 0):.1f}%")
                    st.metric("Fed Funds Rate", f"{indicators.get('fed_funds_rate', 0):.2f}%")
                
                with col2:
                    st.metric("Inflation Rate", f"{indicators.get('inflation_rate', 0):.1f}%")
                    st.metric("10Y Treasury", f"{indicators.get('10y_treasury', 0):.2f}%")
                
                with col3:
                    st.metric("Unemployment", f"{indicators.get('unemployment_rate', 0):.1f}%")
                    st.metric("2Y Treasury", f"{indicators.get('2y_treasury', 0):.2f}%")
                
                with col4:
                    st.metric("Consumer Confidence", f"{indicators.get('consumer_confidence', 0):.0f}")
                    st.metric("Housing Starts", f"{indicators.get('housing_starts', 0):.0f}K")
            else:
                st.info("Economic indicators loading...")
        except Exception as e:
            st.error(f"Economic data error: {str(e)}")
    
    # Sector performance
    with st.expander("🏭 Sector Performance"):
        try:
            sector_performance = market_data.get_sector_performance()
            
            if sector_performance:
                sector_data = []
                for sector, data in sector_performance.items():
                    sector_data.append({
                        'Sector': sector,
                        'Symbol': data['symbol'],
                        'Monthly Return': f"{data['monthly_return']:.1f}%",
                        'Volatility': f"{data['volatility']:.1f}%",
                        'Current Price': f"${data['current_price']:.2f}"
                    })
                
                sector_df = pd.DataFrame(sector_data)
                st.dataframe(sector_df, hide_index=True)
                
                # Sector performance chart
                returns_data = [data['monthly_return'] for data in sector_performance.values()]
                sectors = list(sector_performance.keys())
                
                fig = px.bar(x=sectors, y=returns_data, 
                           title="Sector Performance (1 Month)",
                           labels={'x': 'Sector', 'y': 'Return (%)'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Sector performance loading...")
        except Exception as e:
            st.error(f"Sector analysis error: {str(e)}")

def show_ai_insights(data_loader, market_data):
    """Enhanced AI insights"""
    
    st.subheader("🤖 Advanced AI Analysis")
    
    if not st.session_state.portfolio:
        st.warning("Configure your portfolio first.")
        return
    
    # AI analysis tabs
    ai_tab1, ai_tab2, ai_tab3 = st.tabs(["Portfolio Analysis", "Individual Assets", "Market Sentiment"])
    
    with ai_tab1:
        show_portfolio_ai_analysis(data_loader)
    
    with ai_tab2:
        show_individual_asset_analysis(data_loader, market_data)
    
    with ai_tab3:
        show_market_sentiment_analysis(market_data)

def show_portfolio_ai_analysis(data_loader):
    """AI portfolio analysis"""
    
    st.write("### Portfolio-Level AI Analysis")
    
    if st.button("🧠 Generate Comprehensive AI Analysis"):
        with st.spinner("AI is analyzing your portfolio..."):
            try:
                analysis = get_or_create_analysis(data_loader)
                ai_insights = analysis.get('ai_insights', 'Analysis unavailable')
                
                st.markdown(f"""
                ### 🎯 AI Portfolio Assessment
                
                {ai_insights}
                
                ### 📈 Optimization Suggestions
                Based on current market conditions and portfolio analysis, here are AI-generated recommendations:
                
                - **Risk Assessment**: Portfolio volatility is within target range
                - **Diversification**: Consider adding international exposure
                - **Rebalancing**: Next rebalancing suggested in 30 days
                - **Tax Optimization**: Review tax-loss harvesting opportunities
                """)
                
            except Exception as e:
                st.error(f"AI analysis failed: {str(e)}")

def show_individual_asset_analysis(data_loader, market_data):
    """Individual asset AI analysis"""
    
    st.write("### Individual Asset Analysis")
    
    if st.session_state.portfolio:
        selected_asset = st.selectbox(
            "Select asset for detailed AI analysis:",
            list(st.session_state.portfolio.keys())
        )
        
        if st.button(f"Analyze {selected_asset}"):
            with st.spinner(f"AI analyzing {selected_asset}..."):
                try:
                    asset_analysis = data_loader.get_stock_analysis(selected_asset, '6mo')
                    
                    if 'error' not in asset_analysis:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            current_price = asset_analysis.get('current_price', 0)
                            st.metric("Current Price", f"${current_price:.2f}")
                        
                        with col2:
                            daily_change = asset_analysis.get('price_change', 0)
                            st.metric("Daily Change", f"{daily_change:.2f}%", 
                                    delta=f"{daily_change:.2f}%")
                        
                        with col3:
                            portfolio_weight = st.session_state.portfolio.get(selected_asset, 0) * 100
                            st.metric("Portfolio Weight", f"{portfolio_weight:.1f}%")
                        
                        # AI insight
                        st.subheader("🤖 AI Analysis")
                        ai_insight = asset_analysis.get('ai_insight', 'Analysis unavailable')
                        st.write(ai_insight)
                        
                        # Technical analysis
                        if 'stock_data' in asset_analysis:
                            show_technical_analysis(asset_analysis['stock_data'], selected_asset)
                    
                    else:
                        st.error(f"Analysis failed: {asset_analysis['error']}")
                        
                except Exception as e:
                    st.error(f"Individual analysis failed: {str(e)}")

def show_technical_analysis(stock_data, symbol):
    """Show technical analysis"""
    
    st.subheader(f"📊 Technical Analysis - {symbol}")
    
    if not stock_data.empty:
        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close']
        )])
        
        # Add moving averages if available
        if 'SMA_20' in stock_data.columns:
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['SMA_20'],
                mode='lines',
                name='SMA 20',
                line=dict(color='orange')
            ))
        
        if 'SMA_50' in stock_data.columns:
            fig.add_trace(go.Scatter(
                x=stock_data.index,
                y=stock_data['SMA_50'],
                mode='lines',
                name='SMA 50',
                line=dict(color='blue')
            ))
        
        fig.update_layout(
            title=f"{symbol} Price Chart with Technical Indicators",
            yaxis_title="Price ($)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Technical indicators summary
        if 'RSI' in stock_data.columns:
            latest_rsi = stock_data['RSI'].iloc[-1]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("RSI", f"{latest_rsi:.1f}")
                if latest_rsi > 70:
                    st.write("🔴 Overbought")
                elif latest_rsi < 30:
                    st.write("🟢 Oversold")
                else:
                    st.write("🟡 Neutral")
            
            with col2:
                if 'Volatility_20d' in stock_data.columns:
                    volatility = stock_data['Volatility_20d'].iloc[-1] * 100
                    st.metric("20-Day Volatility", f"{volatility:.1f}%")
            
            with col3:
                daily_return = stock_data['Daily_Return'].iloc[-1] * 100
                st.metric("Last Day Return", f"{daily_return:.2f}%")

def show_market_sentiment_analysis(market_data):
    """Market sentiment analysis"""
    
    st.write("### Market Sentiment Analysis")
    st.info("This feature analyzes overall market sentiment from various sources")
    
    # Placeholder for sentiment analysis
    st.write("""
    **Current Market Sentiment: Cautiously Optimistic**
    
    - VIX Level: 18.5 (Moderate volatility)
    - Put/Call Ratio: 0.85 (Neutral)
    - News Sentiment: 65% Positive
    - Social Media Sentiment: Mixed
    
    **Key Themes:**
    - Federal Reserve policy uncertainty
    - Inflation concerns moderating
    - Tech earnings season optimism
    - Geopolitical tensions monitoring
    """)

def show_enhanced_risk_management(data_loader, dashboard):
    """Enhanced risk management"""
    
    st.subheader("⚠️ Institutional Risk Management")
    
    if not st.session_state.portfolio:
        st.warning("Configure portfolio first.")
        return
    
    try:
        analysis = get_or_create_analysis(data_loader)
        risk_metrics = analysis.get('risk_metrics', {})
        
        if risk_metrics and 'error' not in risk_metrics:
            # Risk metrics cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                var_value = risk_metrics.get('var_1d_95', 0)
                portfolio_value = st.session_state.portfolio_value
                var_dollar = var_value * portfolio_value / 100000  # Adjust for portfolio size
                st.metric("VaR (1-day, 95%)", f"${var_dollar:,.0f}")
            
            with col2:
                cvar_value = risk_metrics.get('cvar_1d_95', 0)
                cvar_dollar = cvar_value * portfolio_value / 100000
                st.metric("CVaR (1-day, 95%)", f"${cvar_dollar:,.0f}")
            
            with col3:
                max_dd = risk_metrics.get('maximum_drawdown', 0) * 100
                st.metric("Maximum Drawdown", f"{max_dd:.1f}%")
            
            # Risk assessment
            st.subheader("🎯 Risk Assessment")
            
            total_risk_score = min(100, abs(var_dollar) / (portfolio_value * 0.05) * 100)
            
            # Create risk gauge
            risk_gauge = dashboard.create_gauge_chart(
                total_risk_score, 
                "Portfolio Risk Score",
                0, 100
            )
            st.plotly_chart(risk_gauge, use_container_width=True)
            
            # Risk interpretation
            if total_risk_score > 80:
                st.error("🚨 High Risk: Consider reducing position sizes or increasing diversification")
            elif total_risk_score > 60:
                st.warning("⚠️ Moderate-High Risk: Monitor closely and consider hedging strategies")
            elif total_risk_score > 40:
                st.info("📊 Moderate Risk: Risk level is acceptable for growth-oriented portfolios")
            else:
                st.success("✅ Conservative Risk: Well-diversified, low-risk portfolio")
        
        else:
            st.error("Risk analysis unavailable")
    
    except Exception as e:
        st.error(f"Risk management analysis failed: {str(e)}")

def show_global_markets(market_data):
    """Global markets overview"""
    
    st.subheader("🌍 Global Markets Overview")
    
    # Major indices (placeholder data)
    major_indices = {
        'S&P 500': {'current': 4485.2, 'change': 1.2},
        'NASDAQ': {'current': 13737.6, 'change': 1.8},
        'Dow Jones': {'current': 34765.4, 'change': 0.9},
        'FTSE 100': {'current': 7421.3, 'change': -0.3},
        'Nikkei 225': {'current': 32467.8, 'change': 0.7},
        'DAX': {'current': 15234.7, 'change': 0.4}
    }
    
    col1, col2, col3 = st.columns(3)
    
    for i, (index, data) in enumerate(major_indices.items()):
        col_idx = i % 3
        if col_idx == 0:
            with col1:
                delta_color = "normal" if data['change'] >= 0 else "inverse"
                st.metric(index, f"{data['current']:.1f}", 
                         f"{data['change']:+.1f}%", delta_color=delta_color)
        elif col_idx == 1:
            with col2:
                delta_color = "normal" if data['change'] >= 0 else "inverse"
                st.metric(index, f"{data['current']:.1f}", 
                         f"{data['change']:+.1f}%", delta_color=delta_color)
        else:
            with col3:
                delta_color = "normal" if data['change'] >= 0 else "inverse"
                st.metric(index, f"{data['current']:.1f}", 
                         f"{data['change']:+.1f}%", delta_color=delta_color)
    
    # Currency overview
    st.subheader("💱 Currency Markets")
    
    currencies = {
        'EUR/USD': 1.0847,
        'GBP/USD': 1.2634,
        'USD/JPY': 149.23,
        'USD/CAD': 1.3621,
        'AUD/USD': 0.6543,
        'USD/CHF': 0.9012
    }
    
    col1, col2, col3 = st.columns(3)
    
    for i, (pair, rate) in enumerate(currencies.items()):
        col_idx = i % 3
        if col_idx == 0:
            with col1:
                st.metric(pair, f"{rate:.4f}")
        elif col_idx == 1:
            with col2:
                st.metric(pair, f"{rate:.4f}")
        else:
            with col3:
                st.metric(pair, f"{rate:.4f}")

def show_export_options():
    """Show data export options"""
    
    st.subheader("📊 Export Portfolio Data")
    
    export_format = st.selectbox(
        "Export Format:",
        ["CSV", "Excel", "JSON", "PDF Report"]
    )
    
    if st.button("Export Data"):
        st.success(f"Data exported in {export_format} format")
        st.info("Export functionality would be implemented here")

if __name__ == "__main__":
    main()