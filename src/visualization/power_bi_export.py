"""
Power BI Export Module
Export portfolio data to Power BI compatible formats
"""

import pandas as pd
import json
from typing import Dict, List
from datetime import datetime
import logging
from pathlib import Path

class PowerBIExporter:
    """Export data for Power BI integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.export_dir = Path("data/powerbi_exports")
        self.export_dir.mkdir(exist_ok=True)
    
    def export_portfolio_data(self, portfolio_data: pd.DataFrame, 
                            filename: str = None) -> str:
        """Export portfolio data to CSV for Power BI"""
        
        if filename is None:
            filename = f"portfolio_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = self.export_dir / filename
        
        # Prepare data for Power BI
        export_data = portfolio_data.reset_index()
        export_data['Date'] = pd.to_datetime(export_data['Date'])
        
        # Add metadata columns
        export_data['ExportTimestamp'] = datetime.now()
        export_data['DataSource'] = 'AI Portfolio Manager'
        
        export_data.to_csv(filepath, index=False)
        
        self.logger.info(f"Portfolio data exported to {filepath}")
        return str(filepath)
    
    def export_performance_metrics(self, metrics: Dict, 
                                 filename: str = None) -> str:
        """Export performance metrics as JSON"""
        
        if filename is None:
            filename = f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.export_dir / filename
        
        # Add metadata
        export_data = {
            'metrics': metrics,
            'export_timestamp': datetime.now().isoformat(),
            'data_source': 'AI Portfolio Manager',
            'version': '1.0'
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Metrics exported to {filepath}")
        return str(filepath)
    
    def create_powerbi_dataset_schema(self) -> Dict:
        """Generate Power BI dataset schema"""
        
        schema = {
            "name": "AIPortfolioManager",
            "tables": [
                {
                    "name": "PortfolioData",
                    "columns": [
                        {"name": "Date", "dataType": "DateTime"},
                        {"name": "Daily_Return", "dataType": "Double"},
                        {"name": "Cumulative_Return", "dataType": "Double"},
                        {"name": "Portfolio_Value", "dataType": "Double"},
                        {"name": "Volatility_30d", "dataType": "Double"},
                        {"name": "Drawdown", "dataType": "Double"}
                    ]
                },
                {
                    "name": "Holdings",
                    "columns": [
                        {"name": "Symbol", "dataType": "String"},
                        {"name": "Weight", "dataType": "Double"},
                        {"name": "Current_Price", "dataType": "Double"},
                        {"name": "Daily_Change", "dataType": "Double"}
                    ]
                },
                {
                    "name": "Metrics",
                    "columns": [
                        {"name": "Metric_Name", "dataType": "String"},
                        {"name": "Value", "dataType": "Double"},
                        {"name": "Category", "dataType": "String"},
                        {"name": "Date", "dataType": "DateTime"}
                    ]
                }
            ]
        }
        
        return schema
    
    def export_holdings_data(self, portfolio_weights: Dict[str, float],
                           stock_data: Dict[str, pd.DataFrame],
                           filename: str = None) -> str:
        """Export holdings data for Power BI"""
        
        if filename is None:
            filename = f"holdings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        holdings_data = []
        
        for symbol, weight in portfolio_weights.items():
            if symbol in stock_data and not stock_data[symbol].empty:
                latest = stock_data[symbol].iloc[-1]
                holdings_data.append({
                    'Symbol': symbol,
                    'Weight': weight,
                    'Current_Price': latest['Close'],
                    'Daily_Change': latest.get('Daily_Return', 0) * 100,
                    'Volume': latest['Volume'],
                    'Export_Date': datetime.now().date()
                })
        
        df = pd.DataFrame(holdings_data)
        filepath = self.export_dir / filename
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Holdings data exported to {filepath}")
        return str(filepath)
    
    def create_power_bi_report_config(self) -> Dict:
        """Generate Power BI report configuration"""
        
        config = {
            "version": "1.0",
            "reportName": "AI Portfolio Dashboard",
            "pages": [
                {
                    "name": "Overview",
                    "visuals": [
                        {
                            "type": "lineChart",
                            "title": "Portfolio Performance",
                            "data": "PortfolioData",
                            "x": "Date",
                            "y": "Cumulative_Return"
                        },
                        {
                            "type": "pieChart", 
                            "title": "Asset Allocation",
                            "data": "Holdings",
                            "category": "Symbol",
                            "value": "Weight"
                        }
                    ]
                },
                {
                    "name": "Risk Analysis",
                    "visuals": [
                        {
                            "type": "gauge",
                            "title": "Risk Score",
                            "data": "Metrics",
                            "value": "Value",
                            "filter": "Metric_Name = 'Risk_Score'"
                        }
                    ]
                }
            ]
        }
        
        return config

# Global instance
powerbi_exporter = PowerBIExporter()