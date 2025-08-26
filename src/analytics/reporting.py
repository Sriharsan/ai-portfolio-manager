"""
Reporting Module - Generate Portfolio Reports
"""

import pandas as pd
from datetime import datetime
from typing import Dict

class ReportGenerator:
    """Generate formatted portfolio reports"""
    
    def generate_summary_report(self, analysis_data: Dict) -> str:
        """Generate text summary report"""
        
        if 'error' in analysis_data:
            return f"Report unavailable: {analysis_data['error']}"
        
        metrics = analysis_data.get('performance_metrics', {})
        risk_metrics = analysis_data.get('risk_metrics', {})
        
        report = f"""
PORTFOLIO PERFORMANCE REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*50}

PERFORMANCE SUMMARY
• Total Return: {metrics.get('total_return', 0)*100:.1f}%
• Annualized Return: {metrics.get('annualized_return', 0)*100:.1f}%
• Volatility: {metrics.get('volatility', 0)*100:.1f}%
• Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}
• Max Drawdown: {metrics.get('max_drawdown', 0)*100:.1f}%

RISK ASSESSMENT
• VaR (1-day): ${risk_metrics.get('var_1d_95', 0):,.0f}
• CVaR (1-day): ${risk_metrics.get('cvar_1d_95', 0):,.0f}
• Annual Volatility: {risk_metrics.get('volatility_annual', 0)*100:.1f}%

AI INSIGHTS
{analysis_data.get('ai_insights', 'AI analysis pending...')}
"""
        
        return report
    
    def export_to_csv(self, data: pd.DataFrame, filename: str) -> str:
        """Export data to CSV"""
        filepath = f"data/processed/{filename}_{datetime.now().strftime('%Y%m%d')}.csv"
        data.to_csv(filepath)
        return filepath

report_generator = ReportGenerator()