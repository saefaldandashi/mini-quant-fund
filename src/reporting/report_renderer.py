"""
Report Renderer - Generates HTML reports with embedded charts.

Creates professional, print-ready reports that can be viewed in browser or saved as PDF.
"""

import logging
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

from .live_collectors import ReportData
from .report_charts import ReportChartGenerator


class ReportRenderer:
    """Renders reports to HTML with embedded charts."""
    
    def __init__(self):
        """Initialize renderer."""
        self.chart_generator = ReportChartGenerator()
    
    def render_daily_report(self, data: ReportData) -> str:
        """
        Render daily report to HTML.
        
        Args:
            data: ReportData object with all metrics
        
        Returns:
            Complete HTML string
        """
        # Generate charts
        charts = self.chart_generator.generate_all_charts(data)
        
        # Build HTML
        html = self._render_html(data, charts, 'Daily')
        return html
    
    def render_weekly_report(self, data: ReportData) -> str:
        """Render weekly report."""
        charts = self.chart_generator.generate_all_charts(data)
        return self._render_html(data, charts, 'Weekly')
    
    def render_monthly_report(self, data: ReportData) -> str:
        """Render monthly report."""
        charts = self.chart_generator.generate_all_charts(data)
        return self._render_html(data, charts, 'Monthly')
    
    def _render_html(self, data: ReportData, charts: Dict[str, str], report_type: str) -> str:
        """Build complete HTML report."""
        
        # Format values
        p = data.portfolio
        
        # Build data quality section
        data_quality_html = self._build_data_quality_section(data)
        
        # Build positions table
        positions_html = self._build_positions_table(data.positions)
        
        # Build trades table
        trades_html = self._build_trades_table(data.trades)
        
        # Build strategy table
        strategy_html = self._build_strategy_table(data.strategy_performance)
        
        # Build macro section
        macro_html = self._build_macro_section(data.macro)
        
        # Build recommendations
        recommendations_html = self._build_recommendations(data.recommendations)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{report_type} Report - {data.report_date.strftime('%B %d, %Y')}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #ffffff;
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 2em;
            margin-bottom: 5px;
            background: linear-gradient(90deg, #00d4aa, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        h2 {{
            font-size: 1.3em;
            color: #00d4aa;
            margin: 30px 0 15px 0;
            padding-bottom: 5px;
            border-bottom: 1px solid #333;
        }}
        h3 {{
            font-size: 1.1em;
            color: #888;
            margin: 20px 0 10px 0;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #333;
        }}
        .header-right {{
            text-align: right;
            color: #888;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-box {{
            background: #1a1a1a;
            border-radius: 8px;
            padding: 15px;
            border-left: 3px solid #00d4aa;
        }}
        .metric-label {{
            font-size: 0.85em;
            color: #888;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 1.5em;
            font-weight: 600;
        }}
        .positive {{ color: #22c55e; }}
        .negative {{ color: #ef4444; }}
        .neutral {{ color: #888; }}
        
        .chart-container {{
            background: #1a1a1a;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
        }}
        .chart-container img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 0.9em;
        }}
        th {{
            background: #1a1a1a;
            color: #00d4aa;
            padding: 12px;
            text-align: left;
            font-weight: 500;
        }}
        td {{
            padding: 10px 12px;
            border-bottom: 1px solid #333;
        }}
        tr:hover {{
            background: rgba(0, 212, 170, 0.05);
        }}
        
        .insight-box {{
            background: #1a1a1a;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            border-left: 3px solid #a855f7;
        }}
        .insight-box li {{
            margin: 8px 0;
            padding-left: 10px;
        }}
        
        .regime-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }}
        .regime-bull {{ background: rgba(34, 197, 94, 0.2); color: #22c55e; }}
        .regime-bear {{ background: rgba(239, 68, 68, 0.2); color: #ef4444; }}
        .regime-neutral {{ background: rgba(136, 136, 136, 0.2); color: #888; }}
        
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #333;
            text-align: center;
            color: #666;
            font-size: 0.85em;
        }}
        
        @media print {{
            body {{ background: white; color: black; }}
            .metric-box, .chart-container, .insight-box {{
                background: #f5f5f5;
                border-color: #ddd;
            }}
            .metric-label {{ color: #666; }}
            h1 {{ color: #333; -webkit-text-fill-color: #333; }}
            h2 {{ color: #333; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>🤖 Mini Quant Fund</h1>
                <div style="color: #888; font-size: 1.1em;">{report_type} Report</div>
            </div>
            <div class="header-right">
                <div style="font-size: 1.2em;">{data.report_date.strftime('%B %d, %Y')}</div>
                <div>Generated at {datetime.now().strftime('%H:%M:%S')}</div>
            </div>
        </div>
        
        <!-- Data Quality Indicator -->
        {data_quality_html}
        
        <!-- Executive Summary -->
        <h2>📊 Executive Summary</h2>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Portfolio Value</div>
                <div class="metric-value">${p.portfolio_value:,.0f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Return (1D)</div>
                <div class="metric-value {self._pnl_class(p.return_1d) if p.has_portfolio_history else 'neutral'}">{self._format_return(p.return_1d, p.has_portfolio_history)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Return (1M)</div>
                <div class="metric-value {self._pnl_class(p.return_1m) if p.has_portfolio_history else 'neutral'}">{self._format_return(p.return_1m, p.has_portfolio_history)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Alpha vs SPY (1D)</div>
                <div class="metric-value {self._pnl_class(p.alpha_1d) if (p.has_portfolio_history and p.has_benchmark_data) else 'neutral'}">{self._format_return(p.alpha_1d, p.has_portfolio_history and p.has_benchmark_data)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Volatility (20D Ann.)</div>
                <div class="metric-value neutral">{self._format_pct(p.volatility_20d, getattr(p, 'has_sufficient_data_for_vol', p.has_portfolio_history))}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Current Drawdown</div>
                <div class="metric-value negative">{self._format_pct(p.current_drawdown, p.has_portfolio_history)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value {self._sharpe_class(p.sharpe_ratio) if getattr(p, 'has_sufficient_data_for_sharpe', False) else 'neutral'}">{self._format_sharpe(p.sharpe_ratio, getattr(p, 'has_sufficient_data_for_sharpe', p.has_portfolio_history))}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Cash Available</div>
                <div class="metric-value">${p.cash:,.0f}</div>
            </div>
        </div>
        
        <!-- L/S Exposure (if applicable) -->
        {self._build_exposure_section(p)}
        
        <!-- Performance Chart -->
        <h2>📈 Performance</h2>
        {self._chart_section(charts.get('equity_curve'), 'Portfolio vs Benchmark')}
        
        <!-- Risk Charts -->
        <h2>⚠️ Risk Analysis</h2>
        {self._chart_section(charts.get('drawdown'), 'Drawdown')}
        
        <!-- Holdings -->
        <h2>💼 Holdings ({len(data.positions)} positions)</h2>
        {self._chart_section(charts.get('holdings'), 'Position Weights')}
        {positions_html}
        
        <!-- P/L Breakdown -->
        <h2>💰 P/L Breakdown</h2>
        {self._chart_section(charts.get('pnl_breakdown'), 'Unrealized P/L by Position')}
        
        <!-- Strategy Performance -->
        <h2>🎯 Strategy Performance</h2>
        {self._chart_section(charts.get('strategy_weights'), 'Strategy Weights & Scores')}
        {strategy_html}
        
        <!-- Macro Context -->
        <h2>🌍 Macro Context</h2>
        {macro_html}
        
        <!-- Recent Trades -->
        <h2>📝 Recent Trades ({len(data.trades)} in period)</h2>
        {trades_html}
        
        <!-- Learning Insights -->
        <h2>🧠 Learning Insights</h2>
        <div class="insight-box">
            <h3>Patterns Discovered: {data.patterns_found}</h3>
            {recommendations_html}
        </div>
        
        <div class="footer">
            <p>Mini Quant Fund - Automated Report</p>
            <p>Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def _pnl_class(self, value: float) -> str:
        """Return CSS class based on P/L value."""
        if value > 0:
            return 'positive'
        elif value < 0:
            return 'negative'
        return 'neutral'
    
    def _sharpe_class(self, value: float) -> str:
        """Return CSS class based on Sharpe ratio."""
        if value > 1.0:
            return 'positive'
        elif value < 0:
            return 'negative'
        return 'neutral'
    
    def _format_return(self, value: float, has_data: bool) -> str:
        """Format a return value with N/A handling."""
        if not has_data:
            return '<span style="color: #666;">N/A</span>'
        return f'{value*100:+.2f}%'
    
    def _format_pct(self, value: float, has_data: bool) -> str:
        """Format a percentage value with N/A handling."""
        if not has_data:
            return '<span style="color: #666;">N/A</span>'
        return f'{value*100:.1f}%'
    
    def _format_sharpe(self, value: float, has_data: bool) -> str:
        """Format Sharpe ratio with N/A handling."""
        if not has_data:
            return '<span style="color: #666;">N/A</span>'
        return f'{value:.2f}'
    
    def _chart_section(self, chart_data: Optional[str], title: str) -> str:
        """Build chart section HTML."""
        if not chart_data:
            return f'<div class="chart-container"><p style="color: #888; text-align: center;">No data available for {title}</p></div>'
        
        return f'''<div class="chart-container">
            <img src="{chart_data}" alt="{title}">
        </div>'''
    
    def _build_positions_table(self, positions) -> str:
        """Build positions table HTML with L/S attribution."""
        if not positions:
            return '<p style="color: #888;">No positions</p>'
        
        # Calculate L/S attribution
        long_positions = [p for p in positions if getattr(p, 'side', 'long') == 'long']
        short_positions = [p for p in positions if getattr(p, 'side', 'long') == 'short']
        
        long_pnl = sum(p.unrealized_pnl for p in long_positions)
        short_pnl = sum(p.unrealized_pnl for p in short_positions)
        long_value = sum(p.market_value for p in long_positions)
        short_value = sum(abs(p.market_value) for p in short_positions)
        
        # L/S attribution summary
        attribution = ""
        if short_positions:
            long_pnl_class = 'positive' if long_pnl >= 0 else 'negative'
            short_pnl_class = 'positive' if short_pnl >= 0 else 'negative'
            attribution = f"""
            <div style="display: flex; gap: 20px; margin-bottom: 15px; padding: 10px; background: #1a1a2e; border-radius: 8px;">
                <div style="flex: 1;">
                    <span style="color: #00d4aa;">LONG BOOK</span>
                    <div style="font-size: 0.9em; color: #888;">
                        {len(long_positions)} positions | ${long_value:,.0f}
                    </div>
                    <div class="{long_pnl_class}" style="font-size: 1.2em;">${long_pnl:+,.0f} P/L</div>
                </div>
                <div style="flex: 1;">
                    <span style="color: #ff6b6b;">SHORT BOOK</span>
                    <div style="font-size: 0.9em; color: #888;">
                        {len(short_positions)} positions | ${short_value:,.0f}
                    </div>
                    <div class="{short_pnl_class}" style="font-size: 1.2em;">${short_pnl:+,.0f} P/L</div>
                </div>
            </div>"""
        
        rows = ""
        for pos in positions[:15]:  # Limit to top 15
            pnl_class = 'positive' if pos.unrealized_pnl >= 0 else 'negative'
            side = getattr(pos, 'side', 'long')
            side_badge = '<span style="color: #00d4aa;">▲ L</span>' if side == 'long' else '<span style="color: #ff6b6b;">▼ S</span>'
            rows += f"""<tr>
                <td>{side_badge} <strong>{pos.symbol}</strong></td>
                <td>{pos.quantity}</td>
                <td>${pos.current_price:.2f}</td>
                <td>${pos.market_value:,.0f}</td>
                <td>{pos.weight*100:.1f}%</td>
                <td class="{pnl_class}">${pos.unrealized_pnl:+,.0f}</td>
                <td class="{pnl_class}">{pos.unrealized_pnl_pct:+.1f}%</td>
            </tr>"""
        
        return f"""{attribution}<table>
            <thead>
                <tr>
                    <th>Symbol</th>
                    <th>Qty</th>
                    <th>Price</th>
                    <th>Value</th>
                    <th>Weight</th>
                    <th>P/L ($)</th>
                    <th>P/L (%)</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>"""
    
    def _build_trades_table(self, trades) -> str:
        """Build trades table HTML."""
        if not trades:
            return '<p style="color: #888;">No trades in this period</p>'
        
        rows = ""
        for trade in trades[:20]:  # Limit to 20
            side_class = 'positive' if trade.side.lower() == 'buy' else 'negative'
            rows += f"""<tr>
                <td>{trade.timestamp.strftime('%m/%d %H:%M')}</td>
                <td><strong>{trade.symbol}</strong></td>
                <td class="{side_class}">{trade.side.upper()}</td>
                <td>{trade.quantity}</td>
                <td>${trade.filled_price:.2f}</td>
                <td>${trade.value:,.0f}</td>
            </tr>"""
        
        return f"""<table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Symbol</th>
                    <th>Side</th>
                    <th>Qty</th>
                    <th>Price</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>"""
    
    def _build_strategy_table(self, strategies) -> str:
        """Build strategy performance table HTML."""
        if not strategies:
            return '<p style="color: #888;">No strategy data available. Run a rebalance to generate strategy analysis.</p>'
        
        rows = ""
        for s in strategies:
            # Format win rate - show N/A if no data (0%)
            if s.win_rate > 0:
                wr_class = 'positive' if s.win_rate >= 0.5 else 'negative'
                wr_display = f'{s.win_rate*100:.0f}%'
            else:
                wr_class = 'neutral'
                wr_display = 'N/A'
            
            # Format confidence - show N/A if no data
            if s.confidence > 0:
                conf_display = f'{s.confidence*100:.0f}%'
            else:
                conf_display = 'N/A'
            
            # Format debate score - highlight if non-zero
            if s.debate_score > 0:
                ds_class = 'positive' if s.debate_score >= 0.5 else ''
                ds_display = f'{s.debate_score:.2f}'
            else:
                ds_class = 'neutral'
                ds_display = '--'
            
            rows += f"""<tr>
                <td><strong>{s.name}</strong></td>
                <td>{s.weight*100:.0f}%</td>
                <td class="{wr_class}">{wr_display}</td>
                <td>{conf_display}</td>
                <td class="{ds_class}">{ds_display}</td>
            </tr>"""
        
        return f"""<table>
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Weight</th>
                    <th>Win Rate</th>
                    <th>Confidence</th>
                    <th>Debate Score</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>"""
    
    def _build_macro_section(self, macro) -> str:
        """Build macro context section with live data."""
        regime_class = 'regime-bull' if 'bull' in macro.regime_label.lower() else (
            'regime-bear' if 'bear' in macro.regime_label.lower() or 'crisis' in macro.regime_label.lower() else 'regime-neutral'
        )
        
        # Data source indicator
        data_source = '<span style="color: #22c55e;">✓ Live Data</span>' if getattr(macro, 'has_live_data', False) else '<span style="color: #f59e0b;">○ Cached Data</span>'
        
        # VIX color coding
        vix_class = 'positive' if macro.vix < 20 else ('negative' if macro.vix > 25 else 'neutral')
        
        # SPY change
        spy_change = getattr(macro, 'spy_change_pct', 0)
        spy_class = 'positive' if spy_change > 0 else ('negative' if spy_change < 0 else 'neutral')
        
        events_html = ""
        if macro.top_events:
            events_html = "<h3 style='margin-top: 20px;'>Top Market Events</h3><div class='insight-box'><ul>"
            for event in macro.top_events[:5]:
                events_html += f"<li>{event}</li>"
            events_html += "</ul></div>"
        
        # Build market data section
        market_data = f"""
        <h3>Market Data</h3>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Market Regime</div>
                <div><span class="regime-badge {regime_class}">{macro.regime_label.upper().replace('_', ' ')}</span></div>
                <div style="font-size: 0.8em; color: #888;">Confidence: {macro.regime_confidence*100:.0f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">VIX (Fear Index)</div>
                <div class="metric-value {vix_class}">{macro.vix:.1f}</div>
                <div style="font-size: 0.8em; color: #888;">{"Low Vol" if macro.vix < 20 else ("High Vol" if macro.vix > 25 else "Normal")}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">SPY Price</div>
                <div class="metric-value">${macro.spy_price:.2f}</div>
                <div style="font-size: 0.8em;" class="{spy_class}">{spy_change:+.2f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Risk Sentiment</div>
                <div class="metric-value" style="color: {'#22c55e' if 'ON' in macro.risk_sentiment else ('#ef4444' if 'OFF' in macro.risk_sentiment else '#888')}">{macro.risk_sentiment}</div>
            </div>
        </div>
        """
        
        # Build economic indicators section
        cpi = getattr(macro, 'cpi_yoy', 0)
        unemployment = getattr(macro, 'unemployment_rate', 0)
        fed_funds = getattr(macro, 'fed_funds_rate', 0)
        gold = getattr(macro, 'gold_price', 0)
        oil = getattr(macro, 'oil_price', 0)
        
        econ_html = f"""
        <h3>Economic Indicators</h3>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">10Y Treasury</div>
                <div class="metric-value">{macro.treasury_10y:.2f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">CPI (YoY)</div>
                <div class="metric-value">{cpi:.1f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Unemployment</div>
                <div class="metric-value">{unemployment:.1f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Fed Funds Rate</div>
                <div class="metric-value">{fed_funds:.2f}%</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Gold</div>
                <div class="metric-value">${gold:,.0f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Oil (WTI)</div>
                <div class="metric-value">${oil:.2f}</div>
            </div>
        </div>
        """
        
        # Build indices section
        indices_html = f"""
        <h3>Computed Indices</h3>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Inflation Pressure</div>
                <div class="metric-value {self._index_class(macro.inflation_pressure)}">{macro.inflation_pressure:+.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Growth Momentum</div>
                <div class="metric-value {self._index_class(macro.growth_momentum)}">{macro.growth_momentum:+.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Geopolitical Risk</div>
                <div class="metric-value {self._index_class(-macro.geopolitical_risk)}">{macro.geopolitical_risk:+.2f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Financial Stress</div>
                <div class="metric-value {self._index_class(-macro.financial_stress)}">{macro.financial_stress:+.2f}</div>
            </div>
        </div>
        """
        
        return f"""
        <div style="margin-bottom: 10px; font-size: 0.9em;">Data Source: {data_source}</div>
        {market_data}
        {econ_html}
        {indices_html}
        {events_html}
        """
    
    def _index_class(self, value: float) -> str:
        """Get CSS class for index value."""
        if value > 0.3:
            return 'positive'
        elif value < -0.3:
            return 'negative'
        return 'neutral'
    
    def _build_exposure_section(self, portfolio) -> str:
        """Build L/S exposure summary if applicable."""
        # Check if we have L/S exposure data
        gross = getattr(portfolio, 'gross_exposure', 0)
        net = getattr(portfolio, 'net_exposure', 0)
        long_exp = getattr(portfolio, 'long_exposure', 0)
        short_exp = getattr(portfolio, 'short_exposure', 0)
        long_count = getattr(portfolio, 'long_count', 0)
        short_count = getattr(portfolio, 'short_count', 0)
        leverage = getattr(portfolio, 'leverage', 1.0)
        
        # Only show if we have short positions
        if short_count == 0 and short_exp == 0:
            return ""
        
        net_class = 'positive' if net >= 0 else 'negative'
        
        return f"""
        <h2>📊 Long/Short Exposure</h2>
        <div class="metrics-grid">
            <div class="metric-box">
                <div class="metric-label">Gross Exposure</div>
                <div class="metric-value">${gross:,.0f}</div>
                <div style="font-size: 0.8em; color: #888;">Leverage: {leverage:.1f}x</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Net Exposure</div>
                <div class="metric-value {net_class}">${net:+,.0f}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Long Book</div>
                <div class="metric-value positive">${long_exp:,.0f}</div>
                <div style="font-size: 0.8em; color: #888;">{long_count} positions</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Short Book</div>
                <div class="metric-value negative">${short_exp:,.0f}</div>
                <div style="font-size: 0.8em; color: #888;">{short_count} positions</div>
            </div>
        </div>
        """
    
    def _build_recommendations(self, recommendations) -> str:
        """Build recommendations list."""
        if not recommendations:
            return '<p style="color: #888;">No recommendations at this time</p>'
        
        items = "".join(f"<li>{rec}</li>" for rec in recommendations)
        return f"<ul>{items}</ul>"
    
    def _build_data_quality_section(self, data: ReportData) -> str:
        """Build data quality indicator section."""
        p = data.portfolio
        quality = data.data_quality
        
        indicators = []
        
        # Portfolio history
        if p.has_portfolio_history:
            indicators.append(f'<span style="color: #22c55e;">✓ Portfolio History ({p.data_points} days)</span>')
        else:
            indicators.append('<span style="color: #ef4444;">✗ No Portfolio History</span>')
        
        # Benchmark data
        if p.has_benchmark_data:
            indicators.append(f'<span style="color: #22c55e;">✓ Benchmark Data (SPY)</span>')
        else:
            indicators.append('<span style="color: #ef4444;">✗ No Benchmark Data</span>')
        
        # Positions
        if quality.get('positions'):
            indicators.append(f'<span style="color: #22c55e;">✓ Positions</span>')
        else:
            indicators.append('<span style="color: #f59e0b;">○ No Positions</span>')
        
        # Trades
        if quality.get('trades'):
            indicators.append(f'<span style="color: #22c55e;">✓ Trade History</span>')
        else:
            indicators.append('<span style="color: #888;">○ No Recent Trades</span>')
        
        return f'''<div style="background: #1a1a1a; border-radius: 8px; padding: 10px 15px; margin-bottom: 20px; font-size: 0.9em;">
            <strong>Data Sources:</strong> {" | ".join(indicators)}
        </div>'''
    
    def _format_metric(self, value: float, format_str: str = "+.2f%", 
                       has_data: bool = True, multiply_100: bool = True) -> str:
        """Format a metric value, showing N/A if no data."""
        if not has_data:
            return '<span style="color: #666;">N/A</span>'
        
        if multiply_100:
            value = value * 100
        
        if format_str.startswith('+'):
            return f'{value:+.2f}%' if format_str.endswith('%') else f'{value:+.2f}'
        else:
            return f'{value:.2f}%' if format_str.endswith('%') else f'{value:.2f}'


def generate_report(
    broker,
    learning_engine=None,
    app_state: Dict = None,
    report_type: str = 'daily',
    output_path: Optional[Path] = None,
) -> Dict[str, any]:
    """
    Generate a complete report and feed insights to learning system.
    
    Args:
        broker: AlpacaBroker instance
        learning_engine: LearningEngine instance
        app_state: Dict with app global state
        report_type: 'daily', 'weekly', or 'monthly'
        output_path: Optional path to save HTML file
    
    Returns:
        Dict with 'success', 'html', 'path', 'data', 'learning_insights'
    """
    try:
        # Collect data
        from .live_collectors import LiveDataCollector
        collector = LiveDataCollector(broker, learning_engine, app_state)
        data = collector.collect_report_data(report_type=report_type)
        
        # === LEARNING INTEGRATION ===
        # 1. Extract insights and update learning system
        from .learning_feedback import ReportLearningFeedback
        feedback = ReportLearningFeedback(learning_engine)
        insights = feedback.extract_and_learn(data)
        
        # 2. Persist report summary for long-term learning
        try:
            from src.learning.report_learning import ReportLearningStore, create_report_summary_from_data
            
            report_store = ReportLearningStore()
            summary = create_report_summary_from_data(data)
            report_store.record_report(summary)
            
            # Get learned insights to include in report
            learned_insights = report_store.get_learned_insights()
            data.recommendations.extend(learned_insights[:3])  # Add top 3
            
            logging.info(f"Persisted report to learning store, total history: {len(report_store.report_history)}")
        except Exception as e:
            logging.warning(f"Could not persist to learning store: {e}")
        
        # Render HTML
        renderer = ReportRenderer()
        if report_type == 'daily':
            html = renderer.render_daily_report(data)
        elif report_type == 'weekly':
            html = renderer.render_weekly_report(data)
        else:
            html = renderer.render_monthly_report(data)
        
        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
        
        return {
            'success': True,
            'html': html,
            'path': str(output_path) if output_path else None,
            'data': data,
            'learning_insights': [{'category': i.category, 'description': i.description} for i in insights],
        }
    
    except Exception as e:
        logging.error(f"Report generation failed: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'html': None,
            'path': None,
        }
