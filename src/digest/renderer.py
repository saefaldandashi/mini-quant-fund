"""
HTML Renderer for Daily Digest.

Renders digest output to clean, investor-grade HTML using Jinja2.
Produces embeddable HTML that works well for PDF conversion.
"""

import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .schema import DigestOutput, DigestSection, MarketSnapshot

logger = logging.getLogger(__name__)


class DigestRenderer:
    """Renders digest to HTML."""
    
    def __init__(self, template_dir: Optional[str] = None):
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        else:
            template_dir = Path(template_dir)
            
        self.template_dir = template_dir
        
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(['html', 'xml']),
        )
        
        # Add custom filters
        self.env.filters['format_date'] = self._format_date
        self.env.filters['format_time'] = self._format_time
        self.env.filters['format_pct'] = self._format_pct
        self.env.filters['format_price'] = self._format_price
        self.env.filters['risk_color'] = self._risk_color
        self.env.filters['direction_icon'] = self._direction_icon
    
    def render(
        self, 
        digest: DigestOutput,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Render digest to HTML.
        
        Args:
            digest: DigestOutput object to render
            output_path: Optional path to save HTML file
            
        Returns:
            Rendered HTML string
        """
        try:
            template = self.env.get_template("daily_digest.html")
        except Exception:
            # Use inline template if file not found
            logger.warning("Template not found, using inline template")
            template = self.env.from_string(INLINE_TEMPLATE)
        
        # Prepare context
        context = {
            'digest': digest,
            'date': digest.metadata.date,
            'generated_at': digest.metadata.generated_at,
            'sections': digest.get_sections_by_impact(),
            'executive_brief': digest.executive_brief,
            'market_snapshot': digest.market_snapshot,
            'metadata': digest.metadata,
            'now': datetime.utcnow(),
        }
        
        html = template.render(**context)
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html)
            logger.info(f"Saved HTML digest to {output_path}")
            
        return html
    
    @staticmethod
    def _format_date(dt) -> str:
        if isinstance(dt, str):
            return dt
        return dt.strftime('%B %d, %Y')
    
    @staticmethod
    def _format_time(dt) -> str:
        if isinstance(dt, str):
            return dt
        return dt.strftime('%H:%M UTC')
    
    @staticmethod
    def _format_pct(value, decimals=2) -> str:
        if value is None:
            return "N/A"
        sign = '+' if value >= 0 else ''
        return f"{sign}{value:.{decimals}f}%"
    
    @staticmethod
    def _format_price(value, decimals=2) -> str:
        if value is None:
            return "N/A"
        return f"{value:,.{decimals}f}"
    
    @staticmethod
    def _risk_color(risk_tone: str) -> str:
        colors = {
            'Risk-On': '#22c55e',
            'Risk-Off': '#ef4444',
            'Neutral': '#f59e0b',
        }
        return colors.get(risk_tone, '#6b7280')
    
    @staticmethod
    def _direction_icon(direction: str) -> str:
        icons = {
            'risk_on': '📈',
            'risk_off': '📉',
            'oil_up': '🛢️⬆️',
            'oil_down': '🛢️⬇️',
            'rates_up': '📊⬆️',
            'rates_down': '📊⬇️',
            'usd_up': '💵⬆️',
            'usd_down': '💵⬇️',
            'neutral': '➖',
        }
        return icons.get(direction.lower(), '➖')


# Inline template as fallback
INLINE_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Intelligence Digest - {{ date }}</title>
    <style>
        :root {
            --primary: #1e3a5f;
            --secondary: #3b82f6;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --text: #1f2937;
            --text-dim: #6b7280;
            --bg: #ffffff;
            --bg-alt: #f9fafb;
            --border: #e5e7eb;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 11pt;
            line-height: 1.5;
            color: var(--text);
            background: var(--bg);
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            border-bottom: 3px solid var(--primary);
            padding-bottom: 15px;
            margin-bottom: 20px;
        }
        
        .header h1 {
            font-size: 24pt;
            color: var(--primary);
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .header .date {
            font-size: 12pt;
            color: var(--text-dim);
        }
        
        .executive-brief {
            background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 25px;
            border-left: 4px solid var(--secondary);
        }
        
        .executive-brief h2 {
            font-size: 14pt;
            color: var(--primary);
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .risk-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 10pt;
            font-weight: 600;
            color: white;
        }
        
        .takeaways {
            margin-bottom: 15px;
        }
        
        .takeaways li {
            margin-bottom: 8px;
            padding-left: 5px;
        }
        
        .themes {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        
        .theme-chip {
            background: var(--primary);
            color: white;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 9pt;
            font-weight: 500;
        }
        
        .market-snapshot {
            background: var(--bg-alt);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 25px;
        }
        
        .market-snapshot h3 {
            font-size: 11pt;
            color: var(--text-dim);
            margin-bottom: 10px;
        }
        
        .market-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
        }
        
        .market-item {
            text-align: center;
            padding: 8px;
            background: white;
            border-radius: 6px;
            border: 1px solid var(--border);
        }
        
        .market-item .label {
            font-size: 8pt;
            color: var(--text-dim);
            text-transform: uppercase;
        }
        
        .market-item .value {
            font-size: 11pt;
            font-weight: 600;
        }
        
        .market-item .change {
            font-size: 9pt;
        }
        
        .positive { color: var(--success); }
        .negative { color: var(--danger); }
        
        .section {
            margin-bottom: 25px;
            page-break-inside: avoid;
        }
        
        .section-header {
            display: flex;
            align-items: center;
            gap: 10px;
            padding-bottom: 8px;
            border-bottom: 2px solid var(--border);
            margin-bottom: 12px;
        }
        
        .section-icon {
            font-size: 18pt;
        }
        
        .section-title {
            font-size: 14pt;
            color: var(--primary);
            font-weight: 600;
        }
        
        .impact-badge {
            margin-left: auto;
            background: var(--bg-alt);
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 9pt;
            color: var(--text-dim);
        }
        
        .summary-block {
            margin-bottom: 15px;
        }
        
        .summary-block h4 {
            font-size: 10pt;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }
        
        .summary-block ul {
            margin-left: 15px;
        }
        
        .summary-block li {
            margin-bottom: 5px;
            font-size: 10pt;
        }
        
        .market-impacts {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-top: 10px;
        }
        
        .impact-column {
            background: var(--bg-alt);
            border-radius: 6px;
            padding: 12px;
        }
        
        .impact-column h5 {
            font-size: 10pt;
            color: var(--primary);
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        
        .impact-column li {
            font-size: 9pt;
            margin-bottom: 4px;
        }
        
        .source-tag {
            color: var(--secondary);
            font-size: 8pt;
        }
        
        .articles-list {
            margin-top: 12px;
            border-top: 1px dashed var(--border);
            padding-top: 10px;
        }
        
        .article-item {
            display: flex;
            gap: 10px;
            margin-bottom: 8px;
            font-size: 9pt;
        }
        
        .article-time {
            color: var(--text-dim);
            min-width: 50px;
        }
        
        .article-headline {
            flex: 1;
        }
        
        .article-headline a {
            color: var(--text);
            text-decoration: none;
        }
        
        .article-headline a:hover {
            text-decoration: underline;
        }
        
        .watchlist {
            background: #fffbeb;
            border-radius: 6px;
            padding: 10px 15px;
            margin-top: 10px;
        }
        
        .watchlist h5 {
            font-size: 9pt;
            color: var(--warning);
            margin-bottom: 6px;
        }
        
        .watchlist li {
            font-size: 9pt;
        }
        
        .footer {
            margin-top: 30px;
            padding-top: 15px;
            border-top: 1px solid var(--border);
            font-size: 8pt;
            color: var(--text-dim);
            text-align: center;
        }
        
        @media print {
            body {
                max-width: none;
                padding: 0;
            }
            .section {
                page-break-inside: avoid;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Daily Intelligence Digest</h1>
        <div class="date">{{ date | format_date }} | Generated {{ generated_at | format_time }}</div>
    </div>
    
    {% if executive_brief %}
    <div class="executive-brief">
        <h2>
            🎯 Executive Summary
            {% if executive_brief.risk_tone %}
            <span class="risk-badge" style="background: {{ executive_brief.risk_tone | risk_color }}">
                {{ executive_brief.risk_tone }}
            </span>
            {% endif %}
        </h2>
        
        {% if executive_brief.top_takeaways %}
        <ul class="takeaways">
            {% for takeaway in executive_brief.top_takeaways %}
            <li>{{ takeaway }}</li>
            {% endfor %}
        </ul>
        {% endif %}
        
        {% if executive_brief.todays_themes %}
        <div class="themes">
            {% for theme in executive_brief.todays_themes %}
            <span class="theme-chip">{{ theme }}</span>
            {% endfor %}
        </div>
        {% endif %}
    </div>
    {% endif %}
    
    {% if market_snapshot %}
    <div class="market-snapshot">
        <h3>📈 Market Snapshot</h3>
        <div class="market-grid">
            {% if market_snapshot.spx %}
            <div class="market-item">
                <div class="label">S&P 500</div>
                <div class="value">{{ market_snapshot.spx | format_price(0) }}</div>
                <div class="change {{ 'positive' if market_snapshot.spx_change >= 0 else 'negative' }}">
                    {{ market_snapshot.spx_change | format_pct(2) }}
                </div>
            </div>
            {% endif %}
            {% if market_snapshot.ust_10y %}
            <div class="market-item">
                <div class="label">10Y Yield</div>
                <div class="value">{{ market_snapshot.ust_10y | format_price(2) }}%</div>
            </div>
            {% endif %}
            {% if market_snapshot.dxy %}
            <div class="market-item">
                <div class="label">DXY</div>
                <div class="value">{{ market_snapshot.dxy | format_price(2) }}</div>
                <div class="change {{ 'positive' if market_snapshot.dxy_change >= 0 else 'negative' }}">
                    {{ market_snapshot.dxy_change | format_pct(2) }}
                </div>
            </div>
            {% endif %}
            {% if market_snapshot.brent %}
            <div class="market-item">
                <div class="label">Brent</div>
                <div class="value">${{ market_snapshot.brent | format_price(2) }}</div>
                <div class="change {{ 'positive' if market_snapshot.brent_change >= 0 else 'negative' }}">
                    {{ market_snapshot.brent_change | format_pct(2) }}
                </div>
            </div>
            {% endif %}
            {% if market_snapshot.gold %}
            <div class="market-item">
                <div class="label">Gold</div>
                <div class="value">${{ market_snapshot.gold | format_price(0) }}</div>
                <div class="change {{ 'positive' if market_snapshot.gold_change >= 0 else 'negative' }}">
                    {{ market_snapshot.gold_change | format_pct(2) }}
                </div>
            </div>
            {% endif %}
        </div>
    </div>
    {% endif %}
    
    {% for section in sections %}
    {% if section.items and section.summary %}
    <div class="section">
        <div class="section-header">
            <span class="section-icon">{{ section.icon }}</span>
            <span class="section-title">{{ section.display_name }}</span>
            <span class="impact-badge">Impact: {{ (section.max_impact_score * 100) | int }}%</span>
        </div>
        
        <div class="summary-block">
            <h4>What Happened</h4>
            <ul>
                {% for bullet in section.summary.what_happened %}
                <li>{{ bullet }}</li>
                {% endfor %}
            </ul>
        </div>
        
        {% if section.summary.why_it_matters %}
        <div class="summary-block">
            <h4>Why It Matters</h4>
            <ul>
                {% for bullet in section.summary.why_it_matters %}
                <li>{{ bullet }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}
        
        <div class="market-impacts">
            <div class="impact-column">
                <h5>🇺🇸 US Market Impact</h5>
                <ul>
                    {% for bullet in section.summary.market_impact_us.bullets %}
                    <li>{{ bullet }}</li>
                    {% endfor %}
                </ul>
            </div>
            <div class="impact-column">
                <h5>🏛️ GCC Market Impact</h5>
                <ul>
                    {% for bullet in section.summary.market_impact_gcc.bullets %}
                    <li>{{ bullet }}</li>
                    {% endfor %}
                </ul>
            </div>
        </div>
        
        {% if section.summary.watchlist %}
        <div class="watchlist">
            <h5>👀 Watchlist</h5>
            <ul>
                {% for item in section.summary.watchlist %}
                <li>{{ item }}</li>
                {% endfor %}
            </ul>
        </div>
        {% endif %}
        
        <div class="articles-list">
            <strong style="font-size: 9pt; color: var(--text-dim);">Source Articles:</strong>
            {% for item in section.items %}
            <div class="article-item">
                <span class="article-time">{{ item.timestamp.strftime('%H:%M') }}</span>
                <span class="article-headline">
                    {% if item.url %}
                    <a href="{{ item.url }}" target="_blank">{{ item.headline[:80] }}{% if item.headline|length > 80 %}...{% endif %}</a>
                    {% else %}
                    {{ item.headline[:80] }}{% if item.headline|length > 80 %}...{% endif %}
                    {% endif %}
                    <span class="source-tag">[{{ item.source }}]</span>
                </span>
            </div>
            {% endfor %}
        </div>
    </div>
    {% endif %}
    {% endfor %}
    
    <div class="footer">
        <p>Daily Intelligence Digest | {{ date }} | Version {{ metadata.version }}</p>
        <p>Generated in {{ '%.1f' | format(metadata.generation_time_seconds) }}s | 
           {{ metadata.total_items_selected }}/{{ metadata.total_items_processed }} items selected</p>
        <p style="margin-top: 5px; font-style: italic;">
            This report is for informational purposes only. Not investment advice.
        </p>
    </div>
</body>
</html>'''


def render_digest(
    digest: DigestOutput,
    output_path: Optional[str] = None,
) -> str:
    """
    Convenience function to render a digest.
    
    Args:
        digest: DigestOutput to render
        output_path: Optional path to save HTML
        
    Returns:
        Rendered HTML string
    """
    renderer = DigestRenderer()
    return renderer.render(digest, output_path)
