"""
Report Chart Generator - Creates publication-quality charts for reports.

Generates 5 core charts:
1. Equity curve with benchmark
2. Drawdown chart
3. Holdings breakdown
4. Strategy weights
5. Attribution waterfall
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import base64
from io import BytesIO

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .live_collectors import ReportData, Position, StrategyPerformance


# Dark theme matching the app UI
DARK_THEME = {
    'bg': '#0a0a0a',
    'bg_card': '#1a1a1a',
    'text': '#ffffff',
    'text_muted': '#888888',
    'primary': '#00d4aa',  # Cyan
    'secondary': '#a855f7',  # Purple
    'success': '#22c55e',
    'danger': '#ef4444',
    'warning': '#f59e0b',
    'neutral': '#6b7280',
    'grid': '#333333',
}


def _apply_dark_theme():
    """Apply dark theme to matplotlib."""
    plt.rcParams.update({
        'figure.facecolor': DARK_THEME['bg_card'],
        'axes.facecolor': DARK_THEME['bg_card'],
        'axes.edgecolor': DARK_THEME['grid'],
        'axes.labelcolor': DARK_THEME['text'],
        'text.color': DARK_THEME['text'],
        'xtick.color': DARK_THEME['text_muted'],
        'ytick.color': DARK_THEME['text_muted'],
        'grid.color': DARK_THEME['grid'],
        'legend.facecolor': DARK_THEME['bg_card'],
        'legend.edgecolor': DARK_THEME['grid'],
        'figure.figsize': (10, 5),
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
    })


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string for embedding."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight',
                facecolor=DARK_THEME['bg_card'], edgecolor='none')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


class ReportChartGenerator:
    """Generates charts for reports."""
    
    def __init__(self):
        """Initialize chart generator."""
        _apply_dark_theme()
    
    def generate_all_charts(self, data: ReportData) -> Dict[str, str]:
        """
        Generate all charts for report.
        
        Args:
            data: ReportData object
        
        Returns:
            Dict of chart_name -> base64 image data
        """
        charts = {}
        
        # 1. Equity curve with benchmark
        if len(data.equity_curve) > 0:
            charts['equity_curve'] = self.equity_curve_chart(
                data.equity_curve,
                data.benchmark_curve,
                data.portfolio.portfolio_value,
            )
        
        # 2. Drawdown chart
        if len(data.drawdown_curve) > 0:
            charts['drawdown'] = self.drawdown_chart(data.drawdown_curve)
        
        # 3. Holdings breakdown
        if data.positions:
            charts['holdings'] = self.holdings_chart(data.positions)
        
        # 4. Strategy weights
        if data.strategy_performance:
            charts['strategy_weights'] = self.strategy_weights_chart(data.strategy_performance)
        
        # 5. P/L by position
        if data.positions:
            charts['pnl_breakdown'] = self.pnl_breakdown_chart(data.positions)
        
        return charts
    
    def equity_curve_chart(
        self,
        equity: pd.Series,
        benchmark: pd.Series,
        current_value: float,
    ) -> str:
        """
        Generate equity curve chart with benchmark overlay.
        
        Shows portfolio value vs benchmark (SPY) over time.
        """
        _apply_dark_theme()
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Normalize both to start at 100
        if len(equity) > 0 and equity.iloc[0] > 0:
            equity_norm = (equity / equity.iloc[0]) * 100
            ax.plot(equity.index, equity_norm.values,
                   color=DARK_THEME['primary'], linewidth=2, label='Portfolio')
        
        if len(benchmark) > 0 and benchmark.iloc[0] > 0:
            benchmark_norm = (benchmark / benchmark.iloc[0]) * 100
            ax.plot(benchmark.index, benchmark_norm.values,
                   color=DARK_THEME['secondary'], linewidth=1.5,
                   linestyle='--', label='SPY Benchmark', alpha=0.7)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Value (Normalized to 100)')
        ax.set_title(f'Portfolio Performance vs Benchmark | Current: ${current_value:,.0f}',
                    fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        if len(equity) > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add performance annotation
        if len(equity) > 1:
            total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0] * 100
            color = DARK_THEME['success'] if total_return >= 0 else DARK_THEME['danger']
            ax.annotate(f'{total_return:+.1f}%', 
                       xy=(0.98, 0.95), xycoords='axes fraction',
                       fontsize=14, fontweight='bold', color=color,
                       ha='right', va='top')
        
        plt.tight_layout()
        return _fig_to_base64(fig)
    
    def drawdown_chart(self, drawdown: pd.Series) -> str:
        """
        Generate drawdown chart.
        
        Shows underwater equity curve (distance from peak).
        """
        _apply_dark_theme()
        fig, ax = plt.subplots(figsize=(12, 4))
        
        drawdown_pct = drawdown * 100
        
        ax.fill_between(drawdown.index, 0, drawdown_pct.values,
                       color=DARK_THEME['danger'], alpha=0.6)
        ax.plot(drawdown.index, drawdown_pct.values,
               color=DARK_THEME['danger'], linewidth=1)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.set_title('Drawdown (Distance from Peak)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color=DARK_THEME['text_muted'], linestyle='-', linewidth=0.5)
        
        # Format y-axis to show negative values
        ax.set_ylim(top=0)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add max drawdown annotation
        max_dd = drawdown_pct.min()
        ax.annotate(f'Max: {max_dd:.1f}%',
                   xy=(0.98, 0.05), xycoords='axes fraction',
                   fontsize=12, fontweight='bold', color=DARK_THEME['danger'],
                   ha='right', va='bottom')
        
        plt.tight_layout()
        return _fig_to_base64(fig)
    
    def holdings_chart(self, positions: List[Position]) -> str:
        """
        Generate holdings breakdown chart.
        
        Horizontal bar chart showing position weights.
        """
        _apply_dark_theme()
        fig, ax = plt.subplots(figsize=(10, max(4, len(positions) * 0.4)))
        
        # Limit to top 15 positions
        top_positions = positions[:15]
        
        symbols = [p.symbol for p in top_positions]
        weights = [p.weight * 100 for p in top_positions]
        
        # Color by P/L
        colors = [
            DARK_THEME['success'] if p.unrealized_pnl >= 0 else DARK_THEME['danger']
            for p in top_positions
        ]
        
        y_pos = np.arange(len(symbols))
        bars = ax.barh(y_pos, weights, color=colors, alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(symbols)
        ax.set_xlabel('Portfolio Weight (%)')
        ax.set_title('Holdings Breakdown by Weight', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add weight labels
        for bar, weight in zip(bars, weights):
            ax.text(weight + 0.3, bar.get_y() + bar.get_height()/2,
                   f'{weight:.1f}%', va='center', fontsize=9,
                   color=DARK_THEME['text'])
        
        ax.invert_yaxis()  # Highest at top
        plt.tight_layout()
        return _fig_to_base64(fig)
    
    def strategy_weights_chart(self, strategies: List[StrategyPerformance]) -> str:
        """
        Generate strategy weights pie/donut chart.
        """
        _apply_dark_theme()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        if not strategies:
            ax1.text(0.5, 0.5, 'No Strategy Data', ha='center', va='center', 
                     color=DARK_THEME['text_muted'], fontsize=12)
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')
            ax2.text(0.5, 0.5, 'Run a rebalance to generate\nstrategy debate scores', 
                     ha='center', va='center', color=DARK_THEME['text_muted'], fontsize=12)
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
            plt.tight_layout()
            return _fig_to_base64(fig)
        
        names = [s.name[:15] for s in strategies]
        weights = [max(s.weight, 0.01) for s in strategies]
        scores = [s.debate_score for s in strategies]
        
        # Colors gradient from primary to secondary
        n = len(strategies)
        colors = [plt.cm.viridis(i/n) for i in range(n)]
        
        # Donut chart for weights
        wedges, texts, autotexts = ax1.pie(
            weights, labels=names, autopct='%1.0f%%',
            colors=colors, pctdistance=0.75,
            wedgeprops=dict(width=0.5, edgecolor=DARK_THEME['bg_card'])
        )
        
        # Style the text
        for text in texts:
            text.set_color(DARK_THEME['text_muted'])
            text.set_fontsize(8)
        for autotext in autotexts:
            autotext.set_color(DARK_THEME['text'])
            autotext.set_fontsize(8)
        
        ax1.set_title('Strategy Weights', fontweight='bold')
        
        # Bar chart for debate scores - handle all-zero case
        has_debate_scores = any(s > 0 for s in scores)
        y_pos = np.arange(len(names))
        
        if has_debate_scores:
            bars = ax2.barh(y_pos, scores, color=colors, alpha=0.8)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(names)
            ax2.set_xlabel('Debate Score')
            ax2.set_title('Strategy Debate Scores', fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            ax2.invert_yaxis()
        else:
            # Show message that debate hasn't run yet
            ax2.text(0.5, 0.5, 'No debate scores yet.\nRun a rebalance to\ngenerate debate analysis.', 
                     ha='center', va='center', color=DARK_THEME['text_muted'], fontsize=11,
                     transform=ax2.transAxes, linespacing=1.5)
            ax2.set_title('Strategy Debate Scores', fontweight='bold')
            ax2.axis('off')
        
        plt.tight_layout()
        return _fig_to_base64(fig)
    
    def pnl_breakdown_chart(self, positions: List[Position]) -> str:
        """
        Generate P/L breakdown waterfall chart.
        """
        _apply_dark_theme()
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Sort by P/L
        sorted_positions = sorted(positions, key=lambda p: p.unrealized_pnl, reverse=True)
        
        # Limit to top/bottom 10
        if len(sorted_positions) > 15:
            top_5 = sorted_positions[:7]
            bottom_5 = sorted_positions[-7:]
            positions_to_show = top_5 + bottom_5
        else:
            positions_to_show = sorted_positions
        
        symbols = [p.symbol for p in positions_to_show]
        pnls = [p.unrealized_pnl for p in positions_to_show]
        
        colors = [
            DARK_THEME['success'] if pnl >= 0 else DARK_THEME['danger']
            for pnl in pnls
        ]
        
        x_pos = np.arange(len(symbols))
        bars = ax.bar(x_pos, pnls, color=colors, alpha=0.8)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(symbols, rotation=45, ha='right')
        ax.set_ylabel('Unrealized P/L ($)')
        ax.set_title('P/L by Position', fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color=DARK_THEME['text_muted'], linestyle='-', linewidth=1)
        
        # Add total annotation
        total_pnl = sum(pnls)
        color = DARK_THEME['success'] if total_pnl >= 0 else DARK_THEME['danger']
        ax.annotate(f'Total: ${total_pnl:+,.0f}',
                   xy=(0.98, 0.95), xycoords='axes fraction',
                   fontsize=12, fontweight='bold', color=color,
                   ha='right', va='top')
        
        plt.tight_layout()
        return _fig_to_base64(fig)
    
    def volatility_chart(self, history: pd.Series, window: int = 20) -> str:
        """
        Generate rolling volatility chart.
        """
        _apply_dark_theme()
        fig, ax = plt.subplots(figsize=(12, 4))
        
        if len(history) < window:
            return ""
        
        returns = history.pct_change().dropna()
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
        
        ax.plot(rolling_vol.index, rolling_vol.values,
               color=DARK_THEME['warning'], linewidth=1.5)
        ax.fill_between(rolling_vol.index, 0, rolling_vol.values,
                       color=DARK_THEME['warning'], alpha=0.3)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Volatility (% Annualized)')
        ax.set_title(f'{window}-Day Rolling Volatility', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return _fig_to_base64(fig)
