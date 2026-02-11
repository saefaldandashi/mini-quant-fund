#!/usr/bin/env python3
"""
Analyze Daily Loss - Diagnose why daily P&L is negative.

This script helps understand what caused a daily loss by:
1. Checking current positions and their P/L
2. Analyzing transaction costs
3. Checking borrow costs for shorts
4. Identifying which positions are losing money
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from broker_alpaca import AlpacaBroker

def analyze_daily_loss():
    """Analyze why daily P&L is negative."""
    
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        print("❌ Error: ALPACA_API_KEY and ALPACA_SECRET_KEY not set")
        return
    
    broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=True)
    
    print("=" * 70)
    print("📊 DAILY LOSS ANALYSIS")
    print("=" * 70)
    print()
    
    # Get account info
    account = broker.get_account()
    equity = float(account.get("equity", 0))
    cash = float(account.get("cash", 0))
    buying_power = float(account.get("buying_power", 0))
    
    print(f"💰 Account Equity: ${equity:,.2f}")
    print(f"💵 Cash: ${cash:,.2f}")
    print(f"💳 Buying Power: ${buying_power:,.2f}")
    print()
    
    # Get positions
    positions = broker.get_positions()
    
    if not positions:
        print("⚠️ No open positions")
        return
    
    print(f"📈 Open Positions: {len(positions)}")
    print()
    
    # Analyze each position
    total_unrealized_pnl = 0
    losing_positions = []
    winning_positions = []
    
    print("=" * 70)
    print("POSITION BREAKDOWN")
    print("=" * 70)
    print(f"{'Symbol':<8} {'Side':<6} {'Qty':<8} {'Entry':<10} {'Current':<10} {'P/L $':<12} {'P/L %':<10}")
    print("-" * 70)
    
    for symbol, pos in sorted(positions.items()):
        qty = pos.get("qty", 0)
        avg_entry = pos.get("avg_entry_price", 0)
        current_price = pos.get("current_price", 0)
        pnl = pos.get("pnl", 0)
        pnl_pct = pos.get("pnl_pct", 0)
        
        side = "LONG" if qty > 0 else "SHORT"
        qty_display = f"{qty:+.0f}" if abs(qty) >= 1 else f"{qty:+.2f}"
        
        total_unrealized_pnl += pnl
        
        if pnl < 0:
            losing_positions.append((symbol, pnl, pnl_pct))
        else:
            winning_positions.append((symbol, pnl, pnl_pct))
        
        # Color code: red for losses, green for gains
        pnl_str = f"${pnl:+,.2f}" if abs(pnl) >= 0.01 else "$0.00"
        pnl_pct_str = f"{pnl_pct:+.2f}%" if abs(pnl_pct) >= 0.01 else "0.00%"
        
        print(f"{symbol:<8} {side:<6} {qty_display:<8} ${avg_entry:<9.2f} ${current_price:<9.2f} {pnl_str:<12} {pnl_pct_str:<10}")
    
    print("-" * 70)
    print(f"{'TOTAL UNREALIZED P/L':<50} ${total_unrealized_pnl:+,.2f}")
    print()
    
    # Analyze biggest losers
    if losing_positions:
        losing_positions.sort(key=lambda x: x[1])  # Sort by P/L (most negative first)
        print("=" * 70)
        print("🔴 BIGGEST LOSERS")
        print("=" * 70)
        for symbol, pnl, pnl_pct in losing_positions[:10]:
            print(f"  {symbol}: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
        print()
    
    # Analyze biggest winners
    if winning_positions:
        winning_positions.sort(key=lambda x: x[1], reverse=True)  # Sort by P/L (highest first)
        print("=" * 70)
        print("🟢 BIGGEST WINNERS")
        print("=" * 70)
        for symbol, pnl, pnl_pct in winning_positions[:10]:
            print(f"  {symbol}: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
        print()
    
    # Check short positions (borrow costs)
    short_positions = [(s, p) for s, p in positions.items() if p.get("qty", 0) < 0]
    if short_positions:
        print("=" * 70)
        print("📉 SHORT POSITIONS (Borrow Costs)")
        print("=" * 70)
        total_short_value = sum(abs(p.get("market_value", 0)) for p in short_positions)
        print(f"  Total Short Value: ${total_short_value:,.2f}")
        print(f"  Estimated Daily Borrow Cost: ~${total_short_value * 0.0001:,.2f} (0.01% daily)")
        print()
    
    # Check transaction costs from recent trades
    print("=" * 70)
    print("💡 POSSIBLE CAUSES OF -$505.26 LOSS")
    print("=" * 70)
    print()
    print("1. 📉 Market Moves Against Positions:")
    print(f"   - Unrealized P/L from open positions: ${total_unrealized_pnl:+,.2f}")
    if total_unrealized_pnl < -400:
        print("   ⚠️ This is likely the main cause!")
    print()
    
    print("2. 💸 Transaction Costs:")
    print("   - Commissions from rebalancing")
    print("   - Bid-ask spreads (especially during extended hours)")
    print("   - Estimated: $50-200 per rebalance")
    print()
    
    if short_positions:
        print("3. 📉 Borrow Costs (Short Positions):")
        print(f"   - Daily borrow cost on ${total_short_value:,.2f} in shorts")
        print(f"   - Estimated: ~${total_short_value * 0.0001:,.2f} per day")
        print()
    
    print("4. 📊 Realized Losses:")
    print("   - Closed positions that were sold at a loss")
    print("   - Check recent trade history")
    print()
    
    # Calculate breakdown
    print("=" * 70)
    print("📊 LOSS BREAKDOWN ESTIMATE")
    print("=" * 70)
    print(f"  Unrealized P/L: ${total_unrealized_pnl:+,.2f}")
    estimated_costs = 100  # Rough estimate
    estimated_borrow = total_short_value * 0.0001 if short_positions else 0
    estimated_realized = -505.26 - total_unrealized_pnl - estimated_costs - estimated_borrow
    
    print(f"  Estimated Transaction Costs: ${estimated_costs:+,.2f}")
    if short_positions:
        print(f"  Estimated Borrow Costs: ${estimated_borrow:+,.2f}")
    print(f"  Estimated Realized Losses: ${estimated_realized:+,.2f}")
    print(f"  Total: ${total_unrealized_pnl + estimated_costs + estimated_borrow + estimated_realized:+,.2f}")
    print()
    
    # Recommendations
    print("=" * 70)
    print("💡 RECOMMENDATIONS")
    print("=" * 70)
    print()
    
    if losing_positions:
        biggest_loser = losing_positions[0]
        print(f"1. Review {biggest_loser[0]} position:")
        print(f"   - Currently losing ${abs(biggest_loser[1]):,.2f} ({biggest_loser[2]:+.2f}%)")
        print(f"   - Consider if the thesis is still valid")
        print()
    
    if total_unrealized_pnl < -400:
        print("2. Market moved against your positions:")
        print("   - This is normal market volatility")
        print("   - Consider if positions are still aligned with your strategy")
        print()
    
    print("3. Monitor transaction costs:")
    print("   - Extended hours trading has wider spreads")
    print("   - Consider trading during regular hours when possible")
    print()
    
    if short_positions:
        print("4. Short positions have ongoing borrow costs:")
        print("   - These accumulate daily")
        print("   - Factor into position sizing decisions")
        print()

if __name__ == "__main__":
    analyze_daily_loss()
