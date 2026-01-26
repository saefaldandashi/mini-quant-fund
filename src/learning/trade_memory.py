"""
Trade Memory - Persistent storage of all trades with full context.

Stores:
- Trade details (symbol, side, quantity, price, timestamp)
- Market context at time of trade (regime, volatility, sentiment)
- Strategy signals that led to the trade
- Outcomes (P/L, holding period, exit reason)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd


@dataclass
class StrategySignalSnapshot:
    """Snapshot of a strategy's signal at trade time."""
    strategy_name: str
    weight_proposed: float
    confidence: float
    expected_return: float
    debate_score: float
    explanation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketContextSnapshot:
    """Market conditions at time of trade."""
    regime: str  # 'risk_on', 'risk_off', 'neutral'
    volatility_regime: str  # 'low', 'medium', 'high'
    trend_strength: float
    correlation_regime: str  # 'low', 'high'
    spy_return_1d: float
    spy_return_5d: float
    vix_level: Optional[float] = None
    sector_momentum: Dict[str, float] = field(default_factory=dict)


@dataclass
class TradeRecord:
    """Complete record of a single trade with all context."""
    # Trade identifiers
    trade_id: str
    timestamp: str
    
    # Trade details
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    entry_price: float
    
    # Strategy context
    strategy_signals: List[StrategySignalSnapshot] = field(default_factory=list)
    ensemble_weight: float = 0.0
    ensemble_mode: str = ''
    
    # Market context
    market_context: Optional[MarketContextSnapshot] = None
    
    # Leverage tracking (for leveraged trades)
    leverage_used: float = 1.0          # Leverage at time of trade
    margin_cost_daily: float = 0.0      # Daily margin interest cost
    was_leveraged: bool = False         # True if leverage > 1.0
    leverage_state: str = 'healthy'     # Leverage state at entry
    
    # Holding period tracking (for auto-exit)
    intended_holding_minutes: int = 0   # Strategy's intended holding period (0 = no limit)
    entry_strategy: str = ''            # Primary strategy that triggered this trade
    should_auto_exit: bool = False      # True if position exceeded intended holding
    
    # Outcome tracking (filled in later)
    exit_price: Optional[float] = None
    exit_timestamp: Optional[str] = None
    exit_reason: Optional[str] = None  # 'rebalance', 'stop_loss', 'take_profit', 'regime_change'
    holding_period_days: Optional[int] = None
    pnl_dollars: Optional[float] = None
    pnl_percent: Optional[float] = None
    
    # Learning labels
    was_profitable: Optional[bool] = None
    mistake_category: Optional[str] = None  # 'timing', 'sizing', 'direction', 'none'
    lessons_learned: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert nested dataclasses
        if self.market_context:
            data['market_context'] = asdict(self.market_context)
        data['strategy_signals'] = [asdict(s) for s in self.strategy_signals]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TradeRecord':
        """Create from dictionary."""
        # Reconstruct nested objects
        if data.get('market_context'):
            data['market_context'] = MarketContextSnapshot(**data['market_context'])
        if data.get('strategy_signals'):
            data['strategy_signals'] = [
                StrategySignalSnapshot(**s) for s in data['strategy_signals']
            ]
        return cls(**data)


class TradeMemory:
    """
    Persistent storage for trade history with learning context.
    
    Features:
    - Stores all trades with full context
    - Tracks open positions for outcome updates
    - Provides query interface for learning algorithms
    - Persists to JSON for durability
    """
    
    def __init__(self, storage_path: str = "outputs/trade_memory.json"):
        self.storage_path = Path(storage_path)
        self.trades: List[TradeRecord] = []
        self.open_positions: Dict[str, TradeRecord] = {}  # symbol -> trade
        self._load()
    
    def _load(self):
        """Load trade history from disk with corruption handling."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                self.trades = [TradeRecord.from_dict(t) for t in data.get('trades', [])]
                
                # Rebuild open positions
                for trade in self.trades:
                    if trade.exit_price is None and trade.side == 'buy':
                        self.open_positions[trade.symbol] = trade
                
                logging.info(f"Loaded {len(self.trades)} trades from memory")
            except json.JSONDecodeError as e:
                logging.error(f"Trade memory JSON corrupted: {e}")
                # Create backup of corrupted file
                try:
                    import shutil
                    backup_path = str(self.storage_path) + f".corrupted.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    shutil.copy(self.storage_path, backup_path)
                    logging.warning(f"Corrupted trade memory backed up to: {backup_path}")
                    logging.warning("Starting fresh with empty trade history")
                except Exception as backup_err:
                    logging.error(f"Could not backup corrupted file: {backup_err}")
                self.trades = []
            except Exception as e:
                logging.warning(f"Could not load trade memory: {e}")
                self.trades = []
    
    def _save(self):
        """Persist trade history to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.storage_path, 'w') as f:
                json.dump({
                    'trades': [t.to_dict() for t in self.trades],
                    'last_updated': datetime.now().isoformat(),
                    'total_trades': len(self.trades),
                }, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Could not save trade memory: {e}")
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        strategy_signals: List[Dict],
        ensemble_weight: float,
        ensemble_mode: str,
        market_context: Dict,
        leverage_used: float = 1.0,
        margin_cost_daily: float = 0.0,
        leverage_state: str = 'healthy',
        intended_holding_minutes: int = 0,
        entry_strategy: str = '',
    ) -> TradeRecord:
        """
        Record a new trade with full context.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            price: Execution price
            strategy_signals: List of strategy signal dicts
            ensemble_weight: Final ensemble weight for this symbol
            ensemble_mode: Ensemble mode used
            market_context: Market conditions dict
            leverage_used: Leverage ratio at trade time
            margin_cost_daily: Daily margin interest cost
            leverage_state: Leverage manager state at trade time
            intended_holding_minutes: Expected holding period for auto-exit
            entry_strategy: Primary strategy that triggered trade
        
        Returns:
            The created TradeRecord
        """
        trade_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convert strategy signals
        signals = []
        for sig in strategy_signals:
            signals.append(StrategySignalSnapshot(
                strategy_name=sig.get('name', 'unknown'),
                weight_proposed=sig.get('weight', 0.0),
                confidence=sig.get('confidence', 0.0),
                expected_return=sig.get('expected_return', 0.0),
                debate_score=sig.get('debate_score', 0.0),
                explanation=sig.get('explanation', {}),
            ))
        
        # Convert market context
        ctx = MarketContextSnapshot(
            regime=market_context.get('regime', 'unknown'),
            volatility_regime=market_context.get('volatility_regime', 'unknown'),
            trend_strength=market_context.get('trend_strength', 0.0),
            correlation_regime=market_context.get('correlation_regime', 'unknown'),
            spy_return_1d=market_context.get('spy_return_1d', 0.0),
            spy_return_5d=market_context.get('spy_return_5d', 0.0),
            vix_level=market_context.get('vix_level'),
            sector_momentum=market_context.get('sector_momentum', {}),
        )
        
        trade = TradeRecord(
            trade_id=trade_id,
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=price,
            strategy_signals=signals,
            ensemble_weight=ensemble_weight,
            ensemble_mode=ensemble_mode,
            market_context=ctx,
            leverage_used=leverage_used,
            margin_cost_daily=margin_cost_daily,
            was_leveraged=leverage_used > 1.0,
            leverage_state=leverage_state,
            intended_holding_minutes=intended_holding_minutes,
            entry_strategy=entry_strategy,
        )
        
        self.trades.append(trade)
        
        # Track open positions for buys
        if side == 'buy':
            self.open_positions[symbol] = trade
        elif side == 'sell' and symbol in self.open_positions:
            # Close the position
            open_trade = self.open_positions[symbol]
            self._close_position(open_trade, price, 'rebalance')
            del self.open_positions[symbol]
        
        self._save()
        logging.info(f"Recorded trade: {side} {quantity} {symbol} @ ${price:.2f}")
        
        return trade
    
    def _close_position(self, trade: TradeRecord, exit_price: float, exit_reason: str):
        """Update a trade with exit information, including leverage-adjusted P/L."""
        trade.exit_price = exit_price
        trade.exit_timestamp = datetime.now().isoformat()
        trade.exit_reason = exit_reason
        
        # Calculate holding period first (needed for margin cost)
        entry_dt = datetime.fromisoformat(trade.timestamp)
        trade.holding_period_days = (datetime.now() - entry_dt).days
        
        # Calculate raw P/L
        if trade.side == 'buy':
            raw_pnl_dollars = (exit_price - trade.entry_price) * trade.quantity
            raw_pnl_percent = (exit_price - trade.entry_price) / trade.entry_price * 100
        else:
            raw_pnl_dollars = (trade.entry_price - exit_price) * trade.quantity
            raw_pnl_percent = (trade.entry_price - exit_price) / trade.entry_price * 100
        
        # Calculate total margin cost for holding period
        if trade.was_leveraged and trade.margin_cost_daily > 0:
            total_margin_cost = trade.margin_cost_daily * max(1, trade.holding_period_days)
            trade.pnl_dollars = raw_pnl_dollars - total_margin_cost
            # Adjust percent to include margin cost
            position_value = trade.entry_price * trade.quantity
            trade.pnl_percent = (trade.pnl_dollars / position_value) * 100 if position_value > 0 else 0
            logging.info(f"Leverage-adjusted P/L for {trade.symbol}: ${raw_pnl_dollars:.2f} - ${total_margin_cost:.2f} margin = ${trade.pnl_dollars:.2f}")
        else:
            trade.pnl_dollars = raw_pnl_dollars
            trade.pnl_percent = raw_pnl_percent
        
        trade.was_profitable = trade.pnl_dollars > 0
        
        # === MISTAKE CLASSIFICATION ===
        # Analyze what went wrong for learning
        if not trade.was_profitable:
            trade.mistake_category, trade.lessons_learned = self._classify_mistake(trade)
            logging.info(f"Trade mistake classification for {trade.symbol}: {trade.mistake_category}")
        else:
            trade.mistake_category = 'none'
            trade.lessons_learned = []
        
        self._save()
    
    def _classify_mistake(self, trade: 'TradeRecord') -> Tuple[str, List[str]]:
        """
        Classify what type of mistake led to a losing trade.
        
        Categories:
        - timing: Entered too early or too late
        - sizing: Position too large or too small
        - direction: Bet against the trend
        - regime: Wrong strategy for market conditions
        - execution: Poor fill price increased losses
        - conviction: Low confidence position that failed
        
        Returns:
            (mistake_category, lessons_learned)
        """
        lessons = []
        
        # Get holding period context
        holding_days = trade.holding_period_days or 0
        pnl_pct = trade.pnl_percent or 0
        
        # === TIMING MISTAKES ===
        # Very short hold with loss = entered too late / chased
        if holding_days == 0 and pnl_pct < -1:
            lessons.append("Avoid chasing momentum; wait for pullback entries")
            return ('timing', lessons)
        
        # Long hold with small loss = held too long
        if holding_days > 5 and -3 < pnl_pct < 0:
            lessons.append("Consider tighter stop-losses for intraday strategies")
            return ('timing', lessons)
        
        # === DIRECTION MISTAKES ===
        # Large loss quickly = wrong direction entirely
        if holding_days <= 1 and pnl_pct < -3:
            lessons.append("Direction was wrong; review signals for false positives")
            
            # Check if sentiment disagreed
            if trade.market_context:
                regime = trade.market_context.regime
                if trade.side == 'buy' and regime in ['risk_off', 'crisis']:
                    lessons.append(f"Avoid longs in {regime} regime")
                elif trade.side == 'sell' and regime in ['risk_on', 'euphoria']:
                    lessons.append(f"Avoid shorts in {regime} regime")
            
            return ('direction', lessons)
        
        # === REGIME MISTAKES ===
        # Check if entry strategy was wrong for regime
        if trade.market_context and trade.entry_strategy:
            regime = trade.market_context.regime
            strategy = trade.entry_strategy
            
            # Momentum strategies in mean-reverting markets
            if 'Momentum' in strategy and regime in ['choppy', 'ranging']:
                lessons.append(f"Momentum strategies underperform in {regime} markets")
                return ('regime', lessons)
            
            # Mean reversion in trending markets
            if 'MeanReversion' in strategy and regime in ['trending_up', 'trending_down']:
                lessons.append(f"Mean reversion doesn't work in strong trends")
                return ('regime', lessons)
        
        # === CONVICTION MISTAKES ===
        # Low confidence positions that failed
        low_confidence_signals = [
            s for s in trade.strategy_signals 
            if s.confidence < 0.4 and abs(s.weight_proposed) > 0.01
        ]
        if low_confidence_signals:
            lessons.append("Low-confidence signals should be skipped or sized smaller")
            return ('conviction', lessons)
        
        # === SIZING MISTAKES ===
        # Large position that had outsized loss impact
        if trade.ensemble_weight > 0.1 and pnl_pct < -2:
            lessons.append("Reduce position sizes for higher-risk trades")
            return ('sizing', lessons)
        
        # === EXECUTION MISTAKES ===
        # If we have execution data showing poor fill
        # (This would need execution report data which may not be available)
        
        # Default: unclassified
        lessons.append("Review trade thesis; conditions may have changed unexpectedly")
        return ('other', lessons)
    
    def update_position_prices(self, current_prices: Dict[str, float]):
        """
        Update P/L for open positions (for tracking unrealized gains).
        
        Args:
            current_prices: Dict of symbol -> current price
        """
        for symbol, trade in self.open_positions.items():
            if symbol in current_prices:
                price = current_prices[symbol]
                if trade.side == 'buy':
                    trade.pnl_dollars = (price - trade.entry_price) * trade.quantity
                    trade.pnl_percent = (price - trade.entry_price) / trade.entry_price * 100
                trade.was_profitable = trade.pnl_dollars > 0
        self._save()
    
    def get_recent_trades(self, n: int = 10) -> List[TradeRecord]:
        """
        Get the N most recent trades (both open and closed).
        Used for LLM context to learn from recent outcomes.
        
        Args:
            n: Number of trades to return
            
        Returns:
            List of recent TradeRecords sorted by timestamp descending
        """
        # Sort by timestamp descending
        sorted_trades = sorted(
            self.trades,
            key=lambda t: t.timestamp,
            reverse=True
        )
        return sorted_trades[:n]
    
    def get_closed_trades(self, days: Optional[int] = None) -> List[TradeRecord]:
        """Get all closed trades, optionally filtered by recency."""
        closed = [t for t in self.trades if t.exit_price is not None]
        
        if days:
            cutoff = datetime.now() - timedelta(days=days)
            closed = [
                t for t in closed 
                if datetime.fromisoformat(t.timestamp) > cutoff
            ]
        
        return closed
    
    def get_trades_by_strategy(self, strategy_name: str) -> List[TradeRecord]:
        """Get trades where a specific strategy had significant influence."""
        result = []
        for trade in self.trades:
            for sig in trade.strategy_signals:
                if sig.strategy_name == strategy_name and sig.debate_score > 0.3:
                    result.append(trade)
                    break
        return result
    
    def get_trades_by_regime(self, regime: str) -> List[TradeRecord]:
        """Get trades executed during a specific market regime."""
        return [
            t for t in self.trades
            if t.market_context and t.market_context.regime == regime
        ]
    
    def get_winning_trades(self) -> List[TradeRecord]:
        """Get all profitable closed trades."""
        return [t for t in self.trades if t.was_profitable == True]
    
    def get_losing_trades(self) -> List[TradeRecord]:
        """Get all unprofitable closed trades."""
        return [t for t in self.trades if t.was_profitable == False]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics of trading history."""
        closed = self.get_closed_trades()
        
        if not closed:
            return {
                'total_trades': len(self.trades),
                'closed_trades': 0,
                'open_positions': len(self.open_positions),
                'win_rate': 0.0,
                'avg_pnl_percent': 0.0,
                'total_pnl': 0.0,
            }
        
        winners = [t for t in closed if t.was_profitable]
        total_pnl = sum(t.pnl_dollars or 0 for t in closed)
        avg_pnl = sum(t.pnl_percent or 0 for t in closed) / len(closed)
        
        return {
            'total_trades': len(self.trades),
            'closed_trades': len(closed),
            'open_positions': len(self.open_positions),
            'win_rate': len(winners) / len(closed) if closed else 0.0,
            'avg_pnl_percent': avg_pnl,
            'total_pnl': total_pnl,
            'best_trade': max(closed, key=lambda t: t.pnl_percent or 0).symbol if closed else None,
            'worst_trade': min(closed, key=lambda t: t.pnl_percent or 0).symbol if closed else None,
            'avg_holding_days': sum(t.holding_period_days or 0 for t in closed) / len(closed),
        }
    
    def get_expired_positions(self) -> List[Tuple[str, TradeRecord]]:
        """
        Get positions that have exceeded their intended holding period.
        These should be auto-exited to maintain strategy integrity.
        
        Returns:
            List of (symbol, TradeRecord) tuples for expired positions
        """
        expired = []
        now = datetime.now()
        
        for symbol, trade in self.open_positions.items():
            # Skip if no holding period specified
            if trade.intended_holding_minutes <= 0:
                continue
            
            # Calculate how long position has been held
            try:
                entry_time = datetime.fromisoformat(trade.timestamp.replace('Z', '+00:00'))
                # Handle timezone-naive comparison
                if entry_time.tzinfo:
                    entry_time = entry_time.replace(tzinfo=None)
                
                held_minutes = (now - entry_time).total_seconds() / 60
                
                # Check if exceeded intended holding period
                if held_minutes > trade.intended_holding_minutes:
                    trade.should_auto_exit = True
                    expired.append((symbol, trade))
                    logging.info(
                        f"Position {symbol} exceeded holding period: "
                        f"{held_minutes:.0f}m vs intended {trade.intended_holding_minutes}m"
                    )
            except Exception as e:
                logging.warning(f"Could not check holding period for {symbol}: {e}")
        
        return expired
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert trade history to DataFrame for analysis."""
        if not self.trades:
            return pd.DataFrame()
        
        records = []
        for trade in self.trades:
            record = {
                'trade_id': trade.trade_id,
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side,
                'quantity': trade.quantity,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'pnl_dollars': trade.pnl_dollars,
                'pnl_percent': trade.pnl_percent,
                'was_profitable': trade.was_profitable,
                'holding_days': trade.holding_period_days,
                'ensemble_weight': trade.ensemble_weight,
            }
            
            if trade.market_context:
                record['regime'] = trade.market_context.regime
                record['volatility_regime'] = trade.market_context.volatility_regime
            
            # Add top strategy signals
            for i, sig in enumerate(trade.strategy_signals[:3]):
                record[f'strategy_{i+1}'] = sig.strategy_name
                record[f'strategy_{i+1}_score'] = sig.debate_score
            
            records.append(record)
        
        return pd.DataFrame(records)

    # ==========================================================================
    # CROSS-ASSET CONTEXT TRACKING
    # ==========================================================================
    
    def record_cross_asset_context(
        self,
        timestamp: datetime,
        cross_signals: Dict[str, float],
        market_regime: str = 'unknown',
    ) -> None:
        """
        Record cross-asset signals at time of rebalance for pattern learning.
        """
        cross_asset_path = self.storage_path.parent / "cross_asset_history.json"
        
        try:
            if cross_asset_path.exists():
                with open(cross_asset_path, 'r') as f:
                    history = json.load(f)
            else:
                history = {'records': [], 'summary': {}}
            
            record = {
                'timestamp': timestamp.isoformat(),
                'signals': cross_signals,
                'market_regime': market_regime,
            }
            history['records'].append(record)
            
            if len(history['records']) > 500:
                history['records'] = history['records'][-500:]
            
            for signal_type, value in cross_signals.items():
                if signal_type not in history['summary']:
                    history['summary'][signal_type] = {'count': 0, 'sum': 0.0}
                history['summary'][signal_type]['count'] += 1
                history['summary'][signal_type]['sum'] += value
            
            with open(cross_asset_path, 'w') as f:
                json.dump(history, f, indent=2, default=str)
            
            logging.debug(f"Recorded cross-asset context: {cross_signals}")
            
        except Exception as e:
            logging.warning(f"Could not record cross-asset context: {e}")
    
    def get_cross_asset_history(self, days: int = 30) -> List[Dict]:
        """Get recent cross-asset signal history for pattern analysis."""
        cross_asset_path = self.storage_path.parent / "cross_asset_history.json"
        
        if not cross_asset_path.exists():
            return []
        
        try:
            with open(cross_asset_path, 'r') as f:
                history = json.load(f)
            
            cutoff = datetime.now() - timedelta(days=days)
            return [
                r for r in history.get('records', [])
                if datetime.fromisoformat(r['timestamp']) > cutoff
            ]
        except Exception as e:
            logging.warning(f"Could not load cross-asset history: {e}")
            return []
    
    def analyze_cross_asset_patterns(self) -> Dict[str, Any]:
        """Analyze cross-asset signal patterns for learning insights."""
        recent = self.get_cross_asset_history(days=90)
        if not recent:
            return {'patterns': [], 'insights': []}
        
        patterns = []
        
        oil_signals = [r['signals'].get('oil_signal', 0) for r in recent if 'oil_signal' in r.get('signals', {})]
        if oil_signals:
            avg_oil = sum(oil_signals) / len(oil_signals)
            if avg_oil > 0.1:
                patterns.append({'type': 'oil_bullish_trend', 'strength': avg_oil})
            elif avg_oil < -0.1:
                patterns.append({'type': 'oil_bearish_trend', 'strength': abs(avg_oil)})
        
        return {'patterns': patterns, 'insights': [p.get('type', '') for p in patterns], 'total_records': len(recent)}
