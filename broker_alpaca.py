"""
Alpaca broker adapter with paper trading safety checks.
"""
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

import config


# LIVE BLOCK: URLs that indicate live trading (must be blocked)
LIVE_BASE_URLS = [
    "https://api.alpaca.markets",
    "https://api.alpaca.markets/",
    "api.alpaca.markets",
]


def is_live_url(base_url: str) -> bool:
    """Check if base URL indicates live trading."""
    base_url_lower = base_url.lower().strip()
    
    # First check: if it's explicitly a paper URL, it's safe
    if "paper" in base_url_lower:
        return False
    
    # Second check: if it matches live URLs exactly
    return any(
        base_url_lower == live_url or base_url_lower.startswith(live_url + "/")
        for live_url in LIVE_BASE_URLS
    )


class AlpacaBroker:
    """Alpaca broker adapter for paper trading only."""
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        """
        Initialize Alpaca broker client.
        
        Args:
            api_key: Alpaca API key
            secret_key: Alpaca secret key
            paper: Must be True (paper trading only)
        
        Raises:
            ValueError: If paper=False or live URL detected
        """
        if not paper:
            raise ValueError(
                "SAFETY ERROR: paper=False detected. This bot only supports paper trading. "
                "Set paper=True or use default."
            )
        
        # Get base URL from env or use default
        base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        
        # LIVE BLOCK: Check for live URLs
        if is_live_url(base_url):
            raise ValueError(
                f"SAFETY ERROR: Live trading URL detected: {base_url}. "
                "This bot only supports paper trading. Use paper-api.alpaca.markets"
            )
        
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.base_url = base_url
        
        # Initialize clients
        # Note: When paper=True, TradingClient uses paper URL by default
        # We use url_override only if a custom URL is provided
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
            url_override=base_url if base_url != "https://paper-api.alpaca.markets" else None
        )
        
        self.data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=secret_key
        )
        
        logging.info(f"Alpaca broker initialized (paper={paper}, base_url={base_url})")
    
    def get_account(self) -> Dict:
        """Get account information."""
        account = self.trading_client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
        }
    
    def get_margin_data(self) -> Dict:
        """
        Get comprehensive margin data for leverage management.
        
        Returns:
            Dict with all margin-related fields from Alpaca account
        """
        account = self.trading_client.get_account()
        
        # Calculate gross exposure (sum of absolute position values)
        positions = self.trading_client.get_all_positions()
        long_value = sum(
            float(p.market_value) for p in positions 
            if float(p.qty) > 0
        )
        short_value = sum(
            abs(float(p.market_value)) for p in positions 
            if float(p.qty) < 0
        )
        gross_exposure = long_value + short_value
        net_exposure = long_value - short_value
        
        equity = float(account.equity)
        
        return {
            # Core account values
            "equity": equity,
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            
            # Buying power (margin available)
            "buying_power": float(account.buying_power),
            "regt_buying_power": float(getattr(account, 'regt_buying_power', account.buying_power)),
            "daytrading_buying_power": float(getattr(account, 'daytrading_buying_power', account.buying_power)),
            
            # Margin requirements
            "initial_margin": float(getattr(account, 'initial_margin', 0)),
            "maintenance_margin": float(getattr(account, 'maintenance_margin', 0)),
            "last_maintenance_margin": float(getattr(account, 'last_maintenance_margin', 0)),
            
            # Position values
            "long_market_value": float(getattr(account, 'long_market_value', long_value)),
            "short_market_value": float(getattr(account, 'short_market_value', short_value)),
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            
            # Leverage calculations
            "current_leverage": gross_exposure / equity if equity > 0 else 0,
            "net_leverage": net_exposure / equity if equity > 0 else 0,
            
            # Account status
            "status": str(account.status),
            "trading_blocked": bool(getattr(account, 'trading_blocked', False)),
            "account_blocked": bool(getattr(account, 'account_blocked', False)),
            "pattern_day_trader": bool(getattr(account, 'pattern_day_trader', False)),
            
            # SMA (Special Memorandum Account) - available margin
            "sma": float(getattr(account, 'sma', 0)),
            
            # Multiplier (1 for cash, 2 for RegT margin, 4 for portfolio margin)
            "multiplier": float(getattr(account, 'multiplier', 1)),
        }
    
    def get_positions(self) -> Dict[str, Dict]:
        """
        Get current positions with detailed info including current price and P/L.
        
        Returns:
            Dict mapping symbol to position info (qty, market_value, avg_entry_price, current_price, pnl, pnl_pct)
        """
        positions = self.trading_client.get_all_positions()
        result = {}
        
        if not positions:
            return result
        
        # Get current prices for all positions
        symbols = [pos.symbol for pos in positions]
        current_prices = self.get_current_prices(symbols)
        
        for pos in positions:
            symbol = pos.symbol
            qty = float(pos.qty)
            avg_entry_price = float(pos.avg_entry_price)
            market_value = float(pos.market_value)
            current_price = current_prices.get(symbol, avg_entry_price)
            
            # Calculate P/L
            cost_basis = qty * avg_entry_price
            current_value = qty * current_price
            pnl = current_value - cost_basis
            pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0.0
            
            result[symbol] = {
                "qty": qty,
                "market_value": market_value,
                "avg_entry_price": avg_entry_price,
                "current_price": current_price,
                "cost_basis": cost_basis,
                "current_value": current_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            }
        
        return result
    
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        try:
            clock = self.trading_client.get_clock()
            return clock.is_open
        except Exception as e:
            logging.warning(f"Error checking market status: {e}")
            return False
    
    def get_historical_bars(
        self,
        symbols: List[str],
        days: int,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.Series]:
        """
        Fetch historical daily bars for symbols.
        
        Args:
            symbols: List of symbols to fetch
            days: Number of days of history needed
            end_date: End date (default: today)
        
        Returns:
            Dict mapping symbol to Series of closing prices (sorted by date, ascending)
        """
        if end_date is None:
            end_date = datetime.now()
        
        # Request more calendar days to get enough trading days
        calendar_days = int(days * 1.5) + 60
        start_date = end_date - timedelta(days=calendar_days)
        
        result = {}
        
        try:
            logging.info(f"Fetching data for {len(symbols)} symbols...")
            
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date,
                feed=DataFeed.IEX
            )
            
            bars = self.data_client.get_stock_bars(request)
            
            if bars is None:
                logging.warning("No data returned from API")
                return result
            
            # The BarSet response has a .data attribute which is the dict
            bars_dict = bars.data if hasattr(bars, 'data') else {}
            
            for symbol in symbols:
                try:
                    if symbol in bars_dict:
                        symbol_bars = bars_dict[symbol]
                        
                        # symbol_bars is a list of Bar objects
                        if symbol_bars and len(symbol_bars) > 0:
                            closes = [bar.close for bar in symbol_bars]
                            timestamps = [bar.timestamp for bar in symbol_bars]
                            
                            df = pd.DataFrame({'close': closes}, index=pd.DatetimeIndex(timestamps))
                            df = df.sort_index()
                            result[symbol] = df["close"]
                except Exception as e:
                    logging.debug(f"Error processing {symbol}: {e}")
                    continue
            
            logging.info(f"Fetched data for {len(result)}/{len(symbols)} symbols")
            
        except Exception as e:
            logging.warning(f"Batch fetch failed: {e}")
            logging.info("Falling back to individual symbol fetching...")
            
            for i, symbol in enumerate(symbols):
                if (i + 1) % 10 == 0:
                    logging.info(f"Fetched {i + 1}/{len(symbols)} symbols...")
                try:
                    request = StockBarsRequest(
                        symbol_or_symbols=symbol,
                        timeframe=TimeFrame.Day,
                        start=start_date,
                        end=end_date,
                        feed=DataFeed.IEX
                    )
                    bars = self.data_client.get_stock_bars(request)
                    
                    if bars is None:
                        continue
                    
                    bars_dict = bars.data if hasattr(bars, 'data') else {}
                    
                    if symbol in bars_dict:
                        symbol_bars = bars_dict[symbol]
                        if symbol_bars and len(symbol_bars) > 0:
                            closes = [bar.close for bar in symbol_bars]
                            timestamps = [bar.timestamp for bar in symbol_bars]
                            
                            df = pd.DataFrame({'close': closes}, index=pd.DatetimeIndex(timestamps))
                            df = df.sort_index()
                            result[symbol] = df["close"]
                except Exception as e:
                    continue
            
            logging.info(f"Fetched data for {len(result)}/{len(symbols)} symbols")
        
        return result
    
    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current/latest prices for symbols.
        
        Args:
            symbols: List of symbols
        
        Returns:
            Dict mapping symbol to latest price
        """
        if not symbols:
            return {}
        
        # Use historical bars with recent end date to get latest prices
        bars = self.get_historical_bars(symbols, days=5)
        prices = {}
        
        for symbol in symbols:
            if symbol in bars and len(bars[symbol]) > 0:
                prices[symbol] = float(bars[symbol].iloc[-1])
            else:
                logging.warning(f"Could not get price for {symbol}")
        
        return prices
    
    def calculate_target_shares(
        self,
        target_weights: Dict[str, float],
        equity: float,
        cash_buffer_pct: float = config.CASH_BUFFER_PCT
    ) -> Dict[str, int]:
        """
        Calculate target share quantities given target weights and current prices.
        
        SUPPORTS BOTH LONG AND SHORT POSITIONS:
        - Positive weight = LONG position (buy shares)
        - Negative weight = SHORT position (negative shares in result)
        
        Args:
            target_weights: Dict of symbol -> weight (e.g., 0.10 for 10%, -0.05 for 5% short)
            equity: Total account equity
            cash_buffer_pct: Cash buffer percentage
        
        Returns:
            Dict mapping symbol to target share quantity (positive for long, NEGATIVE for short)
        """
        if not target_weights:
            return {}
        
        symbols = list(target_weights.keys())
        prices = self.get_current_prices(symbols)
        
        investable_equity = equity * (1.0 - cash_buffer_pct)
        
        result = {}
        
        for symbol, weight in target_weights.items():
            # Skip insignificant weights (but check absolute value for shorts!)
            if abs(weight) <= 0.001:
                continue
                
            if symbol not in prices:
                logging.warning(f"No price available for {symbol}, skipping")
                continue
            
            price = prices[symbol]
            if price <= 0:
                logging.warning(f"Invalid price for {symbol}: {price}, skipping")
                continue
            
            # Calculate target notional based on actual weight (preserves sign for shorts)
            target_notional = investable_equity * weight
            
            # For SHORTS: weight is negative, so target_notional is negative
            # We want shares to be NEGATIVE to signal a short position
            if weight > 0:
                # LONG: positive shares
                shares = int(target_notional / price)  # Floor to whole shares
                if shares > 0:
                    result[symbol] = shares
                    logging.info(
                        f"LONG {symbol}: weight={weight:.1%}, target ${target_notional:.2f} / ${price:.2f} = {shares} shares"
                    )
            else:
                # SHORT: negative shares (note: target_notional is already negative)
                shares = int(target_notional / price)  # This will be negative
                if shares < 0:
                    result[symbol] = shares  # NEGATIVE VALUE = SHORT POSITION
                    logging.info(
                        f"SHORT {symbol}: weight={weight:.1%}, target ${target_notional:.2f} / ${price:.2f} = {shares} shares"
                    )
        
        return result
    
    def place_market_order(
        self,
        symbol: str,
        qty: int,
        side: OrderSide,
        dry_run: bool = True
    ) -> Optional[Dict]:
        """
        Place a market order.
        
        Args:
            symbol: Stock symbol
            qty: Quantity (must be > 0)
            side: OrderSide.BUY or OrderSide.SELL
            dry_run: If True, log but don't place order
        
        Returns:
            Order info dict if placed, None if dry_run
        """
        if qty <= 0:
            logging.warning(f"Invalid quantity {qty} for {symbol}, skipping order")
            return None
        
        side_str = "BUY" if side == OrderSide.BUY else "SELL"
        
        if dry_run:
            logging.info(f"[DRY RUN] Would place {side_str} order: {symbol} x {qty}")
            return None
        
        try:
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            order = self.trading_client.submit_order(order_request)
            
            logging.info(
                f"Placed {side_str} order: {symbol} x {qty}, order_id={order.id}, "
                f"status={order.status}"
            )
            
            return {
                "id": order.id,
                "symbol": symbol,
                "qty": qty,
                "side": side_str,
                "status": order.status.value if hasattr(order.status, 'value') else str(order.status)
            }
        except Exception as e:
            logging.error(f"Error placing {side_str} order for {symbol}: {e}")
            raise
    
    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.
        
        Returns:
            Number of orders cancelled
        """
        try:
            # Get all open orders
            orders = self.trading_client.get_orders()
            
            if not orders:
                logging.info("No open orders to cancel")
                return 0
            
            # Cancel all open orders
            cancelled_count = 0
            for order in orders:
                try:
                    self.trading_client.cancel_order_by_id(order.id)
                    logging.info(f"Cancelled order {order.id}: {order.side} {order.qty} {order.symbol}")
                    cancelled_count += 1
                except Exception as e:
                    logging.warning(f"Could not cancel order {order.id}: {e}")
            
            logging.info(f"Cancelled {cancelled_count} orders")
            return cancelled_count
            
        except Exception as e:
            logging.error(f"Error cancelling orders: {e}")
            return 0
    
    def get_open_orders(self) -> List[Dict]:
        """Get all open orders."""
        try:
            orders = self.trading_client.get_orders()
            return [
                {
                    "id": str(o.id),
                    "symbol": o.symbol,
                    "side": o.side.value if hasattr(o.side, 'value') else str(o.side),
                    "qty": float(o.qty),
                    "status": o.status.value if hasattr(o.status, 'value') else str(o.status),
                    "created_at": str(o.created_at),
                }
                for o in orders
            ]
        except Exception as e:
            logging.error(f"Error getting orders: {e}")
            return []
    
    def get_orders(self, status: str = 'all', limit: int = 100):
        """
        Get orders with optional status filter.
        
        Args:
            status: 'all', 'open', 'closed', 'filled'
            limit: Max orders to return
        
        Returns:
            List of order objects
        """
        try:
            from alpaca.trading.enums import QueryOrderStatus
            
            if status == 'all':
                request = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
            elif status == 'filled':
                request = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=limit)
            elif status == 'open':
                request = GetOrdersRequest(status=QueryOrderStatus.OPEN, limit=limit)
            else:
                request = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
            
            orders = self.trading_client.get_orders(filter=request)
            return orders if orders else []
        except Exception as e:
            logging.error(f"Error getting orders: {e}")
            return []
    
    def get_portfolio_history(self, days: int = 30) -> pd.Series:
        """
        Get portfolio equity history using Alpaca Portfolio History API.
        
        This is the CORRECT way to get historical portfolio values for:
        - Equity curve charts
        - Returns calculation
        - Volatility calculation
        - Sharpe ratio
        - Drawdown analysis
        
        Args:
            days: Number of days of history
        
        Returns:
            pd.Series with datetime index and equity values
        """
        import requests
        
        try:
            # Calculate period
            if days <= 7:
                period = "1W"
                timeframe = "1D"
            elif days <= 30:
                period = "1M"
                timeframe = "1D"
            elif days <= 90:
                period = "3M"
                timeframe = "1D"
            elif days <= 365:
                period = "1A"
                timeframe = "1D"
            else:
                period = "all"
                timeframe = "1D"
            
            # Use Alpaca Portfolio History API
            url = f"{self.base_url}/v2/account/portfolio/history"
            params = {
                "period": period,
                "timeframe": timeframe,
                "extended_hours": "false",
            }
            
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code != 200:
                logging.warning(f"Portfolio history API error: {response.status_code} - {response.text}")
                return pd.Series(dtype=float)
            
            data = response.json()
            
            if not data or 'timestamp' not in data or 'equity' not in data:
                logging.warning("Portfolio history API returned empty data")
                return pd.Series(dtype=float)
            
            # Build Series from response
            timestamps = data['timestamp']
            equity_values = data['equity']
            
            if not timestamps or not equity_values:
                return pd.Series(dtype=float)
            
            # Convert timestamps to datetime (Alpaca returns Unix timestamps)
            dates = pd.to_datetime(timestamps, unit='s')
            
            series = pd.Series(equity_values, index=dates)
            series = series.sort_index()
            
            # Filter to requested days
            cutoff = datetime.now() - timedelta(days=days)
            series = series[series.index >= cutoff]
            
            logging.info(f"Portfolio history: {len(series)} data points over {days} days")
            return series
            
        except Exception as e:
            logging.error(f"Error getting portfolio history: {e}")
            return pd.Series(dtype=float)
    
    def get_account_activities(self, activity_types: List[str] = None, limit: int = 100) -> List[Dict]:
        """
        Get account activities (trades, dividends, etc.)
        
        Args:
            activity_types: List of activity types (e.g., ['FILL'])
            limit: Max activities to return
        
        Returns:
            List of activity dicts
        """
        import requests
        
        try:
            url = f"{self.base_url}/v2/account/activities"
            if activity_types:
                url += f"/{','.join(activity_types)}"
            
            params = {"page_size": limit}
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code != 200:
                logging.warning(f"Account activities API error: {response.status_code}")
                return []
            
            activities = response.json()
            return activities if activities else []
            
        except Exception as e:
            logging.error(f"Error getting account activities: {e}")
            return []
    
    # === LONG/SHORT TRADING METHODS ===
    
    def check_shortable(self, symbol: str) -> Dict[str, Any]:
        """
        Check if a symbol is shortable and get borrow info.
        
        Returns:
            Dict with 'shortable', 'easy_to_borrow', 'borrow_rate' keys
        """
        import requests
        
        try:
            url = f"{self.base_url}/v2/assets/{symbol}"
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                logging.warning(f"Asset info API error for {symbol}: {response.status_code}")
                return {'shortable': False, 'easy_to_borrow': False, 'borrow_rate': 0.0}
            
            asset = response.json()
            
            return {
                'shortable': asset.get('shortable', False),
                'easy_to_borrow': asset.get('easy_to_borrow', False),
                'borrow_rate': 0.02,  # Alpaca doesn't provide borrow rates, use default
                'tradable': asset.get('tradable', False),
                'marginable': asset.get('marginable', False),
            }
            
        except Exception as e:
            logging.error(f"Error checking shortable for {symbol}: {e}")
            return {'shortable': False, 'easy_to_borrow': False, 'borrow_rate': 0.0}
    
    def short_sell(
        self,
        symbol: str,
        qty: int,
        dry_run: bool = True
    ) -> Optional[Dict]:
        """
        Place a short sell order.
        
        Args:
            symbol: Stock symbol
            qty: Quantity to short (must be > 0)
            dry_run: If True, log but don't place order
        
        Returns:
            Order info dict if placed, None if dry_run or failed
        """
        if qty <= 0:
            logging.warning(f"Invalid short quantity {qty} for {symbol}")
            return None
        
        # Check if shortable
        short_info = self.check_shortable(symbol)
        if not short_info.get('shortable'):
            logging.warning(f"{symbol} is not shortable")
            return None
        
        if dry_run:
            logging.info(f"[DRY RUN] Would SHORT SELL: {symbol} x {qty}")
            return None
        
        # Place the short sell order (same as regular sell in Alpaca)
        return self.place_market_order(symbol, qty, OrderSide.SELL, dry_run=False)
    
    def cover_short(
        self,
        symbol: str,
        qty: int,
        dry_run: bool = True
    ) -> Optional[Dict]:
        """
        Cover a short position (buy to close).
        
        Args:
            symbol: Stock symbol
            qty: Quantity to cover (must be > 0)
            dry_run: If True, log but don't place order
        
        Returns:
            Order info dict if placed, None if dry_run or failed
        """
        if qty <= 0:
            logging.warning(f"Invalid cover quantity {qty} for {symbol}")
            return None
        
        if dry_run:
            logging.info(f"[DRY RUN] Would COVER SHORT: {symbol} x {qty}")
            return None
        
        # Cover is just a buy order
        return self.place_market_order(symbol, qty, OrderSide.BUY, dry_run=False)
    
    def get_position_side(self, symbol: str) -> str:
        """
        Get the side of a position (long/short/flat).
        
        Returns:
            'long', 'short', or 'flat'
        """
        positions = self.get_positions()
        if symbol not in positions:
            return 'flat'
        
        qty = positions[symbol].get('qty', 0)
        if qty > 0:
            return 'long'
        elif qty < 0:
            return 'short'
        return 'flat'
    
    def get_short_positions(self) -> Dict[str, Dict]:
        """Get all short positions."""
        positions = self.get_positions()
        return {k: v for k, v in positions.items() if v.get('qty', 0) < 0}
    
    def get_long_positions(self) -> Dict[str, Dict]:
        """Get all long positions."""
        positions = self.get_positions()
        return {k: v for k, v in positions.items() if v.get('qty', 0) > 0}
    
    def get_exposure_summary(self) -> Dict[str, float]:
        """
        Get current exposure summary.
        
        Returns:
            Dict with 'gross', 'net', 'long', 'short' exposures
        """
        positions = self.get_positions()
        
        long_exposure = 0.0
        short_exposure = 0.0
        
        for symbol, pos in positions.items():
            market_value = float(pos.get('market_value', 0))
            if market_value > 0:
                long_exposure += market_value
            else:
                short_exposure += abs(market_value)
        
        return {
            'long': long_exposure,
            'short': short_exposure,
            'gross': long_exposure + short_exposure,
            'net': long_exposure - short_exposure,
        }
    
    def get_current_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get real-time bid-ask quotes from Alpaca.
        
        This provides ACTUAL spreads instead of estimated ones,
        improving transaction cost accuracy significantly.
        
        Args:
            symbols: List of symbols to get quotes for
            
        Returns:
            Dict of symbol -> {'bid': float, 'ask': float, 'spread_pct': float, 'mid': float}
        """
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            
            # Batch request for all symbols
            request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            quotes = self.data_client.get_stock_latest_quote(request)
            
            result = {}
            for symbol in symbols:
                if symbol in quotes:
                    q = quotes[symbol]
                    bid = float(q.bid_price) if q.bid_price else 0
                    ask = float(q.ask_price) if q.ask_price else 0
                    
                    # Calculate mid and spread
                    if bid > 0 and ask > 0:
                        mid = (bid + ask) / 2
                        spread_pct = ((ask - bid) / mid) * 100 if mid > 0 else 0.05
                    else:
                        mid = ask if ask > 0 else bid
                        spread_pct = 0.05  # Default 5 bps if no bid-ask
                    
                    result[symbol] = {
                        'bid': bid,
                        'ask': ask,
                        'mid': mid,
                        'spread_pct': spread_pct,
                        'bid_size': int(q.bid_size) if q.bid_size else 0,
                        'ask_size': int(q.ask_size) if q.ask_size else 0,
                    }
            
            logging.info(f"Fetched real-time quotes for {len(result)} symbols")
            return result
            
        except Exception as e:
            logging.warning(f"Could not fetch quotes: {e}")
            # Return empty - caller should use estimated spreads
            return {}
    
    def get_spread_for_symbol(self, symbol: str) -> float:
        """
        Get the current bid-ask spread percentage for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Spread as percentage (e.g., 0.05 for 5 basis points)
        """
        quotes = self.get_current_quotes([symbol])
        if symbol in quotes:
            return quotes[symbol]['spread_pct']
        return 0.05  # Default 5 bps
    
    def get_order_status(self, order_id: str) -> Dict:
        """
        Get the status of an order.
        
        Args:
            order_id: The order ID to check
            
        Returns:
            Dict with order status details
        """
        try:
            order = self.client.get_order_by_id(order_id)
            
            return {
                'order_id': str(order.id),
                'symbol': order.symbol,
                'side': str(order.side),
                'status': str(order.status),
                'qty': int(order.qty) if order.qty else 0,
                'filled_qty': int(order.filled_qty) if order.filled_qty else 0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else 0,
                'created_at': order.created_at.isoformat() if order.created_at else None,
                'filled_at': order.filled_at.isoformat() if order.filled_at else None,
                'type': str(order.type),
                'time_in_force': str(order.time_in_force),
            }
        except Exception as e:
            logging.warning(f"Could not get order status for {order_id}: {e}")
            return {'order_id': order_id, 'status': 'unknown', 'error': str(e)}
    
    def wait_for_fill(
        self, 
        order_id: str, 
        max_wait_seconds: int = 30,
        check_interval: float = 1.0,
    ) -> Dict:
        """
        Wait for an order to fill, with status monitoring.
        
        Args:
            order_id: The order ID to monitor
            max_wait_seconds: Maximum time to wait for fill
            check_interval: How often to check status (seconds)
            
        Returns:
            Dict with final order status and fill details
        """
        import time
        
        start_time = time.time()
        last_status = None
        
        while time.time() - start_time < max_wait_seconds:
            status = self.get_order_status(order_id)
            
            order_status = status.get('status', '').lower()
            
            if order_status == 'filled':
                return {
                    'success': True,
                    'status': 'filled',
                    'order_id': order_id,
                    'filled_qty': status.get('filled_qty', 0),
                    'filled_avg_price': status.get('filled_avg_price', 0),
                    'filled_at': status.get('filled_at'),
                    'wait_time': time.time() - start_time,
                }
            
            elif order_status == 'partially_filled':
                # Log progress
                filled = status.get('filled_qty', 0)
                total = status.get('qty', 0)
                logging.info(f"Order {order_id}: Partial fill {filled}/{total}")
                last_status = status
            
            elif order_status in ['cancelled', 'canceled', 'rejected', 'expired']:
                return {
                    'success': False,
                    'status': order_status,
                    'order_id': order_id,
                    'filled_qty': status.get('filled_qty', 0),
                    'reason': order_status,
                    'wait_time': time.time() - start_time,
                }
            
            time.sleep(check_interval)
        
        # Timeout - return partial fill status if any
        final_status = self.get_order_status(order_id)
        filled_qty = final_status.get('filled_qty', 0)
        
        return {
            'success': filled_qty > 0,
            'status': 'timeout',
            'order_id': order_id,
            'filled_qty': filled_qty,
            'filled_avg_price': final_status.get('filled_avg_price', 0),
            'partial': filled_qty > 0 and filled_qty < final_status.get('qty', 0),
            'wait_time': time.time() - start_time,
        }
    
    def submit_order_with_monitoring(
        self,
        symbol: str,
        side: str,
        quantity: int,
        order_type: str = 'market',
        limit_price: Optional[float] = None,
        max_wait: int = 30,
    ) -> Dict:
        """
        Submit an order and monitor it until filled or timeout.
        
        This provides better fill tracking than fire-and-forget orders.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            order_type: 'market' or 'limit'
            limit_price: Price for limit orders
            max_wait: Maximum seconds to wait for fill
            
        Returns:
            Dict with order result and fill details
        """
        # Submit the order
        if order_type == 'limit' and limit_price:
            from alpaca.trading.requests import LimitOrderRequest
            request = LimitOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
            )
            order = self.client.submit_order(request)
        else:
            request = MarketOrderRequest(
                symbol=symbol,
                qty=quantity,
                side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            )
            order = self.client.submit_order(request)
        
        order_id = str(order.id)
        logging.info(f"Submitted {order_type} order {order_id}: {side} {quantity} {symbol}")
        
        # Monitor for fill
        result = self.wait_for_fill(order_id, max_wait_seconds=max_wait)
        
        # If timeout and not filled, cancel and retry with market order
        if result['status'] == 'timeout' and result['filled_qty'] == 0 and order_type == 'limit':
            logging.warning(f"Limit order {order_id} timed out, converting to market order")
            
            # Cancel the limit order
            try:
                self.client.cancel_order_by_id(order_id)
            except:
                pass
            
            # Submit as market order
            return self.submit_order_with_monitoring(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type='market',
                max_wait=10,
            )
        
        return result
