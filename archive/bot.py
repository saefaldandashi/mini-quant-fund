"""
Main bot orchestration: monthly rebalancing with market regime filter.
"""
import os
import sys
import logging
from typing import Dict, List

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional

from utils import (
    setup_logging,
    retry_with_backoff,
    is_already_rebalanced_today,
    save_rebalance_date,
    get_current_date,
    get_env_bool,
)
from broker_alpaca import AlpacaBroker
from strategy import (
    determine_market_regime,
    select_top_momentum_stocks,
)
from alpaca.trading.enums import OrderSide
import config


def main(exit_on_error=True, force_rebalance=False):
    """
    Main entry point for the bot.
    
    Args:
        exit_on_error: If True, call sys.exit() on errors. If False, raise exceptions.
        force_rebalance: If True, skip the daily idempotency check.
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    # Setup logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    setup_logging(log_level)
    
    logging.info("=" * 60)
    logging.info("Mini Quant Fund Bot - Starting")
    logging.info("=" * 60)
    
    # Get environment variables
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    dry_run = get_env_bool("DRY_RUN", default=True)
    allow_after_hours = get_env_bool("ALLOW_AFTER_HOURS", default=False)
    force_rebalance = force_rebalance or get_env_bool("FORCE_REBALANCE", default=False)
    
    # Validate API keys
    if not api_key or not secret_key:
        error_msg = "Missing API keys. Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
        logging.error(error_msg)
        if exit_on_error:
            sys.exit(1)
        else:
            raise ValueError(error_msg)
    
    # Check daily idempotency (skip if force_rebalance is True)
    if not force_rebalance and is_already_rebalanced_today():
        logging.info("Exiting: already rebalanced today. Use FORCE_REBALANCE=1 to run again.")
        if exit_on_error:
            sys.exit(0)
        else:
            return 0
    
    try:
        # Initialize broker (paper-only enforced in broker)
        broker = AlpacaBroker(api_key=api_key, secret_key=secret_key, paper=True)
        
        # Check market hours
        market_open = broker.is_market_open()
        if not market_open and not allow_after_hours:
            logging.warning(
                "Market is closed and ALLOW_AFTER_HOURS=0. "
                "Set ALLOW_AFTER_HOURS=1 to allow trading after hours."
            )
            if exit_on_error:
                sys.exit(0)
            else:
                return 0
        
        if not market_open:
            logging.info("Market is closed, but ALLOW_AFTER_HOURS=1, proceeding...")
        
        # Get account info
        account = broker.get_account()
        equity = account["equity"]
        logging.info(f"Account equity: ${equity:,.2f}")
        
        # Fetch historical data for benchmark and universe
        logging.info("Fetching historical data...")
        
        @retry_with_backoff(max_retries=3, exceptions=(Exception,))
        def fetch_data():
            # Fetch SPY data (need MA_DAYS + buffer)
            spy_data = broker.get_historical_bars(
                [config.BENCHMARK],
                days=config.MA_DAYS + 20
            )
            
            if config.BENCHMARK not in spy_data:
                raise ValueError(f"Could not fetch data for {config.BENCHMARK}")
            
            # Fetch universe data (need LOOKBACK_DAYS + buffer)
            universe_data = broker.get_historical_bars(
                config.UNIVERSE,
                days=config.LOOKBACK_DAYS + 20
            )
            
            return spy_data[config.BENCHMARK], universe_data
        
        spy_prices, universe_data = fetch_data()
        
        # Determine market regime
        risk_on, spy_close, spy_ma = determine_market_regime(spy_prices, config.MA_DAYS)
        
        # Get current positions
        current_positions = broker.get_positions()
        logging.info(f"Current positions: {len(current_positions)} stocks")
        for symbol, pos in current_positions.items():
            logging.info(f"  {symbol}: {pos['qty']:.0f} shares, ${pos['market_value']:.2f}")
        
        # Determine target positions
        if risk_on:
            logging.info("Risk-ON: Selecting top momentum stocks...")
            
            # Select top momentum stocks
            top_stocks = select_top_momentum_stocks(
                universe_data,
                lookback_days=config.LOOKBACK_DAYS,
                top_n=config.TOP_N
            )
            
            target_symbols = [symbol for symbol, _ in top_stocks]
            
            # Calculate target shares
            target_shares = broker.calculate_target_shares(
                {s: 0 for s in target_symbols},  # Placeholder dict
                equity,
                cash_buffer_pct=config.CASH_BUFFER_PCT
            )
        else:
            logging.info("Risk-OFF: Exiting all positions to cash...")
            target_shares = {}
        
        logging.info(f"Target positions: {len(target_shares)} stocks")
        for symbol, qty in target_shares.items():
            logging.info(f"  {symbol}: {qty} shares")
        
        # Execute rebalancing
        logging.info("Executing rebalancing...")
        
        # Get current position quantities
        current_shares = {
            symbol: int(pos["qty"])
            for symbol, pos in current_positions.items()
        }
        
        # Determine orders needed
        all_symbols = set(current_shares.keys()) | set(target_shares.keys())
        
        orders_placed = []
        orders_planned = []
        transactions = []  # Track all transactions for display
        
        # Get current prices for transaction tracking
        all_symbols_list = list(all_symbols)
        current_prices = broker.get_current_prices(all_symbols_list)
        
        for symbol in all_symbols:
            current_qty = current_shares.get(symbol, 0)
            target_qty = target_shares.get(symbol, 0)
            
            if current_qty == target_qty:
                continue  # No change needed
            
            price = current_prices.get(symbol, 0.0)
            
            if current_qty > target_qty:
                # Need to sell
                sell_qty = current_qty - target_qty
                order = broker.place_market_order(
                    symbol=symbol,
                    qty=sell_qty,
                    side=OrderSide.SELL,
                    dry_run=dry_run
                )
                if order:
                    orders_placed.append(order)
                    transactions.append({
                        "symbol": symbol,
                        "side": "SELL",
                        "qty": sell_qty,
                        "price": price,
                        "value": sell_qty * price,
                        "status": "executed"
                    })
                else:
                    orders_planned.append((symbol, sell_qty, "SELL"))
                    transactions.append({
                        "symbol": symbol,
                        "side": "SELL",
                        "qty": sell_qty,
                        "price": price,
                        "value": sell_qty * price,
                        "status": "planned"
                    })
            else:
                # Need to buy
                buy_qty = target_qty - current_qty
                order = broker.place_market_order(
                    symbol=symbol,
                    qty=buy_qty,
                    side=OrderSide.BUY,
                    dry_run=dry_run
                )
                if order:
                    orders_placed.append(order)
                    transactions.append({
                        "symbol": symbol,
                        "side": "BUY",
                        "qty": buy_qty,
                        "price": price,
                        "value": buy_qty * price,
                        "status": "executed"
                    })
                else:
                    orders_planned.append((symbol, buy_qty, "BUY"))
                    transactions.append({
                        "symbol": symbol,
                        "side": "BUY",
                        "qty": buy_qty,
                        "price": price,
                        "value": buy_qty * price,
                        "status": "planned"
                    })
        
        # Log transactions
        if transactions:
            logging.info("Transactions:")
            for txn in transactions:
                status_str = "[DRY RUN]" if txn["status"] == "planned" else ""
                logging.info(
                    f"  {status_str} {txn['side']} {txn['qty']} {txn['symbol']} @ ${txn['price']:.2f} = ${txn['value']:,.2f}"
                )
        
        if dry_run:
            logging.info(f"[DRY RUN] Would have placed {len(orders_planned)} orders")
        else:
            logging.info(f"Placed {len(orders_placed)} orders")
        
        # Save rebalance state
        current_date = get_current_date()
        save_rebalance_date(current_date)
        
        logging.info("=" * 60)
        logging.info("Rebalancing complete!")
        logging.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        if exit_on_error:
            sys.exit(1)
        else:
            raise


if __name__ == "__main__":
    main()
