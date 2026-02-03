#!/bin/bash
#
# TRADING BOT STARTUP SCRIPT
# 
# This script starts the trading bot with the watchdog for automatic recovery.
# The watchdog will restart the bot if it crashes.
#
# Usage:
#   ./start_trading_bot.sh          # Run in foreground (for testing)
#   ./start_trading_bot.sh daemon   # Run in background (for production)
#

cd "$(dirname "$0")"

echo "=============================================="
echo "🤖 MINI FUND TRADING BOT"
echo "=============================================="
echo "Starting with watchdog for automatic recovery"
echo ""

# Check if already running
if pgrep -f "python.*watchdog.py" > /dev/null 2>&1; then
    echo "⚠️  Watchdog is already running!"
    echo "    PID: $(pgrep -f 'python.*watchdog.py')"
    echo ""
    echo "To stop: pkill -f 'python.*watchdog.py'"
    exit 1
fi

if pgrep -f "python.*app.py" > /dev/null 2>&1; then
    echo "⚠️  Bot is already running (without watchdog)!"
    echo "    PID: $(pgrep -f 'python.*app.py')"
    echo ""
    echo "To stop: pkill -f 'python.*app.py'"
    exit 1
fi

# Daemon mode
if [ "$1" == "daemon" ]; then
    echo "🚀 Starting in DAEMON mode (background)..."
    nohup python3 watchdog.py > watchdog.log 2>&1 &
    WATCHDOG_PID=$!
    echo ""
    echo "✅ Watchdog started with PID: $WATCHDOG_PID"
    echo ""
    echo "📋 Useful commands:"
    echo "   View logs:    tail -f watchdog.log"
    echo "   Bot logs:     tail -f outputs/bot.log"
    echo "   Stop bot:     pkill -f 'python.*watchdog.py'"
    echo "   Web UI:       http://localhost:5001"
    echo ""
    
    # Wait a moment and show status
    sleep 5
    if pgrep -f "python.*watchdog.py" > /dev/null 2>&1; then
        echo "✅ Bot is running!"
        echo ""
        echo "The bot will:"
        echo "  🔄 Auto-restart if it crashes"
        echo "  🔴 Execute LIVE trades on schedule"
        echo "  📅 Start trading each morning automatically"
        echo "  💾 Persist settings across restarts"
    else
        echo "❌ Failed to start. Check watchdog.log"
    fi
else
    echo "🚀 Starting in FOREGROUND mode..."
    echo "   (Press Ctrl+C to stop)"
    echo ""
    python3 watchdog.py
fi
