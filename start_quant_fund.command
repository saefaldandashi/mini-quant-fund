#!/bin/bash

# ===============================================
# Mini Quant Fund - Quick Start
# ===============================================
# Double-click this file to start the trading bot!

echo "🚀 Starting Mini Quant Fund..."
echo ""

# Navigate to project directory
cd "/Users/saef.aldandashi/Desktop/mini fund tool"

# Kill any existing server
pkill -f "python app.py" 2>/dev/null
sleep 1

# Activate virtual environment
source .venv/bin/activate

# Start server in background
echo "Starting server..."
python app.py > /tmp/mini_quant_fund.log 2>&1 &
SERVER_PID=$!

# Wait for server to start and verify it's responding
echo "Waiting for server to start..."
MAX_WAIT=15
WAITED=0
SERVER_READY=0

while [ $WAITED -lt $MAX_WAIT ]; do
    sleep 1
    WAITED=$((WAITED + 1))
    
    # Check if process is still running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ Server process died. Check /tmp/mini_quant_fund.log for errors."
        tail -20 /tmp/mini_quant_fund.log
        read -p "Press Enter to close..."
        exit 1
    fi
    
    # Check if server is responding
    if curl -s http://localhost:5000 > /dev/null 2>&1; then
        SERVER_READY=1
        break
    fi
done

# Check if server is ready
if [ $SERVER_READY -eq 1 ]; then
    echo ""
    echo "✅ Server running at http://localhost:5000"
    echo ""
    echo "Opening browser..."
    open "http://localhost:5000"
    echo ""
    echo "╔════════════════════════════════════════════════╗"
    echo "║  🤖 MINI QUANT FUND IS RUNNING                 ║"
    echo "║                                                ║"
    echo "║  Dashboard: http://localhost:5000              ║"
    echo "║                                                ║"
    echo "║  Press Ctrl+C to stop the server               ║"
    echo "║  Logs: /tmp/mini_quant_fund.log                ║"
    echo "╚════════════════════════════════════════════════╝"
    echo ""
    
    # Keep terminal open and show server output
    tail -f /tmp/mini_quant_fund.log
else
    echo "❌ Server failed to start after ${MAX_WAIT} seconds."
    echo "Check /tmp/mini_quant_fund.log for errors:"
    tail -30 /tmp/mini_quant_fund.log
    kill $SERVER_PID 2>/dev/null
    read -p "Press Enter to close..."
    exit 1
fi
