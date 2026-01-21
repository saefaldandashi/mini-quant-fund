#!/bin/bash
# Simple server startup script

cd "/Users/saef.aldandashi/Desktop/mini fund tool"

# Kill any existing server
pkill -f "python app.py" 2>/dev/null
sleep 1

# Activate virtual environment
source .venv/bin/activate

# Start server
echo "Starting server..."
python app.py
