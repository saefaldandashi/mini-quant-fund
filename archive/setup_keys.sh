#!/bin/bash
# Quick setup script for Alpaca API keys

echo "Setting up Alpaca API keys..."
echo ""
echo "Your API Key: PKMANE2UCOEV3WIQX7WOQHG522"
echo ""
echo "You also need your SECRET KEY from Alpaca."
echo "Get it from: https://app.alpaca.markets/paper/dashboard/overview"
echo ""
read -sp "Enter your Alpaca SECRET KEY: " SECRET_KEY
echo ""

export ALPACA_API_KEY="PKMANE2UCOEV3WIQX7WOQHG522"
export ALPACA_SECRET_KEY="$SECRET_KEY"

echo ""
echo "Keys set! You can now run: python bot.py"
echo ""
echo "To make these permanent for this terminal session, run:"
echo "  export ALPACA_API_KEY='PKMANE2UCOEV3WIQX7WOQHG522'"
echo "  export ALPACA_SECRET_KEY='your_secret_key'"
