#!/bin/bash
# SUPER SIMPLE DEPLOYMENT - Just pull and restart
# This is the absolute minimum you need to do

echo "🚀 Simple Deployment"
echo "==================="
echo ""

# Step 1: Pull latest code
echo "Step 1: Pulling latest code from GitHub..."
git pull origin main

if [ $? -eq 0 ]; then
    echo "✅ Code updated!"
else
    echo "❌ Error pulling code. Check your connection."
    exit 1
fi

echo ""
echo "Step 2: Restart your application"
echo "================================="
echo ""
echo "Choose how you run your app:"
echo ""
echo "A) systemd service:"
echo "   sudo systemctl restart your-service-name"
echo ""
echo "B) PM2:"
echo "   pm2 restart your-app-name"
echo ""
echo "C) Docker:"
echo "   docker-compose restart"
echo ""
echo "D) Manual:"
echo "   Stop it (Ctrl+C) and start it again"
echo ""
echo "After restarting, check logs:"
echo "   tail -f server.log"
echo ""
