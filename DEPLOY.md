# Cloud Deployment Guide - Autonomous Trading Bot

This guide will help you deploy the Mini Quant Fund to the cloud so it runs **24/7** without your computer being on.

## Why Cloud Deployment?

| Local (Current) | Cloud (Target) |
|-----------------|----------------|
| ❌ Stops when computer sleeps | ✅ Runs 24/7 |
| ❌ Manual startup required | ✅ Auto-restarts on crash |
| ❌ No trading when you're away | ✅ Trades autonomously |
| ❌ Power outages stop trading | ✅ 99.9% uptime |

---

## Option 1: Railway.app (Recommended)

**Cost:** ~$5-20/month depending on usage
**Difficulty:** Easy (10 minutes)

### Step 1: Prepare Your Code

1. Make sure all your API keys are in environment variables (not hardcoded)
2. Create a `.env` file with your keys (this stays LOCAL, never commit it)

### Step 2: Sign Up for Railway

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Connect your GitHub account

### Step 3: Create New Project

1. Click "New Project"
2. Choose "Deploy from GitHub repo"
3. Select your `mini fund tool` repository
4. Railway will auto-detect it's a Python app

### Step 4: Set Environment Variables

In Railway dashboard, go to your project → Variables tab, add:

```
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets
GOOGLE_API_KEY=your_gemini_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
```

### Step 5: Deploy

1. Railway will automatically deploy
2. You'll get a URL like `https://mini-quant-fund-production.up.railway.app`
3. Access your bot at that URL!

### Step 6: Enable Auto-Rebalance

Open your deployed app and call:
```bash
curl -X POST https://YOUR-APP.up.railway.app/api/auto-rebalance \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "interval_minutes": 30, "dry_run": false}'
```

---

## Option 2: Render.com (Free Tier Available)

**Cost:** Free tier available, ~$7/month for always-on
**Difficulty:** Easy

### Step 1: Sign Up

1. Go to [render.com](https://render.com)
2. Sign up with GitHub

### Step 2: Create Web Service

1. Click "New +" → "Web Service"
2. Connect your GitHub repo
3. Settings:
   - **Name:** mini-quant-fund
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4`

### Step 3: Add Environment Variables

Same as Railway (see above)

### Step 4: Deploy

Click "Create Web Service" and wait for deployment.

---

## Option 3: Heroku

**Cost:** ~$7/month (no free tier anymore)
**Difficulty:** Easy

### Deploy with CLI:

```bash
# Install Heroku CLI
brew install heroku/brew/heroku

# Login
heroku login

# Create app
heroku create mini-quant-fund

# Set environment variables
heroku config:set ALPACA_API_KEY=xxx
heroku config:set ALPACA_API_SECRET=xxx
heroku config:set GOOGLE_API_KEY=xxx

# Deploy
git push heroku main
```

---

## Verifying Deployment

### Health Check

```bash
curl https://YOUR-APP-URL/api/health
```

Should return:
```json
{
  "status": "healthy",
  "broker_connected": true,
  "auto_rebalance_enabled": true,
  "next_scheduled_run": "2026-01-21T23:30:00Z"
}
```

### Check Status

```bash
curl https://YOUR-APP-URL/api/status
```

### Trigger Manual Rebalance

```bash
curl -X POST https://YOUR-APP-URL/api/run \
  -H "Content-Type: application/json" \
  -d '{"dry_run": false}'
```

---

## Setting Up Autonomous Trading

Once deployed, enable auto-rebalance:

```bash
curl -X POST https://YOUR-APP-URL/api/auto-rebalance \
  -H "Content-Type: application/json" \
  -d '{
    "enabled": true,
    "interval_minutes": 30,
    "dry_run": false,
    "allow_after_hours": false,
    "exposure_pct": 0.8
  }'
```

**Settings:**
- `interval_minutes`: How often to rebalance (30 = every 30 min)
- `dry_run`: Set to `false` for real trades
- `allow_after_hours`: Trade outside market hours
- `exposure_pct`: Capital to deploy (0.8 = 80%)

---

## Monitoring

### Option 1: Railway/Render Dashboard
- View logs in real-time
- See CPU/memory usage
- Get alerts on crashes

### Option 2: External Monitoring (Better Uptime)
- Use [UptimeRobot](https://uptimerobot.com) (free)
- Monitor `/api/health` endpoint
- Get alerts if bot goes down

### Option 3: Slack/Discord Notifications
Add a webhook to get trade notifications (can be added later)

---

## Troubleshooting

### Bot Crashed?
- Check logs in Railway/Render dashboard
- The platform will auto-restart it

### Trades Not Executing?
1. Check health endpoint: is `broker_connected: true`?
2. Check if `auto_rebalance_enabled: true`
3. Is `dry_run` set to `false`?
4. Is market open? (check `allow_after_hours` setting)

### Out of Memory?
- Upgrade to a larger instance
- Railway: Increase RAM in settings
- Render: Upgrade plan

---

## Security Notes

1. **NEVER commit API keys to Git**
2. Use environment variables for all secrets
3. Enable 2FA on your cloud provider account
4. Enable 2FA on your Alpaca account
5. Start with paper trading (`ALPACA_BASE_URL=https://paper-api.alpaca.markets`)

---

## Quick Start Commands

```bash
# 1. Push to GitHub (if not already)
git add .
git commit -m "Add cloud deployment config"
git push origin main

# 2. Deploy to Railway
# (Do this in Railway web UI - connect GitHub repo)

# 3. Set environment variables in Railway dashboard

# 4. Enable auto-rebalance
curl -X POST https://YOUR-APP.up.railway.app/api/auto-rebalance \
  -H "Content-Type: application/json" \
  -d '{"enabled": true, "interval_minutes": 30, "dry_run": false}'

# 5. Verify
curl https://YOUR-APP.up.railway.app/api/health
```

Your bot is now running 24/7 autonomously! 🚀
