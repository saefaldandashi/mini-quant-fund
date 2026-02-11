# Quick Deployment Guide

## One-Command Deployment

On your cloud server, run:

```bash
cd /path/to/mini-fund-tool && git pull origin main && bash deploy_to_cloud.sh
```

Or if the script is already there:

```bash
bash deploy_to_cloud.sh
```

## What the Script Does

1. ✅ Navigates to project directory
2. ✅ Checks Git status
3. ✅ Pulls latest code from GitHub
4. ✅ Verifies all critical files exist
5. ✅ Runs syntax checks
6. ✅ Tests all critical imports
7. ✅ Verifies `can_trade()` method exists
8. ✅ Runs deployment verification
9. ✅ Provides restart instructions

## After Running the Script

The script will tell you how to restart your service. Common options:

### If using systemd:
```bash
sudo systemctl restart your-service-name
```

### If using PM2:
```bash
pm2 restart your-app-name
```

### If using Docker:
```bash
docker-compose restart
```

### If running manually:
Stop your current process (Ctrl+C) and restart it.

## Verify It Worked

After restarting, monitor the first rebalance:

```bash
tail -f server.log | grep -E "REBALANCE|ERROR|WARNING"
```

Look for:
- ✅ No `UnboundLocalError` crashes
- ✅ No `can_trade` attribute errors
- ✅ Rebalance completes successfully
- ✅ Larger position sizes
- ✅ More positions created

## Troubleshooting

If you get errors:

1. **Check you're in the right directory:**
   ```bash
   pwd
   ls -la app.py
   ```

2. **Check Git is working:**
   ```bash
   git status
   git log -1
   ```

3. **Manually verify the fix:**
   ```bash
   python3 -c "from src.risk.realtime_monitor import RealtimeRiskMonitor; from unittest.mock import Mock; m = RealtimeRiskMonitor(Mock()); print('can_trade exists:', hasattr(m, 'can_trade'))"
   ```

4. **Check the file directly:**
   ```bash
   grep -n "def can_trade" src/risk/realtime_monitor.py
   ```

---

**The script is ready to use!** Just run it on your cloud server.
