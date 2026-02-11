# Step-by-Step Deployment Guide (Like You're 5)

## 🎯 What We're Doing
We're going to update your cloud server with all the fixes so your trading bot works better!

---

## Step 1: Open Your Cloud Server

1. Open your terminal (the black window where you type commands)
2. Connect to your cloud server. Type this command:
   ```bash
   ssh your-username@your-server-address
   ```
   (Replace `your-username` and `your-server-address` with your actual details)

3. If it asks for a password, type it and press Enter

**✅ You should now see something like: `ec2-user@your-server:~$`**

---

## Step 2: Go to Your Project Folder

Type this command and press Enter:
```bash
cd mini-fund-tool
```

If that doesn't work, try:
```bash
cd ~/mini-fund-tool
```

Or:
```bash
cd /home/ec2-user/mini-fund-tool
```

**✅ You should see the folder name in your prompt now**

---

## Step 3: Check Where You Are

Type this to make sure you're in the right place:
```bash
ls -la app.py
```

**✅ You should see information about `app.py` file (not an error)**

---

## Step 4: Get the Latest Code

Type this command and press Enter:
```bash
git pull origin main
```

**What this does:** Downloads all the new fixes from GitHub

**✅ You should see messages like:**
- `Updating...`
- `Fast-forward`
- `X files changed`

---

## Step 5: Make the Script Ready to Run

Type this command and press Enter:
```bash
chmod +x deploy_to_cloud.sh
```

**What this does:** Makes the script file executable (able to run)

**✅ You should see nothing (that's good - no error means success!)**

---

## Step 6: Run the Deployment Script

Type this command and press Enter:
```bash
bash deploy_to_cloud.sh
```

**What this does:** Runs the automated script that checks everything

**✅ You should see lots of green checkmarks (✓) and messages like:**
- `✓ Current directory: /path/to/mini-fund-tool`
- `✓ Git repository found`
- `✓ Code updated successfully`
- `✓ SymbolValidator imports OK`
- `✓ RealtimeRiskMonitor imports OK`
- `✓ can_trade() method exists`
- `✓ All imports successful`
- `✅ DEPLOYMENT COMPLETE`

**⚠️ If you see red X's or errors, stop and check what the error says**

---

## Step 7: Restart Your Application

The script will tell you how to restart. Choose the method that matches how you run your app:

### Option A: If you use systemd (most common)
Type this:
```bash
sudo systemctl restart your-service-name
```
(Replace `your-service-name` with your actual service name)

### Option B: If you use PM2
Type this:
```bash
pm2 restart your-app-name
```
(Replace `your-app-name` with your actual app name)

### Option C: If you use Docker
Type this:
```bash
docker-compose restart
```

### Option D: If you run it manually
1. Find the process running your app (press `Ctrl+C` to stop it)
2. Start it again the same way you normally start it

**✅ Your app should restart**

---

## Step 8: Watch the Logs

Type this command to watch what's happening:
```bash
tail -f server.log
```

**What this does:** Shows you the latest messages from your app

**✅ You should see messages scrolling by**

**To stop watching:** Press `Ctrl+C`

---

## Step 9: Check for Success

While watching the logs, look for these good signs:

✅ **Good signs:**
- `🧠 SMART REBALANCE INITIATED`
- `✅ All positions passed validation`
- `REBALANCE COMPLETE`
- No `UnboundLocalError`
- No `can_trade` errors
- Larger position sizes mentioned

❌ **Bad signs (if you see these, something is wrong):**
- `UnboundLocalError`
- `AttributeError: 'RealtimeRiskMonitor' object has no attribute 'can_trade'`
- `REBALANCE FAILED`
- Lots of red error messages

---

## Step 10: Celebrate! 🎉

If you see the good signs, you're done! The fixes are working!

---

## 🆘 Troubleshooting (If Something Goes Wrong)

### Problem: "cd mini-fund-tool" doesn't work
**Solution:** Find where your project is:
```bash
find ~ -name "app.py" -type f 2>/dev/null
```
This will show you where `app.py` is. Then `cd` to that folder.

### Problem: "git pull" says "Already up to date"
**Solution:** That's fine! It means you already have the latest code. Continue to Step 5.

### Problem: "Permission denied" when running the script
**Solution:** Make sure you made it executable:
```bash
chmod +x deploy_to_cloud.sh
```

### Problem: Import errors in Step 6
**Solution:** Make sure you're in the right directory:
```bash
pwd
ls -la app.py
```
If `app.py` doesn't exist, you're in the wrong folder!

### Problem: Can't restart the service
**Solution:** 
1. Find out how your app is running:
   ```bash
   ps aux | grep python
   ```
2. Look for a process running `app.py`
3. Note the process ID (the number in the second column)
4. Stop it: `kill [process-id]`
5. Start it again however you normally start it

### Problem: Still seeing errors after restart
**Solution:** Check the exact error message:
```bash
tail -50 server.log | grep -i error
```
Then share that error message to get help fixing it.

---

## 📋 Quick Checklist

Before you start:
- [ ] I have access to my cloud server
- [ ] I know how to SSH into it
- [ ] I know where my project folder is
- [ ] I know how my app is running (systemd/PM2/Docker/manual)

After deployment:
- [ ] Script ran without errors
- [ ] App restarted successfully
- [ ] Logs show no `UnboundLocalError`
- [ ] Logs show no `can_trade` errors
- [ ] Rebalance completes successfully

---

## 🎯 Summary (The Super Short Version)

1. SSH to server
2. `cd mini-fund-tool`
3. `git pull origin main`
4. `bash deploy_to_cloud.sh`
5. Restart your app
6. Watch logs for success

That's it! 🚀
