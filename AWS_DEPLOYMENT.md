# AWS Deployment Guide for Mini Quant Fund

This guide covers deploying your bot to AWS and keeping it updated.

## Deployment Options

### Option 1: AWS EC2 (Recommended for Beginners)
A virtual server you control completely.

### Option 2: AWS Elastic Beanstalk
Managed platform - easier but less control.

### Option 3: AWS ECS with Fargate
Container-based - scalable and modern.

---

## Option 1: AWS EC2 Deployment

### Step 1: Launch EC2 Instance

1. Go to AWS Console → EC2 → Launch Instance
2. Choose:
   - **AMI**: Amazon Linux 2023 or Ubuntu 22.04
   - **Instance Type**: t3.small (2 vCPU, 2GB RAM) - ~$15/month
   - **Storage**: 20GB SSD
   - **Security Group**: Allow ports 22 (SSH), 5000 (App), 443 (HTTPS)

3. Create/download your `.pem` key file

### Step 2: Connect to Your Instance

```bash
# Make key file secure
chmod 400 your-key.pem

# Connect via SSH
ssh -i your-key.pem ec2-user@YOUR_EC2_PUBLIC_IP
```

### Step 3: Initial Server Setup

Run these commands on your EC2 instance:

```bash
# Update system
sudo yum update -y  # Amazon Linux
# or
sudo apt update && sudo apt upgrade -y  # Ubuntu

# Install Python 3.11
sudo yum install python3.11 python3.11-pip git -y  # Amazon Linux
# or
sudo apt install python3.11 python3.11-venv git -y  # Ubuntu

# Install nginx (for production)
sudo yum install nginx -y  # Amazon Linux
# or
sudo apt install nginx -y  # Ubuntu

# Create app directory
mkdir -p ~/mini-fund-tool
cd ~/mini-fund-tool
```

### Step 4: Upload Your Code (First Time)

**Option A: Using Git (Recommended)**

```bash
# On EC2 instance
cd ~
git clone https://github.com/YOUR_USERNAME/mini-fund-tool.git
cd mini-fund-tool
```

**Option B: Using SCP (Direct Upload)**

```bash
# On your LOCAL machine
cd "/Users/saef.aldandashi/Desktop/mini fund tool"

# Upload entire folder
scp -i your-key.pem -r . ec2-user@YOUR_EC2_PUBLIC_IP:~/mini-fund-tool/
```

### Step 5: Configure Environment

```bash
# On EC2 instance
cd ~/mini-fund-tool

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create environment file
nano .env
```

Add your API keys to `.env`:
```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
NEWS_API_KEY=your_news_api_key
FLASK_ENV=production
```

### Step 6: Create Systemd Service (Auto-Start)

```bash
sudo nano /etc/systemd/system/quantfund.service
```

Paste this:
```ini
[Unit]
Description=Mini Quant Fund Trading Bot
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/mini-fund-tool
Environment=PATH=/home/ec2-user/mini-fund-tool/.venv/bin
EnvironmentFile=/home/ec2-user/mini-fund-tool/.env
ExecStart=/home/ec2-user/mini-fund-tool/.venv/bin/gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 300 app:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable quantfund
sudo systemctl start quantfund

# Check status
sudo systemctl status quantfund
```

---

## Keeping Code Updated

### Method 1: Git-Based Updates (Recommended)

**Setup (One Time):**
1. Push your code to GitHub
2. On EC2, clone from GitHub

**To Update:**

```bash
# SSH into EC2
ssh -i your-key.pem ec2-user@YOUR_EC2_PUBLIC_IP

# Pull latest changes
cd ~/mini-fund-tool
git pull origin main

# Restart service
sudo systemctl restart quantfund
```

### Method 2: Automatic GitHub Actions (Best)

Create `.github/workflows/deploy.yml` in your repo:

```yaml
name: Deploy to AWS EC2

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - name: Deploy to EC2
      uses: appleboy/ssh-action@master
      with:
        host: ${{ secrets.EC2_HOST }}
        username: ec2-user
        key: ${{ secrets.EC2_SSH_KEY }}
        script: |
          cd ~/mini-fund-tool
          git pull origin main
          source .venv/bin/activate
          pip install -r requirements.txt
          sudo systemctl restart quantfund
```

**Setup GitHub Secrets:**
1. Go to your GitHub repo → Settings → Secrets
2. Add:
   - `EC2_HOST`: Your EC2 public IP
   - `EC2_SSH_KEY`: Contents of your .pem file

Now every push to `main` branch auto-deploys!

### Method 3: Simple Deploy Script

Create `deploy.sh` on your LOCAL machine:

```bash
#!/bin/bash
# Deploy to AWS EC2

EC2_HOST="YOUR_EC2_PUBLIC_IP"
KEY_PATH="path/to/your-key.pem"
APP_DIR="/home/ec2-user/mini-fund-tool"

echo "🚀 Deploying to AWS EC2..."

# Sync files (excludes .git, .venv, cache)
rsync -avz --progress \
  --exclude '.git' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude 'outputs/' \
  --exclude '*.pyc' \
  -e "ssh -i $KEY_PATH" \
  . ec2-user@$EC2_HOST:$APP_DIR/

# Restart service
ssh -i $KEY_PATH ec2-user@$EC2_HOST "sudo systemctl restart quantfund"

echo "✅ Deployment complete!"
```

Run with: `./deploy.sh`

---

## Monitoring Your Bot

### View Logs
```bash
# Service logs
sudo journalctl -u quantfund -f

# Application logs
tail -f ~/mini-fund-tool/outputs/app.log
```

### Check Status
```bash
# Service status
sudo systemctl status quantfund

# Health check
curl http://localhost:5000/api/health
```

### Auto-Restart on Crash
Already configured in systemd service with `Restart=always`

---

## Security Best Practices

1. **Never commit API keys** - Use `.env` file
2. **Use HTTPS** - Set up nginx reverse proxy with SSL
3. **Restrict Security Group** - Only allow your IP for SSH
4. **Enable AWS CloudWatch** - For monitoring and alerts

---

## Cost Estimate

| Resource | Monthly Cost |
|----------|-------------|
| EC2 t3.small | ~$15 |
| EBS 20GB | ~$2 |
| Data Transfer | ~$1 |
| **Total** | **~$18/month** |

---

## Quick Reference

### Start/Stop/Restart
```bash
sudo systemctl start quantfund
sudo systemctl stop quantfund
sudo systemctl restart quantfund
```

### Update Code
```bash
cd ~/mini-fund-tool && git pull && sudo systemctl restart quantfund
```

### View Logs
```bash
sudo journalctl -u quantfund -f
```
