# How to Find Your SSH Command

## Quick Ways to Find Your Server Details

### Method 1: Check Your Cloud Provider Dashboard

**If using AWS EC2:**
1. Go to AWS Console → EC2 → Instances
2. Find your instance
3. Look for "Public IPv4 address" or "Public DNS"
4. Username is usually: `ec2-user` (for Amazon Linux) or `ubuntu` (for Ubuntu)

**If using DigitalOcean:**
1. Go to Droplets
2. Click on your droplet
3. Look for "IPv4" address
4. Username is usually: `root` or `ubuntu`

**If using Google Cloud:**
1. Go to Compute Engine → VM instances
2. Find your instance
3. Look for "External IP"
4. Username is usually: your Google account username

**If using Azure:**
1. Go to Virtual machines
2. Find your VM
3. Look for "Public IP address"
4. Username is what you set when creating the VM

---

### Method 2: Check Your Previous SSH Commands

If you've connected before, check your terminal history:

```bash
history | grep ssh
```

This will show your previous SSH commands.

---

### Method 3: Check Your SSH Config

Look for an SSH config file:

```bash
cat ~/.ssh/config
```

This might have your server details saved.

---

### Method 4: Common Formats

Your SSH command will look like one of these:

**AWS EC2 (Amazon Linux):**
```bash
ssh ec2-user@your-server-ip-address
```

**AWS EC2 (Ubuntu):**
```bash
ssh ubuntu@your-server-ip-address
```

**DigitalOcean:**
```bash
ssh root@your-server-ip-address
```

**Generic Linux:**
```bash
ssh username@your-server-ip-address
```

---

### Method 5: If You Have a Domain Name

If you set up a domain name pointing to your server:

```bash
ssh username@your-domain.com
```

---

### Method 6: Check Your Cloud Provider's Connection Guide

Most cloud providers have a "Connect" button that shows you the exact SSH command:
- AWS: Click "Connect" button on your instance
- DigitalOcean: Click "Access" → "Launch Droplet Console"
- Google Cloud: Click "SSH" button
- Azure: Click "Connect" → "SSH"

---

## Still Can't Find It?

**Tell me:**
1. What cloud provider are you using? (AWS, DigitalOcean, Google Cloud, Azure, etc.)
2. Do you remember setting up a server?
3. Do you have access to your cloud provider's dashboard?

**Or try this:**
If you know your server's IP address, try these common usernames:

```bash
ssh ec2-user@YOUR-IP-ADDRESS
ssh ubuntu@YOUR-IP-ADDRESS
ssh root@YOUR-IP-ADDRESS
ssh admin@YOUR-IP-ADDRESS
```

One of them should work!

---

## Example

If your server IP is `54.123.45.67` and you're using AWS EC2 with Amazon Linux:

```bash
ssh ec2-user@54.123.45.67
```

If it's DigitalOcean:

```bash
ssh root@54.123.45.67
```

---

## Need Help?

Share with me:
- Your cloud provider (AWS, DigitalOcean, etc.)
- Your server's IP address (if you know it)
- Any error messages you get when trying to connect

And I'll help you figure out the exact command!
