#!/usr/bin/env python3
"""
TRADING BOT WATCHDOG

This script monitors the trading bot and automatically restarts it if it crashes.
Run this INSTEAD of running app.py directly for maximum resilience.

Features:
- Monitors the trading bot process
- Restarts automatically if it crashes
- Health check via HTTP endpoint
- Logs all restarts
- Sends alerts (can be extended to email/SMS)

Usage:
    python3 watchdog.py

To run in background:
    nohup python3 watchdog.py > watchdog.log 2>&1 &
"""

import subprocess
import time
import sys
import os
import signal
import logging
from datetime import datetime, timedelta
import requests

# Configuration
BOT_SCRIPT = "app.py"
HEALTH_CHECK_URL = "http://localhost:5001/api/health"
HEALTH_CHECK_INTERVAL = 30  # seconds
MAX_RESTART_ATTEMPTS = 10
RESTART_COOLDOWN = 60  # seconds between restart attempts
LOG_FILE = "watchdog.log"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - WATCHDOG - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TradingBotWatchdog:
    """
    Monitors and auto-restarts the trading bot.
    """
    
    def __init__(self):
        self.process = None
        self.restart_count = 0
        self.last_restart = None
        self.start_time = datetime.now()
        self.running = True
        
        # Track failures
        self.consecutive_failures = 0
        self.total_restarts = 0
        
    def start_bot(self):
        """Start the trading bot process."""
        try:
            logger.info("🚀 Starting trading bot...")
            
            # Get the directory of this script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            bot_path = os.path.join(script_dir, BOT_SCRIPT)
            
            # Start the bot
            self.process = subprocess.Popen(
                [sys.executable, bot_path],
                cwd=script_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
            )
            
            self.last_restart = datetime.now()
            self.total_restarts += 1
            
            logger.info(f"✅ Bot started with PID: {self.process.pid}")
            
            # Wait a moment for startup
            time.sleep(5)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start bot: {e}")
            return False
    
    def stop_bot(self):
        """Stop the trading bot process gracefully."""
        if self.process:
            logger.info("🛑 Stopping trading bot...")
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Bot didn't stop gracefully, forcing kill...")
                self.process.kill()
            self.process = None
    
    def is_bot_running(self) -> bool:
        """Check if the bot process is still running."""
        if self.process is None:
            return False
        
        # Check if process is still alive
        poll = self.process.poll()
        if poll is not None:
            logger.warning(f"⚠️ Bot process exited with code: {poll}")
            return False
        
        return True
    
    def health_check(self) -> bool:
        """Check if the bot is responding to health checks."""
        try:
            response = requests.get(HEALTH_CHECK_URL, timeout=10)
            if response.status_code == 200:
                data = response.json()
                
                # Check scheduler health
                scheduler_healthy = data.get('scheduler_healthy', False)
                broker_connected = data.get('broker_connected', False)
                
                if not scheduler_healthy:
                    logger.warning("⚠️ Scheduler not healthy")
                if not broker_connected:
                    logger.warning("⚠️ Broker not connected")
                
                # Consider healthy if at least responding
                return True
            else:
                logger.warning(f"⚠️ Health check returned status: {response.status_code}")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.warning("⚠️ Cannot connect to bot (connection refused)")
            return False
        except requests.exceptions.Timeout:
            logger.warning("⚠️ Health check timeout")
            return False
        except Exception as e:
            logger.warning(f"⚠️ Health check error: {e}")
            return False
    
    def restart_bot(self) -> bool:
        """Restart the bot."""
        # Check cooldown
        if self.last_restart:
            elapsed = (datetime.now() - self.last_restart).total_seconds()
            if elapsed < RESTART_COOLDOWN:
                wait_time = RESTART_COOLDOWN - elapsed
                logger.info(f"⏳ Waiting {wait_time:.0f}s before restart (cooldown)")
                time.sleep(wait_time)
        
        # Check max attempts
        if self.consecutive_failures >= MAX_RESTART_ATTEMPTS:
            logger.error(f"❌ Max restart attempts ({MAX_RESTART_ATTEMPTS}) reached!")
            logger.error("💀 Watchdog giving up. Manual intervention required.")
            return False
        
        logger.info(f"🔄 Restarting bot (attempt {self.consecutive_failures + 1}/{MAX_RESTART_ATTEMPTS})")
        
        # Stop existing process
        self.stop_bot()
        time.sleep(2)
        
        # Start new process
        if self.start_bot():
            # Wait for startup and check health
            time.sleep(10)
            if self.health_check():
                logger.info("✅ Bot restarted successfully and responding")
                self.consecutive_failures = 0
                return True
            else:
                logger.warning("⚠️ Bot started but not responding to health check")
                self.consecutive_failures += 1
                return False
        else:
            self.consecutive_failures += 1
            return False
    
    def run(self):
        """Main watchdog loop."""
        logger.info("=" * 60)
        logger.info("🐕 TRADING BOT WATCHDOG STARTED")
        logger.info("=" * 60)
        logger.info(f"Monitoring: {BOT_SCRIPT}")
        logger.info(f"Health check: {HEALTH_CHECK_URL}")
        logger.info(f"Check interval: {HEALTH_CHECK_INTERVAL}s")
        logger.info(f"Max restart attempts: {MAX_RESTART_ATTEMPTS}")
        logger.info("=" * 60)
        
        # Initial start
        if not self.start_bot():
            logger.error("❌ Failed initial bot start")
            return
        
        # Main monitoring loop
        while self.running:
            try:
                time.sleep(HEALTH_CHECK_INTERVAL)
                
                # Check if process is running
                if not self.is_bot_running():
                    logger.error("🚨 BOT PROCESS DIED!")
                    if not self.restart_bot():
                        break
                    continue
                
                # Health check
                if not self.health_check():
                    self.consecutive_failures += 1
                    logger.warning(f"⚠️ Health check failed ({self.consecutive_failures} consecutive)")
                    
                    # After 3 consecutive failures, restart
                    if self.consecutive_failures >= 3:
                        logger.error("🚨 Too many health check failures - restarting!")
                        if not self.restart_bot():
                            break
                else:
                    self.consecutive_failures = 0
                    
                    # Log status periodically
                    uptime = datetime.now() - self.start_time
                    hours = uptime.total_seconds() / 3600
                    if int(time.time()) % 300 < HEALTH_CHECK_INTERVAL:  # Every ~5 min
                        logger.info(f"✅ Bot healthy | Uptime: {hours:.1f}h | Restarts: {self.total_restarts}")
                
            except KeyboardInterrupt:
                logger.info("🛑 Watchdog interrupted by user")
                break
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                time.sleep(10)
        
        # Cleanup
        self.stop_bot()
        logger.info("🐕 Watchdog stopped")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}")
        self.running = False


def main():
    """Main entry point."""
    watchdog = TradingBotWatchdog()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, watchdog.signal_handler)
    signal.signal(signal.SIGTERM, watchdog.signal_handler)
    
    # Run watchdog
    watchdog.run()


if __name__ == "__main__":
    main()
