#!/bin/bash
# =============================================================================
# AWS EC2 Deployment Script for Mini Quant Fund
# =============================================================================
#
# Usage: ./deploy.sh [options]
#
# Options:
#   --full      Full deployment (sync all files + restart)
#   --restart   Just restart the service
#   --logs      View live logs
#   --status    Check service status
#
# Setup:
#   1. Edit EC2_HOST and KEY_PATH below
#   2. Make executable: chmod +x deploy.sh
#   3. Run: ./deploy.sh
# =============================================================================

# ============ CONFIGURATION ============
EC2_HOST="13.48.132.230"
KEY_PATH="$HOME/Downloads/MQ Fund.pem"
EC2_USER="ec2-user"
APP_DIR="/home/$EC2_USER/mini-fund-tool"
# ========================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if configured
if [[ "$EC2_HOST" == "YOUR_EC2_PUBLIC_IP" ]]; then
    echo -e "${RED}❌ Error: Please edit deploy.sh and set your EC2_HOST${NC}"
    exit 1
fi

# SSH command helper
ssh_cmd() {
    ssh -i "$KEY_PATH" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" "$@"
}

# Parse arguments
ACTION="${1:-full}"

case $ACTION in
    --full|full)
        echo -e "${YELLOW}🚀 Starting full deployment to AWS EC2...${NC}"
        echo "   Host: $EC2_HOST"
        echo ""
        
        # Sync files (excludes unnecessary directories)
        echo -e "${YELLOW}📦 Syncing files...${NC}"
        rsync -avz --progress \
            --exclude '.git' \
            --exclude '.venv' \
            --exclude '__pycache__' \
            --exclude 'outputs/' \
            --exclude '*.pyc' \
            --exclude '.env' \
            --exclude 'node_modules' \
            --exclude '.DS_Store' \
            -e "ssh -i $KEY_PATH -o StrictHostKeyChecking=no" \
            . "$EC2_USER@$EC2_HOST:$APP_DIR/"
        
        echo ""
        echo -e "${YELLOW}📥 Installing dependencies...${NC}"
        ssh_cmd "cd $APP_DIR && source .venv/bin/activate && pip install -r requirements.txt -q"
        
        echo ""
        echo -e "${YELLOW}🔄 Restarting service...${NC}"
        ssh_cmd "sudo systemctl restart quantfund"
        
        # Wait a moment and check status
        sleep 3
        STATUS=$(ssh_cmd "sudo systemctl is-active quantfund")
        
        if [[ "$STATUS" == "active" ]]; then
            echo -e "${GREEN}✅ Deployment complete! Service is running.${NC}"
            echo ""
            
            # Quick health check
            HEALTH=$(ssh_cmd "curl -s http://localhost:5000/api/health | python3 -c \"import sys,json; print(json.load(sys.stdin).get('status', 'unknown'))\"" 2>/dev/null)
            echo -e "   Health: ${GREEN}$HEALTH${NC}"
            echo "   URL: http://$EC2_HOST:5000"
        else
            echo -e "${RED}❌ Service failed to start. Check logs with: ./deploy.sh --logs${NC}"
            exit 1
        fi
        ;;
        
    --restart|restart)
        echo -e "${YELLOW}🔄 Restarting service...${NC}"
        ssh_cmd "sudo systemctl restart quantfund"
        sleep 2
        ssh_cmd "sudo systemctl status quantfund --no-pager"
        ;;
        
    --logs|logs)
        echo -e "${YELLOW}📜 Viewing live logs (Ctrl+C to exit)...${NC}"
        ssh_cmd "sudo journalctl -u quantfund -f"
        ;;
        
    --status|status)
        echo -e "${YELLOW}📊 Service Status:${NC}"
        ssh_cmd "sudo systemctl status quantfund --no-pager"
        echo ""
        echo -e "${YELLOW}🏥 Health Check:${NC}"
        ssh_cmd "curl -s http://localhost:5000/api/health | python3 -m json.tool"
        ;;
        
    --setup|setup)
        echo -e "${YELLOW}🔧 Initial server setup...${NC}"
        
        # Install dependencies
        ssh_cmd "sudo yum update -y && sudo yum install python3.11 python3.11-pip git nginx -y"
        
        # Create app directory
        ssh_cmd "mkdir -p $APP_DIR"
        
        # Create virtual environment
        ssh_cmd "cd $APP_DIR && python3.11 -m venv .venv"
        
        echo -e "${GREEN}✅ Server setup complete!${NC}"
        echo "   Next steps:"
        echo "   1. Create .env file on server with API keys"
        echo "   2. Run: ./deploy.sh --full"
        ;;
        
    --help|help)
        echo "Usage: ./deploy.sh [option]"
        echo ""
        echo "Options:"
        echo "  --full      Full deployment (sync files + install deps + restart)"
        echo "  --restart   Just restart the service"
        echo "  --logs      View live service logs"
        echo "  --status    Check service and health status"
        echo "  --setup     Initial server setup (first time only)"
        echo "  --help      Show this help message"
        ;;
        
    *)
        echo -e "${RED}Unknown option: $ACTION${NC}"
        echo "Use --help for available options"
        exit 1
        ;;
esac
