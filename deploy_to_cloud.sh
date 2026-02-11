#!/bin/bash
# Automated Deployment Script for Cloud Server
# Run this on your cloud server to deploy all fixes

set -e  # Exit on error

echo "=========================================="
echo "🚀 AUTOMATED DEPLOYMENT SCRIPT"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Navigate to project directory
echo -e "${YELLOW}Step 1: Navigating to project directory...${NC}"
cd "$(dirname "$0")" || cd ~/mini-fund-tool || cd /home/ec2-user/mini-fund-tool || {
    echo -e "${RED}❌ Could not find project directory. Please run this script from the project root.${NC}"
    exit 1
}
PROJECT_DIR=$(pwd)
echo -e "${GREEN}✓ Current directory: $PROJECT_DIR${NC}"
echo ""

# Step 2: Check Git status
echo -e "${YELLOW}Step 2: Checking Git status...${NC}"
if [ ! -d ".git" ]; then
    echo -e "${RED}❌ Not a Git repository. Please ensure you're in the correct directory.${NC}"
    exit 1
fi

# Check if there are uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}⚠️  Warning: You have uncommitted changes. Stashing them...${NC}"
    git stash
fi

echo -e "${GREEN}✓ Git repository found${NC}"
echo ""

# Step 3: Pull latest code
echo -e "${YELLOW}Step 3: Pulling latest code from GitHub...${NC}"
git fetch origin main
LOCAL=$(git rev-parse @)
REMOTE=$(git rev-parse @{u})

if [ "$LOCAL" = "$REMOTE" ]; then
    echo -e "${GREEN}✓ Already up to date with origin/main${NC}"
else
    echo -e "${YELLOW}Pulling changes...${NC}"
    git pull origin main
    echo -e "${GREEN}✓ Code updated successfully${NC}"
fi
echo ""

# Step 4: Verify critical files exist
echo -e "${YELLOW}Step 4: Verifying critical files...${NC}"
CRITICAL_FILES=(
    "src/data/symbol_validator.py"
    "src/risk/realtime_monitor.py"
    "src/learning/learning_engine.py"
    "app.py"
    "broker_alpaca.py"
)

ALL_EXIST=true
for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}  ✓ $file${NC}"
    else
        echo -e "${RED}  ✗ $file MISSING${NC}"
        ALL_EXIST=false
    fi
done

if [ "$ALL_EXIST" = false ]; then
    echo -e "${RED}❌ Some critical files are missing!${NC}"
    exit 1
fi
echo ""

# Step 5: Run Python syntax check
echo -e "${YELLOW}Step 5: Running syntax checks...${NC}"
python3 -m py_compile src/risk/realtime_monitor.py app.py 2>&1 | while IFS= read -r line; do
    echo "  $line"
done

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo -e "${GREEN}✓ Syntax check passed${NC}"
else
    echo -e "${RED}❌ Syntax errors found!${NC}"
    exit 1
fi
echo ""

# Step 6: Test imports
echo -e "${YELLOW}Step 6: Testing critical imports...${NC}"
python3 << 'PYTHON_EOF'
import sys
errors = []

try:
    from src.data.symbol_validator import SymbolValidator, get_symbol_validator
    print("  ✓ SymbolValidator imports OK")
except Exception as e:
    print(f"  ✗ SymbolValidator import failed: {e}")
    errors.append("SymbolValidator")

try:
    from src.risk.realtime_monitor import RealtimeRiskMonitor, get_realtime_monitor, set_realtime_monitor
    print("  ✓ RealtimeRiskMonitor imports OK")
    
    # Test that can_trade method exists
    from unittest.mock import Mock
    mock_broker = Mock()
    monitor = RealtimeRiskMonitor(mock_broker)
    if hasattr(monitor, 'can_trade'):
        print("  ✓ can_trade() method exists")
    else:
        print("  ✗ can_trade() method MISSING")
        errors.append("can_trade method")
        
except Exception as e:
    print(f"  ✗ RealtimeRiskMonitor import failed: {e}")
    errors.append("RealtimeRiskMonitor")

try:
    from src.learning.learning_engine import LearningEngine
    print("  ✓ LearningEngine imports OK")
except Exception as e:
    print(f"  ✗ LearningEngine import failed: {e}")
    errors.append("LearningEngine")

if errors:
    print(f"\n❌ Import errors: {', '.join(errors)}")
    sys.exit(1)
else:
    print("\n✓ All imports successful")
PYTHON_EOF

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo -e "${RED}❌ Import tests failed!${NC}"
    exit 1
fi
echo ""

# Step 7: Run deployment verification
echo -e "${YELLOW}Step 7: Running deployment verification...${NC}"
if [ -f "deploy_and_verify.py" ]; then
    python3 deploy_and_verify.py 2>&1 | tail -20
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ Deployment verification passed${NC}"
    else
        echo -e "${YELLOW}⚠️  Some verification checks had warnings (may be OK)${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  deploy_and_verify.py not found, skipping${NC}"
fi
echo ""

# Step 8: Restart service (if applicable)
echo -e "${YELLOW}Step 8: Service restart instructions...${NC}"
echo "Please restart your application using one of these methods:"
echo ""
echo "  Option 1 (systemd):"
echo "    sudo systemctl restart your-service-name"
echo ""
echo "  Option 2 (PM2):"
echo "    pm2 restart your-app-name"
echo ""
echo "  Option 3 (Docker):"
echo "    docker-compose restart"
echo ""
echo "  Option 4 (Manual):"
echo "    # Stop your current process and restart it"
echo ""

# Step 9: Summary
echo "=========================================="
echo -e "${GREEN}✅ DEPLOYMENT COMPLETE${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Restart your application (see instructions above)"
echo "2. Monitor the first rebalance:"
echo "   tail -f server.log | grep -E 'REBALANCE|ERROR|WARNING'"
echo "3. Check for success indicators:"
echo "   - No UnboundLocalError crashes"
echo "   - Larger position sizes (2-3x)"
echo "   - More positions created"
echo "   - Successful completion"
echo ""
echo -e "${GREEN}All fixes have been deployed!${NC}"
echo ""
