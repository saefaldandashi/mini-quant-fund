#!/bin/bash
# Check if CI/CD is set up

echo "🔍 Checking for CI/CD Setup"
echo "==========================="
echo ""

FOUND_CI=false

# Check for GitHub Actions
if [ -d ".github/workflows" ]; then
    echo "✅ GitHub Actions found!"
    ls -la .github/workflows/*.yml .github/workflows/*.yaml 2>/dev/null
    FOUND_CI=true
fi

# Check for GitLab CI
if [ -f ".gitlab-ci.yml" ]; then
    echo "✅ GitLab CI found!"
    cat .gitlab-ci.yml | head -20
    FOUND_CI=true
fi

# Check for CircleCI
if [ -d ".circleci" ]; then
    echo "✅ CircleCI found!"
    ls -la .circleci/
    FOUND_CI=true
fi

# Check for Jenkins
if [ -f "Jenkinsfile" ]; then
    echo "✅ Jenkins found!"
    cat Jenkinsfile | head -20
    FOUND_CI=true
fi

# Check for AWS CodeDeploy
if [ -f "appspec.yml" ] || [ -d ".deploy" ]; then
    echo "✅ AWS CodeDeploy found!"
    FOUND_CI=true
fi

# Check for other deployment scripts
if [ -f "deploy.sh" ] || [ -f "deploy.yml" ]; then
    echo "✅ Custom deployment script found!"
    ls -la deploy.* 2>/dev/null
    FOUND_CI=true
fi

echo ""
if [ "$FOUND_CI" = false ]; then
    echo "❌ No CI/CD found"
    echo ""
    echo "You'll need to manually deploy:"
    echo "  1. git pull origin main"
    echo "  2. Restart your app"
    echo ""
    echo "Or use the simple script:"
    echo "  bash SIMPLE_DEPLOY.sh"
else
    echo "✅ CI/CD is configured!"
    echo ""
    echo "Your code might auto-deploy when pushed to GitHub."
    echo "Check your CI/CD dashboard to see if it's running."
fi
