#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Configuration
REPO_DIR="/root/ha_washdata"
PROD_DIR="/root/ha_config/custom_components/ha_washdata"
COMPONENT="ha_washdata"
RUN=0
BACKUP=1
RESTART=1

# Home Assistant API configuration
HA_API_URL="https://home-int.1534.ovh/api/services/homeassistant/restart"
HA_API_TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJhY2JkNTM0MmQ0NzA0M2NhODUyN2MyNDMyN2I4NzFmMiIsImlhdCI6MTc2MTEyNDI0MSwiZXhwIjoyMDc2NDg0MjQxfQ.Y9duiBdbx8al0fyG9gPdEVuYECzPcUpLmXbw_hM9vBk"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--run] [--no-backup] [--no-restart]

Sync FROM development repository TO production Home Assistant.

This script:
  - Syncs code from $REPO_DIR to $PROD_DIR
  - Creates timestamped backup in $REPO_DIR/backups/
  - Removes __pycache__ directories after sync
  - Restarts Home Assistant via API

Options:
  --run              Actually perform the sync. Without --run it's a dry-run.
  --no-backup        Skip creating backup of production code
  --no-restart       Skip Home Assistant restart
  -h, --help         Show this help message

Examples:
  $(basename "$0")                    # Dry run - show what would happen
  $(basename "$0") --run              # Full sync with backup and restart
  $(basename "$0") --run --no-backup  # Sync without backup
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN=1; shift ;;
    --no-backup) BACKUP=0; shift ;;
    --no-restart) RESTART=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# Source and destination paths
SRC_DIR="$REPO_DIR/custom_components/$COMPONENT"
DST_DIR="$PROD_DIR"
BACKUP_DIR="$REPO_DIR/backups"

echo "Deploy script summary:"
echo "  Repository: $REPO_DIR"
echo "  Production: $PROD_DIR"
echo "  Source: $SRC_DIR"
echo "  Destination: $DST_DIR"
echo "  Backup: $BACKUP_DIR"
echo "  Run: $RUN  Backup: $BACKUP  Restart: $RESTART"
echo

# Validate source
if [[ ! -d "$SRC_DIR" ]]; then
  echo "ERROR: Source component not found at $SRC_DIR" >&2
  echo "Make sure you're running this from the repository root or correct path." >&2
  exit 1
fi

# Validate destination directory
if [[ ! -d "$PROD_DIR" ]]; then
  echo "ERROR: Production directory not found at $PROD_DIR" >&2
  echo "Make sure the SMB mount is accessible." >&2
  exit 1
fi

# Create backup directory if needed
if [[ $BACKUP -eq 1 ]]; then
  mkdir -p "$BACKUP_DIR"
fi

# Generate timestamp for backup
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="${COMPONENT}_backup_${TIMESTAMP}"
BACKUP_PATH="$BACKUP_DIR/$BACKUP_NAME"

echo "Planned actions:"
echo " 1. Create backup: $PROD_DIR -> $BACKUP_PATH"
echo " 2. Sync code: $SRC_DIR -> $PROD_DIR"
echo " 3. Remove __pycache__ directories from $PROD_DIR"
if [[ $RESTART -eq 1 ]]; then
  echo " 4. Restart Home Assistant via API"
fi

if [[ $RUN -ne 1 ]]; then
  echo
  echo "Dry-run mode. Rerun with --run to apply changes."
  exit 0
fi

echo
echo "Starting deployment..."

# Step 1: Create backup
if [[ $BACKUP -eq 1 ]]; then
  echo "Creating backup..."
  if [[ -d "$PROD_DIR" && "$(ls -A "$PROD_DIR" 2>/dev/null)" ]]; then
    # Make sure parent dir exists for backup path
    mkdir -p "$BACKUP_DIR"
    cp -r "$PROD_DIR" "$BACKUP_PATH"
    echo "✅ Backup created: $BACKUP_PATH"
  else
    echo "⚠️  Production directory is empty, skipping backup"
  fi
fi

# Step 2: Sync code
echo "Syncing code..."

# Create a temporary directory for the new files
TEMP_DIR="/tmp/ha_washdata_deploy_$$"
echo "Creating temporary directory: $TEMP_DIR"
mkdir -p "$TEMP_DIR"

# Copy files to temporary directory first
echo "Copying files to temporary location..."
cp -r "$SRC_DIR"/* "$TEMP_DIR/" 2>/dev/null || {
  echo "⚠️  cp failed, trying tar method..."
  (cd "$SRC_DIR" && tar -cf - --exclude='.git' --exclude='tools' --exclude='node_modules' --exclude='log' --exclude='backups' --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pyo' .) | (cd "$TEMP_DIR" && tar -xf -)
}

# Remove old destination and move new files
echo "Replacing destination directory..."
if [[ -d "$DST_DIR" ]]; then
  rm -rf "$DST_DIR"
fi
mkdir -p "$(dirname "$DST_DIR")"
mv "$TEMP_DIR" "$DST_DIR"

echo "✅ Code synced successfully"

# Step 3: Remove __pycache__ directories
echo "Cleaning up __pycache__ directories..."
find "$DST_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$DST_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$DST_DIR" -name "*.pyo" -delete 2>/dev/null || true
echo "✅ Cache files removed"

# Step 4: Restart Home Assistant
if [[ $RESTART -eq 1 ]]; then
  echo "Restarting Home Assistant..."
  
  # Make the API call and capture response
  API_RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
    -H "Authorization: Bearer $HA_API_TOKEN" \
    -H "Content-Type: application/json" \
    "$HA_API_URL" 2>/dev/null)
  
  HTTP_CODE=$(echo "$API_RESPONSE" | tail -n1)
  RESPONSE_BODY=$(echo "$API_RESPONSE" | head -n -1)
  
  if [[ "$HTTP_CODE" == "200" ]]; then
    echo "✅ Home Assistant restart initiated successfully"
  elif [[ "$HTTP_CODE" == "401" ]]; then
    echo "❌ Home Assistant restart failed - Authentication error (check API token)"
  elif [[ "$HTTP_CODE" == "404" ]]; then
    echo "❌ Home Assistant restart failed - API endpoint not found (check URL)"
  elif [[ "$HTTP_CODE" == "000" ]]; then
    echo "❌ Home Assistant restart failed - Connection error (check network/URL)"
  else
    echo "⚠️  Home Assistant restart returned HTTP $HTTP_CODE"
    if [[ -n "$RESPONSE_BODY" ]]; then
      echo "   Response: $RESPONSE_BODY"
    fi
  fi
  
  if [[ "$HTTP_CODE" != "200" ]]; then
    echo "   You may need to restart Home Assistant manually"
  fi
fi

echo
echo "🎉 Deployment completed successfully!"
echo
echo "Summary:"
if [[ $BACKUP -eq 1 ]]; then
  echo "  📦 Backup: $BACKUP_PATH"
fi
echo "  📁 Code synced: $SRC_DIR -> $DST_DIR"
echo "  🧹 Cache cleaned: __pycache__ directories removed"

if [[ $RESTART -eq 1 ]]; then
  echo "  🔄 Home Assistant: Restart initiated"
fi
echo
echo "Next steps:"
echo "  - Check Home Assistant logs for any errors"
echo "  - Test the integration functionality"