#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-tx}"
REMOTE_DIR="${2:-/opt/nailong-bot}"
SERVICE_NAME="nailong-bot"
WEB_SERVICE_NAME="nailong-web"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "$PROJECT_DIR/artifacts/model.pt" ]]; then
  echo "artifacts/model.pt is missing; train the model before deployment." >&2
  exit 1
fi
if [[ ! -f "$PROJECT_DIR/artifacts/milk-release-20261031/manifest.json" ]]; then
  echo "Milk release manifest is missing; build the verified release before deployment." >&2
  exit 1
fi

ssh "$HOST" "mkdir -p '$REMOTE_DIR'"
rsync -az --delete \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='.env' \
  --exclude='.env.*' \
  --exclude='*cookies*.json' \
  --exclude='*.sqlite3*' \
  --exclude='backups/' \
  --exclude='data/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.marscode.internal/' \
  --exclude='.screenshots/' \
  --exclude='.DS_Store' \
  --exclude='outputs/' \
  --include='/artifacts/' \
  --include='/artifacts/model.pt' \
  --include='/artifacts/doubao-detector-20261026/***' \
  --include='/artifacts/milk-release-20261031/***' \
  --exclude='/artifacts/*' \
  "$PROJECT_DIR/" "$HOST:$REMOTE_DIR/"

ssh "$HOST" bash -s -- "$REMOTE_DIR" "$SERVICE_NAME" "$WEB_SERVICE_NAME" <<'REMOTE'
set -euo pipefail

REMOTE_DIR="$1"
SERVICE_NAME="$2"
WEB_SERVICE_NAME="$3"

if ! id -u "$SERVICE_NAME" >/dev/null 2>&1; then
  useradd --system --home-dir "$REMOTE_DIR" --shell /sbin/nologin "$SERVICE_NAME"
fi

python3 -m venv "$REMOTE_DIR/.venv"
"$REMOTE_DIR/.venv/bin/python" -m pip install --upgrade pip
"$REMOTE_DIR/.venv/bin/python" -m pip install \
  lark-channel-sdk \
  "httpx>=0.27,<1" \
  "websockets>=15,<16" \
  Pillow \
  "starlette>=0.48,<1" \
  "uvicorn>=0.35,<1" \
  filelock \
  typing-extensions \
  "setuptools>=77" \
  sympy \
  networkx \
  jinja2 \
  fsspec \
  "numpy>=2.0" \
  "imagecodecs>=2025.3.30" \
  "timm==1.0.22" \
  "safetensors==0.6.2" \
  "huggingface-hub==0.36.0"
"$REMOTE_DIR/.venv/bin/python" -m pip install \
  --no-deps \
  --index-url https://download.pytorch.org/whl/cpu \
  torch torchvision
"$REMOTE_DIR/.venv/bin/python" -m pip install --no-deps -e "$REMOTE_DIR"

chown -R root:"$SERVICE_NAME" "$REMOTE_DIR"
chmod -R g+rX "$REMOTE_DIR"
install -m 0644 \
  "$REMOTE_DIR/deploy/nailong-bot.service" \
  "/etc/systemd/system/$SERVICE_NAME.service"
install -m 0644 \
  "$REMOTE_DIR/deploy/nailong-web.service" \
  "/etc/systemd/system/$WEB_SERVICE_NAME.service"
systemctl daemon-reload

if [[ -s "/etc/$SERVICE_NAME.env" ]]; then
  chown root:"$SERVICE_NAME" "/etc/$SERVICE_NAME.env"
  chmod 0640 "/etc/$SERVICE_NAME.env"
  systemctl enable --now "$SERVICE_NAME"
  systemctl restart "$SERVICE_NAME"
  systemctl enable --now "$WEB_SERVICE_NAME"
  systemctl restart "$WEB_SERVICE_NAME"
  systemctl --no-pager --full status "$SERVICE_NAME"
  systemctl --no-pager --full status "$WEB_SERVICE_NAME"
else
  systemctl disable --now "$SERVICE_NAME" >/dev/null 2>&1 || true
  systemctl disable --now "$WEB_SERVICE_NAME" >/dev/null 2>&1 || true
  echo "Code deployed. Add /etc/$SERVICE_NAME.env before starting the service."
fi
REMOTE
