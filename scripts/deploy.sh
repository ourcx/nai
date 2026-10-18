#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-tx}"
REMOTE_DIR="${2:-/opt/nailong-bot}"
SERVICE_NAME="nailong-bot"

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ ! -f artifacts/model.pt ]]; then
  echo "artifacts/model.pt is missing; train the model before deployment." >&2
  exit 1
fi

ssh "$HOST" "mkdir -p '$REMOTE_DIR'"
rsync -az --delete \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='.env' \
  --exclude='data/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  ./ "$HOST:$REMOTE_DIR/"

ssh "$HOST" bash -s -- "$REMOTE_DIR" "$SERVICE_NAME" <<'REMOTE'
set -euo pipefail

REMOTE_DIR="$1"
SERVICE_NAME="$2"

if ! id -u "$SERVICE_NAME" >/dev/null 2>&1; then
  useradd --system --home-dir "$REMOTE_DIR" --shell /sbin/nologin "$SERVICE_NAME"
fi

python3 -m venv "$REMOTE_DIR/.venv"
"$REMOTE_DIR/.venv/bin/python" -m pip install --upgrade pip
"$REMOTE_DIR/.venv/bin/python" -m pip install \
  lark-channel-sdk \
  Pillow \
  filelock \
  typing-extensions \
  "setuptools>=77" \
  sympy \
  networkx \
  jinja2 \
  fsspec \
  numpy
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
systemctl daemon-reload

if [[ -s "/etc/$SERVICE_NAME.env" ]]; then
  chown root:"$SERVICE_NAME" "/etc/$SERVICE_NAME.env"
  chmod 0640 "/etc/$SERVICE_NAME.env"
  systemctl enable --now "$SERVICE_NAME"
  systemctl restart "$SERVICE_NAME"
  systemctl --no-pager --full status "$SERVICE_NAME"
else
  systemctl disable --now "$SERVICE_NAME" >/dev/null 2>&1 || true
  echo "Code deployed. Add /etc/$SERVICE_NAME.env before starting the service."
fi
REMOTE
