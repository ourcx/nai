#!/usr/bin/env bash
set -euo pipefail

BOT_HOST="${1:-tx}"
PROXY_HOST="${2:-devbox}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 复用现有源码发布流程，保留线上模型、数据库、语音配置与 bot 入口回滚。
# 网页或代理阶段失败后，可以复用已经验证稳定的机器人版本继续发布。
if [[ "${DEPLOY_GENERATOR_BOT:-1}" == 1 ]]; then
  bash "$PROJECT_DIR/scripts/deploy_voice_client.sh" "$BOT_HOST"
fi

ssh "$BOT_HOST" bash -s <<'REMOTE'
set -euo pipefail
SOURCE="$(/opt/nailong-bot/.venv/bin/python - <<'PY'
import shlex
import subprocess
environment = subprocess.check_output(
    ["systemctl", "show", "nailong-bot", "-p", "Environment", "--value"], text=True
)
print(next(value.split("=", 1)[1] for value in shlex.split(environment)
           if value.startswith("PYTHONPATH=")))
PY
)"
test -f "$SOURCE/nailong_classifier/imaging/assets/nailong-avatar.png"
test -f "$SOURCE/nailong_classifier/imaging/assets/naiwa-avatar.png"
PYTHONPATH="$SOURCE" /opt/nailong-bot/.venv/bin/python - <<'PY'
from nailong_classifier.imaging.milk_text import parse_milk_text_command, render_milk_text
from nailong_classifier.web.app import main
assert callable(main)
options = parse_milk_text_command("奶峰生成器 鸡腿拿来！")
assert options and options.character == options.background == options.motion == "random"
assert render_milk_text(options).startswith((b"GIF8", b"\x89PNG\r\n\x1a\n"))
options = parse_milk_text_command(
    "奶峰生成器 鸡腿拿来！ --配色 黄边白字 --头像 8 --动作 静态"
)
assert render_milk_text(options).startswith(b"\x89PNG\r\n\x1a\n")
print("New command and bundled assets verified")
PY

DROPIN=/etc/systemd/system/nailong-web.service.d/90-generator.conf
install -d -m 0755 "$(dirname "$DROPIN")" /var/backups/nailong-generator
BACKUP="$(mktemp /var/backups/nailong-generator/web-dropin.XXXXXX)"
HAD_PREVIOUS=false
if [[ -f "$DROPIN" ]]; then
  cp -p "$DROPIN" "$BACKUP"
  HAD_PREVIOUS=true
fi
cat > "$DROPIN" <<CONF
[Service]
Environment=PYTHONPATH=$SOURCE
ExecStart=
ExecStart=/opt/nailong-bot/.venv/bin/python -m nailong_classifier.web.app
CONF
systemctl daemon-reload
systemctl restart nailong-web
if ! curl --silent --show-error --fail --retry 6 --retry-connrefused --retry-delay 1 \
  http://127.0.0.1:8765/text/ | /opt/nailong-bot/.venv/bin/python -c \
  'import sys; assert "奶峰生成器" in sys.stdin.read()'; then
  if "$HAD_PREVIOUS"; then cp -p "$BACKUP" "$DROPIN"; else rm -f "$DROPIN"; fi
  systemctl daemon-reload
  systemctl restart nailong-web
  echo "Generator web startup failed; restored previous entry point" >&2
  exit 1
fi
systemctl show nailong-bot nailong-web -p Id -p ActiveState -p MainPID -p NRestarts
REMOTE

# 仅更新既有图库代理代码，新增受限的 PNG 生成请求转发。
STAGE=".cache/nailong-generator-deploy"
ssh "$PROXY_HOST" "install -d -m 0700 '$STAGE'"
scp -q "$PROJECT_DIR/deploy/nailong-devbox-proxy.py" "$PROXY_HOST:$STAGE/proxy.py"
ssh "$PROXY_HOST" bash -s <<'REMOTE'
set -euo pipefail
SOURCE="$HOME/.cache/nailong-generator-deploy/proxy.py"
TARGET=/opt/nailong-gallery-proxy/proxy.py
BACKUP="$(sudo -n mktemp /opt/nailong-gallery-proxy/proxy-backup.XXXXXX)"
sudo -n cp -p "$TARGET" "$BACKUP"
sudo -n install -m 0644 "$SOURCE" "$TARGET"
sudo -n systemctl restart nailong-gallery-proxy
if ! curl --silent --show-error --fail --retry 6 --retry-connrefused --retry-delay 1 \
  http://127.0.0.1:8765/text/ | python3 -c \
  'import sys; assert "奶峰生成器" in sys.stdin.read()'; then
  sudo -n cp -p "$BACKUP" "$TARGET"
  sudo -n systemctl restart nailong-gallery-proxy
  echo "Generator proxy startup failed; restored previous proxy code" >&2
  exit 1
fi
sudo -n systemctl show nailong-gallery-proxy -p ActiveState -p MainPID -p NRestarts
REMOTE
