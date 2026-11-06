#!/usr/bin/env bash
set -euo pipefail
HOST="${1:-tx}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
RELEASE_ID="$(python3 - <<'PY'
import hashlib
from pathlib import Path
digest = hashlib.sha256()
for path in sorted(Path('src').rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
print(digest.hexdigest()[:12])
PY
)"
REMOTE_SOURCE="/opt/nailong-bot/voice-releases/$RELEASE_ID/src"
ssh "$HOST" "test -f /etc/nailong-tts-client.env; install -d -m 0755 '$REMOTE_SOURCE'"
rsync -az --exclude __pycache__ --exclude '*.pyc' src/ "$HOST:$REMOTE_SOURCE/"
ssh "$HOST" bash -s -- "$REMOTE_SOURCE" <<'SH'
set -euo pipefail
SOURCE="$1"
chmod -R a+rX "$SOURCE"
/opt/nailong-bot/.venv/bin/python -m pip --disable-pip-version-check install --upgrade --no-deps 'av==16.1.0' 'yt-dlp>=2025.1'
PYTHONPATH="$SOURCE" /opt/nailong-bot/.venv/bin/python -c 'from nailong_classifier.application.bot import main; import av; assert "libopus" in av.codecs_available'
DROPIN=/etc/systemd/system/nailong-bot.service.d/90-voice-client.conf
install -d -m 0755 "$(dirname "$DROPIN")" /var/backups/nailong-voice-client
BACKUP="$(mktemp /var/backups/nailong-voice-client/dropin.XXXXXX)"
HAD_PREVIOUS=false
if [[ -f "$DROPIN" ]]; then
  cp -p "$DROPIN" "$BACKUP"
  HAD_PREVIOUS=true
fi
cat > "$DROPIN" <<CONF
[Service]
EnvironmentFile=/etc/nailong-tts-client.env
Environment=PYTHONPATH=$SOURCE
ExecStart=
ExecStart=/opt/nailong-bot/.venv/bin/python -m nailong_classifier.application.bot
CONF
systemctl daemon-reload
systemctl restart nailong-bot
# 模型启动较慢，检查同一进程持续存活，避免把反复崩溃重启当作成功。
PID="$(systemctl show nailong-bot -p MainPID --value)"
sleep 15
if ! systemctl is-active --quiet nailong-bot || [[ "$PID" == 0 || "$PID" != "$(systemctl show nailong-bot -p MainPID --value)" ]]; then
  if "$HAD_PREVIOUS"; then cp -p "$BACKUP" "$DROPIN"; else rm -f "$DROPIN"; fi
  systemctl daemon-reload
  systemctl restart nailong-bot
  echo 'Voice client startup failed; restored previous bot entry point' >&2
  exit 1
fi
systemctl show nailong-bot -p ActiveState -p SubState -p MainPID -p MemoryCurrent
systemctl is-active nailong-web
SH
