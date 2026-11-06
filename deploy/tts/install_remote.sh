#!/usr/bin/env bash
set -euo pipefail
exec 9>/var/lock/nailong-tts-install.lock
flock 9

RELEASE_SOURCE="${1:?Usage: install_remote.sh RELEASE_DIRECTORY}"
SERVICE_ROOT=/srv/nailong-tts
RELEASE_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["release_id"])' "$RELEASE_SOURCE/release.json")"
[[ "$RELEASE_ID" =~ ^nailong-tts-[0-9]{8}-[0-9a-f]{10}$ ]]
cd "$RELEASE_SOURCE"
sha256sum -c SHA256SUMS >/dev/null

if ! id nailong-tts >/dev/null 2>&1; then
  useradd --system --home-dir /var/lib/nailong-tts --shell /usr/sbin/nologin nailong-tts
fi
install -d -m 0755 "$SERVICE_ROOT" "$SERVICE_ROOT/releases"
RELEASE_TARGET="$SERVICE_ROOT/releases/$RELEASE_ID"
if [[ ! -d "$RELEASE_TARGET" ]]; then
  # 中断复制不会留下一个看似已安装、实际缺文件的发布目录。
  COPYING_TARGET="$(mktemp -d "$SERVICE_ROOT/releases/.installing.XXXXXX")"
  rsync -a "$RELEASE_SOURCE/" "$COPYING_TARGET/"
  (cd "$COPYING_TARGET" && sha256sum -c SHA256SUMS >/dev/null)
  chown -R root:nailong-tts "$COPYING_TARGET"
  chmod -R u=rwX,g=rX,o= "$COPYING_TARGET"
  mv -T "$COPYING_TARGET" "$RELEASE_TARGET"
fi
cd "$RELEASE_TARGET"
sha256sum -c SHA256SUMS >/dev/null

# 只给本服务建立独立依赖，避免改变现有机器人或系统 Python 环境。
if [[ ! -x "$SERVICE_ROOT/.venv/bin/python" ]]; then
  python3 -m venv --without-pip "$SERVICE_ROOT/.venv"
  python3 -m pip --python "$SERVICE_ROOT/.venv/bin/python" install pip
fi
if [[ -n "${TTS_WHEELHOUSE:-}" ]]; then
  # 开发机访问 PyPI 较慢时，使用本机预先下载的 Linux 依赖；不改变依赖版本。
  "$SERVICE_ROOT/.venv/bin/python" -m pip install --no-index --find-links "$TTS_WHEELHOUSE" setuptools wheel
  "$SERVICE_ROOT/.venv/bin/python" -m pip install --no-index --no-build-isolation \
    --find-links "$TTS_WHEELHOUSE" 'torch==2.8.0' 'torchaudio==2.8.0' \
    -r "$RELEASE_TARGET/requirements.txt"
else
  "$SERVICE_ROOT/.venv/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0' 'torchaudio==2.8.0'
  "$SERVICE_ROOT/.venv/bin/python" -m pip install -r "$RELEASE_TARGET/requirements.txt"
fi
"$SERVICE_ROOT/.venv/bin/python" -m pip freeze > "$SERVICE_ROOT/requirements.installed.txt"

if [[ ! -f /etc/nailong-tts.env ]]; then
  umask 0077
  python3 - <<'PY'
from pathlib import Path
import secrets
Path('/etc/nailong-tts.env').write_text('TTS_API_TOKEN=' + secrets.token_urlsafe(32) + '\n')
PY
fi
chown root:nailong-tts /etc/nailong-tts.env
chmod 0640 /etc/nailong-tts.env
cat > /etc/nailong-tts-nailong.env <<'ENV'
TTS_VOICE=nailong
TTS_PORT=9880
TTS_STATE_DIR=/var/lib/nailong-tts-nailong
TTS_MAX_WAITING=0
ENV
cat > /etc/nailong-tts-naiwa.env <<'ENV'
TTS_VOICE=naiwa
TTS_PORT=9881
TTS_STATE_DIR=/var/lib/nailong-tts-naiwa
TTS_MAX_WAITING=0
ENV
chown root:nailong-tts /etc/nailong-tts-nailong.env /etc/nailong-tts-naiwa.env
chmod 0640 /etc/nailong-tts-nailong.env /etc/nailong-tts-naiwa.env

PREVIOUS="$(readlink "$SERVICE_ROOT/current" || true)"
if [[ -e "$SERVICE_ROOT/current" && ! -L "$SERVICE_ROOT/current" ]]; then
  echo "Refusing to replace a non-symlink current path" >&2
  exit 1
fi
ln -sfn "$RELEASE_TARGET" "$SERVICE_ROOT/current.next"
mv -Tf "$SERVICE_ROOT/current.next" "$SERVICE_ROOT/current"
install -m 0644 "$RELEASE_TARGET/nailong-tts@.service" /etc/systemd/system/nailong-tts@.service
systemctl daemon-reload
systemctl disable --now nailong-tts.service 2>/dev/null || true
systemctl enable nailong-tts@nailong.service nailong-tts@naiwa.service
systemctl restart nailong-tts@nailong.service nailong-tts@naiwa.service

if ! python3 - <<'PY'
import json
import time
from urllib.request import urlopen
for _ in range(60):
    try:
        with (
            urlopen('http://127.0.0.1:9880/healthz', timeout=2) as nailong,
            urlopen('http://127.0.0.1:9881/healthz', timeout=2) as naiwa,
        ):
            if (
                json.load(nailong).get('voice') == 'nailong'
                and json.load(naiwa).get('voice') == 'naiwa'
            ):
                print('TTS health check passed')
                raise SystemExit(0)
    except (OSError, ValueError, KeyError):
        pass
    time.sleep(2)
raise SystemExit(1)
PY
then
  systemctl stop nailong-tts@nailong.service nailong-tts@naiwa.service
  if [[ -n "$PREVIOUS" ]]; then
    ln -sfn "$PREVIOUS" "$SERVICE_ROOT/current.next"
    mv -Tf "$SERVICE_ROOT/current.next" "$SERVICE_ROOT/current"
    systemctl start nailong-tts@nailong.service nailong-tts@naiwa.service
  fi
  echo 'TTS startup failed; inspect journalctl -u nailong-tts' >&2
  exit 1
fi
systemctl show nailong-tts@nailong.service nailong-tts@naiwa.service \
  -p ActiveState -p SubState -p MemoryCurrent -p MemoryPeak
