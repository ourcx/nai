#!/usr/bin/env bash
set -euo pipefail

RELEASE_ID="${1:?release id required}"
ASSET_DIR="${2:?asset directory required}"
CODE_DIR="${3:?code directory required}"
ROOT=/srv/nailong-tts
FINAL="$ROOT/releases/$RELEASE_ID"

[[ "$RELEASE_ID" =~ ^nailong-tts-[0-9]{8}-[0-9a-f]{10}$ ]]
test ! -e "$FINAL"
test -f "$ASSET_DIR/base-model/s1v3.ckpt"
test -f "$ASSET_DIR/base-model/v2Pro/s2Gv2ProPlus.pth"
test -f "$ASSET_DIR/naiwa-reference.wav"
test -f "$CODE_DIR/service.py"
test -f "$CODE_DIR/tts_engine.py"
test -f "$CODE_DIR/nailong-tts@.service"

TMP="$(mktemp -d "$ROOT/releases/.installing.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT
cp -al "$ROOT/current/." "$TMP/"
rm -f "$TMP/service.py" "$TMP/tts_engine.py" "$TMP/SHA256SUMS"
install -m 0640 "$CODE_DIR/service.py" "$TMP/service.py"
install -m 0640 "$CODE_DIR/tts_engine.py" "$TMP/tts_engine.py"
install -m 0644 "$CODE_DIR/nailong-tts@.service" "$TMP/nailong-tts@.service"
install -m 0640 "$ASSET_DIR/base-model/s1v3.ckpt" "$TMP/weights/s1v3.ckpt"
install -m 0640 \
  "$ASSET_DIR/base-model/v2Pro/s2Gv2ProPlus.pth" \
  "$TMP/weights/s2Gv2ProPlus.pth"
install -m 0640 "$ASSET_DIR/naiwa-reference.wav" "$TMP/naiwa-reference.wav"

python3 - "$TMP/release.json" "$RELEASE_ID" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
release_id = sys.argv[2]
root = metadata_path.parent
metadata = json.loads(metadata_path.read_text())
metadata["release_id"] = release_id
metadata["voices"] = {
    "nailong": {
        "gpt_weights": "weights/NaiLong-e16.ckpt",
        "sovits_weights": "weights/NaiLong_e14_s504.pth",
        "reference": "reference.wav",
        "reference_text": "啊，我才不要这样，好害羞啊。",
    },
    "naiwa": {
        "gpt_weights": "weights/s1v3.ckpt",
        "sovits_weights": "weights/s2Gv2ProPlus.pth",
        "reference": "naiwa-reference.wav",
        "reference_text": "我落地枪，不要慌，全靠运气全靠浪。",
    },
}
known = {item["file"] for item in metadata["weights"]}
for relative in (
    "weights/s1v3.ckpt",
    "weights/s2Gv2ProPlus.pth",
    "naiwa-reference.wav",
):
    if relative in known:
        continue
    metadata["weights"].append({
        "file": relative,
        "sha256": hashlib.sha256((root / relative).read_bytes()).hexdigest(),
    })
metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
PY

(
  cd "$TMP"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    | sed 's#  \./#  #' > SHA256SUMS
)
chown -R root:nailong-tts "$TMP"
chmod -R u=rwX,g=rX,o= "$TMP"
mv -T "$TMP" "$FINAL"
trap - EXIT

PREVIOUS="$(readlink "$ROOT/current")"
ln -sfn "$FINAL" "$ROOT/current.next"
mv -Tf "$ROOT/current.next" "$ROOT/current"
install -m 0644 "$FINAL/nailong-tts@.service" /etc/systemd/system/nailong-tts@.service
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
systemctl daemon-reload
systemctl disable --now nailong-tts.service 2>/dev/null || true
systemctl enable nailong-tts@nailong.service nailong-tts@naiwa.service
systemctl restart nailong-tts@nailong.service nailong-tts@naiwa.service

for _ in $(seq 1 90); do
  if curl -fsS http://127.0.0.1:9880/healthz \
    && curl -fsS http://127.0.0.1:9881/healthz; then
    exit 0
  fi
  sleep 2
done

ln -sfn "$PREVIOUS" "$ROOT/current.next"
mv -Tf "$ROOT/current.next" "$ROOT/current"
systemctl restart nailong-tts@nailong.service nailong-tts@naiwa.service
exit 1
