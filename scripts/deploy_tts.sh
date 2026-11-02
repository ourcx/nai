#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-devbox}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_DIR="${2:-$PROJECT_DIR/artifacts/tts-release}"
test -f "$RELEASE_DIR/SHA256SUMS"
RELEASE_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["release_id"])' "$RELEASE_DIR/release.json")"
[[ "$RELEASE_ID" =~ ^nailong-tts-[0-9]{8}-[0-9a-f]{10}$ ]]
STAGE=".cache/nailong-tts-deploy/$RELEASE_ID"
ssh "$HOST" "umask 077; mkdir -p '$STAGE/release'"
rsync -az --chmod=u=rwX,go= "$RELEASE_DIR/" "$HOST:$STAGE/release/"
rsync -az --chmod=u=rwX,go= "$PROJECT_DIR/deploy/tts/install_remote.sh" "$HOST:$STAGE/install_remote.sh"
ssh "$HOST" "sudo -n bash '$STAGE/install_remote.sh' \"\$HOME/$STAGE/release\""
