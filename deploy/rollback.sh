#!/usr/bin/env bash
# 在 tx 上以 root 执行。默认保留上线后的新聊天记录，只恢复代码、权重、依赖和配置。
set -euo pipefail

BACKUP_DIR="${1:?Usage: bash rollback.sh /var/backups/nailong-bot/full-TIMESTAMP}"
test -f "$BACKUP_DIR/SHA256SUMS"
sha256sum -c "$BACKUP_DIR/SHA256SUMS"
tar -tzf "$BACKUP_DIR/runtime.tar.gz" >/dev/null
systemctl stop nailong-bot nailong-web
tar -xzf "$BACKUP_DIR/runtime.tar.gz" -C /
systemctl daemon-reload
systemctl start nailong-bot nailong-web
systemctl is-active nailong-bot nailong-web
printf 'Restored runtime from %s; current database and assets preserved.\n' "$BACKUP_DIR"
