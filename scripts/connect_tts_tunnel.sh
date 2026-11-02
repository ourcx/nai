#!/usr/bin/env bash
# 使用现有 SSH 配置连接两台机器；私钥始终留在开发机，密钥正文不进入日志。
set -euo pipefail
DEVBOX="${1:-devbox}"
TX="${2:-tx}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ssh "$DEVBOX" 'sudo -n bash -s' <<'SH'
set -eu
id nailong-tts >/dev/null
install -d -m 0750 -o nailong-tts -g nailong-tts /var/lib/nailong-tts
install -d -m 0700 -o nailong-tts -g nailong-tts /var/lib/nailong-tts/ssh
if [ ! -f /var/lib/nailong-tts/ssh/id_ed25519 ]; then
  runuser -u nailong-tts -- ssh-keygen -q -t ed25519 -N '' -C nailong-tts-devbox -f /var/lib/nailong-tts/ssh/id_ed25519
fi
SH

ssh "$TX" 'bash -s' <<'SH'
set -eu
id nailong-tts-tunnel >/dev/null 2>&1 || useradd --system --create-home --home-dir /var/lib/nailong-tts-tunnel --shell /usr/sbin/nologin nailong-tts-tunnel
install -d -m 0700 -o nailong-tts-tunnel -g nailong-tts-tunnel /var/lib/nailong-tts-tunnel/.ssh
cat > /etc/ssh/sshd_config.d/90-nailong-tts-tunnel.conf <<'CONF'
Match User nailong-tts-tunnel
    AllowTcpForwarding remote
    PermitListen 127.0.0.1:19880
    PermitOpen none
    MaxSessions 0
    PermitTTY no
    X11Forwarding no
    AllowAgentForwarding no
Match all
CONF
sshd -t
systemctl reload sshd
SH

ssh "$DEVBOX" 'sudo -n cat /var/lib/nailong-tts/ssh/id_ed25519.pub' |
  ssh "$TX" 'umask 077; { printf '\''restrict,port-forwarding,permitlisten="127.0.0.1:19880" '\''; cat; } > /var/lib/nailong-tts-tunnel/.ssh/authorized_keys; chown nailong-tts-tunnel:nailong-tts-tunnel /var/lib/nailong-tts-tunnel/.ssh/authorized_keys'

# 经已信任的管理连接取得服务器公钥，不接受首次连接时未经核实的主机密钥。
ssh "$TX" 'cat /etc/ssh/ssh_host_ed25519_key.pub' |
  ssh "$DEVBOX" 'sudo -n sh -c '\''awk "{print \"1.14.151.75 \" \$1 \" \" \$2}" > /var/lib/nailong-tts/ssh/known_hosts; chown nailong-tts:nailong-tts /var/lib/nailong-tts/ssh/known_hosts; chmod 0600 /var/lib/nailong-tts/ssh/known_hosts'\'''

# 只传输该服务的令牌到客户端专用文件，不覆盖原有机器人配置。
ssh "$DEVBOX" 'sudo -n cat /etc/nailong-tts.env' |
  ssh "$TX" 'umask 077; { printf "NAILONG_TTS_URL=http://127.0.0.1:19880\n"; sed "s/^TTS_API_TOKEN=/NAILONG_TTS_TOKEN=/"; } > /etc/nailong-tts-client.env'

scp -q "$PROJECT_DIR/deploy/tts/nailong-tts-tunnel.service" "$DEVBOX:.cache/nailong-tts-deploy/"
ssh "$DEVBOX" 'sudo -n install -m 0644 ~/.cache/nailong-tts-deploy/nailong-tts-tunnel.service /etc/systemd/system/nailong-tts-tunnel.service; sudo -n systemctl daemon-reload; sudo -n systemctl enable --now nailong-tts-tunnel.service'
ssh "$TX" 'python3 -c "from urllib.request import urlopen; print(urlopen(\"http://127.0.0.1:19880/healthz\", timeout=10).read().decode())"'
