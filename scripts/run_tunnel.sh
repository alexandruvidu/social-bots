#!/usr/bin/env bash
# Starts a cloudflared quick tunnel pointed at the local webhook server, then
# pushes the resulting URL to Meta as the webhook callback URL. Runs under
# systemd (social-bots-tunnel.service) so it auto-restarts and re-registers
# on every restart/reboot (quick tunnel hostnames are random per run).
set -euo pipefail
cd "$(dirname "$0")/.."

cloudflared tunnel --url http://localhost:5000 --metrics 127.0.0.1:20241 &
CLOUDFLARED_PID=$!
trap 'kill "$CLOUDFLARED_PID" 2>/dev/null' EXIT

.venv/bin/python -m bot.update_webhook

wait "$CLOUDFLARED_PID"
