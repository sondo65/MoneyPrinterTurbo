#!/usr/bin/env sh
# Daily Tử Vi video Telegram bot. Needs the VieNeu TTS server
# (http://127.0.0.1:8000/v1) and [tuvi] + deepseek_api_key in config.toml.
set -e
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ] || ! .venv/bin/python -c "import telegram" >/dev/null 2>&1; then
  uv sync --extra tuvi-bot
fi

VIENEU_URL="${VIENEU_URL:-http://127.0.0.1:8000/v1}"
if ! curl -sf "$VIENEU_URL/voices" >/dev/null 2>&1; then
  echo "Warning: VieNeu TTS server is not reachable at $VIENEU_URL; videos will fail until it is started."
fi

exec .venv/bin/python -m app.bot.telegram_bot
