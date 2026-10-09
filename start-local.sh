#!/usr/bin/env sh
set -e
cd "$(dirname "$0")"
export PATH="/Applications/Ollama.app/Contents/Resources:$PATH"
if ! curl -sf http://127.0.0.1:11434/ >/dev/null 2>&1; then
  echo "Starting Ollama..."
  open -a Ollama 2>/dev/null || ollama serve >/tmp/ollama-serve.log 2>&1 &
  for i in 1 2 3 4 5 6 7 8 9 10; do
    curl -sf http://127.0.0.1:11434/ >/dev/null 2>&1 && break
    sleep 1
  done
fi
MPT_WEBUI_HOST=127.0.0.1 sh webui.sh
