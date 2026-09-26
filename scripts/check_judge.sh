#!/usr/bin/env bash
# Assert the judge endpoint is up AND returning content (not an empty reasoning
# block). A silent content=None is the most common way this pipeline fails.
set -euo pipefail
[ -f ".env" ] && set -a && . .env && set +a
BASE=${VLLM_BASE_URL:-http://localhost:8061/v1}
MODEL=${VLLM_MODEL:-Qwen3-4B}
python3 - "$BASE" "$MODEL" <<'PYCHECK'
import json, sys, urllib.request
base, model = sys.argv[1].rstrip("/"), sys.argv[2]
req = urllib.request.Request(
    f"{base}/chat/completions",
    data=json.dumps({"model": model, "temperature": 0, "max_tokens": 64,
                     "messages": [{"role": "user",
                                   "content": "What is 17 * 24? Reply with just the number."}]}).encode(),
    headers={"Content-Type": "application/json"})
m = json.load(urllib.request.urlopen(req))["choices"][0]["message"]
if not (m.get("content") or "").strip():
    sys.exit("FATAL: content empty -- the chat-template / reasoning-parser combination is wrong")
print("OK: judge reachable and returning content ->", repr(m["content"])[:60])
PYCHECK
