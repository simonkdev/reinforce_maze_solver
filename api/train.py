from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.shared.training import TrainConfig, train_from_matrix


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            config_payload = payload.get("config", {})
            result = train_from_matrix(
                payload["maze"],
                TrainConfig(
                    epochs=int(config_payload.get("epochs", 120)),
                    sampling_quantity=int(config_payload.get("sampling_quantity", 64)),
                    episode_limit=int(config_payload.get("episode_limit", 100)),
                    eval_episodes=int(config_payload.get("eval_episodes", 100)),
                    seed=_optional_int(config_payload.get("seed")),
                ),
            )
            self._send_json(result)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _optional_int(value):
    if value in (None, ""):
        return None
    return int(value)
