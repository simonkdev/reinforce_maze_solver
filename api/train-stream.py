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
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            config_payload = payload.get("config", {})
            config = TrainConfig(
                epochs=int(config_payload.get("epochs", 120)),
                sampling_quantity=int(config_payload.get("sampling_quantity", 64)),
                episode_limit=int(config_payload.get("episode_limit", 100)),
                eval_episodes=int(config_payload.get("eval_episodes", 100)),
                seed=_optional_int(config_payload.get("seed")),
            )

            def on_progress(event):
                self._send_line({"type": "progress", "data": event})

            result = train_from_matrix(payload["maze"], config, on_progress)
            self._send_line({"type": "done", "data": result})
        except Exception as exc:
            self._send_line({"type": "error", "error": str(exc)})

    def _send_line(self, payload):
        self.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))
        self.wfile.flush()


def _optional_int(value):
    if value in (None, ""):
        return None
    return int(value)
