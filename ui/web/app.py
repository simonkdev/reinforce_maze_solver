from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.shared.training import TrainConfig, default_maze, random_maze, train_from_matrix


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def do_GET(self):
        if self.path == "/api/default-maze":
            self._send_json({"maze": default_maze()})
            return
        if self.path == "/api/random-maze":
            self._send_json({"maze": random_maze()})
            return
        super().do_GET()

    def do_POST(self):
        if self.path not in {"/api/train", "/api/train-stream"}:
            self.send_error(404)
            return

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
            if self.path == "/api/train-stream":
                self._send_stream(payload["maze"], config)
                return
            result = train_from_matrix(payload["maze"], config)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        self._send_json(result)

    def _send_stream(self, maze, config):
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def on_progress(event):
            self._send_json_line({"type": "progress", "data": event})

        try:
            result = train_from_matrix(maze, config, on_progress)
            self._send_json_line({"type": "done", "data": result})
        except Exception as exc:
            self._send_json_line({"type": "error", "error": str(exc)})

    def _send_json_line(self, payload):
        self.wfile.write((json.dumps(payload) + "\n").encode("utf-8"))
        self.wfile.flush()

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


def main():
    server = ThreadingHTTPServer(("127.0.0.1", 1212), Handler)
    print("Maze UI running at http://127.0.0.1:1212")
    server.serve_forever()


if __name__ == "__main__":
    main()
