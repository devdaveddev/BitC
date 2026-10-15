"""
dashboard.py – Serves the SwarmLab web UI and a JSON feed of live peer state.

Reads the per-peer snapshots that peer.py writes to ./telemetry and exposes
them at /api/state. Static files are served from ./web. Localhost only.

Usage: python dashboard.py [--port 8765]
"""

import argparse
import glob
import json
import os
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(ROOT, 'web')
TELEMETRY_DIR = os.path.join(ROOT, 'telemetry')


def read_state():
    """Aggregate all peer snapshots; tolerate partial/missing files."""
    peers = []
    for path in sorted(glob.glob(os.path.join(TELEMETRY_DIR, 'peer_*.json'))):
        try:
            with open(path) as f:
                peers.append(json.load(f))
        except (OSError, json.JSONDecodeError):
            continue  # file mid-replace or removed; next poll will get it
    return {'now': time.time(), 'peers': peers}


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split('?')[0] == '/api/state':
            body = json.dumps(read_state()).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, *args):  # keep the terminal quiet
        pass


def main():
    parser = argparse.ArgumentParser(description='SwarmLab dashboard server')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port),
                                 partial(Handler, directory=WEB_DIR))
    print(f"SwarmLab dashboard on http://127.0.0.1:{args.port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
