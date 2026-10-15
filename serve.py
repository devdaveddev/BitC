"""
serve.py – Single-process entry point for hosting (e.g. Render).

Runs the dashboard on $PORT and keeps a 4-peer swarm cycling in the
background, so the live dashboard always has real activity. Each cycle
regenerates the shared file and restarts the peers.

Usage: python serve.py   (env: PORT, CYCLE_SECONDS, PIECES)
"""

import glob
import os
import signal
import subprocess
import sys
import threading
import time
from functools import partial
from http.server import ThreadingHTTPServer

import dashboard
from utils import generate_test_file, PIECE_SIZE

ROOT = os.path.dirname(os.path.abspath(__file__))
CYCLE_SECONDS = int(os.environ.get('CYCLE_SECONDS', 240))
PIECES = int(os.environ.get('PIECES', 48))
PEERS = [(5001, 'seed'), (5002, 'normal'), (5003, 'freerider'), (5004, 'adaptive')]

stop = threading.Event()


def run_cycle():
    generate_test_file(os.path.join(ROOT, 'testfile.txt'), size_bytes=PIECES * PIECE_SIZE)
    for stale in glob.glob(os.path.join(ROOT, 'telemetry', 'peer_*.json')):
        os.remove(stale)
    procs = []
    for port, role in PEERS:
        procs.append(subprocess.Popen(
            [sys.executable, os.path.join(ROOT, 'peer.py'), '--port', str(port), '--role', role],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT))
        if stop.wait(2):
            break
    stop.wait(CYCLE_SECONDS)
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def swarm_loop():
    while not stop.is_set():
        try:
            run_cycle()
        except Exception as e:  # never let the background loop die
            print(f"swarm cycle error: {e}", flush=True)
            stop.wait(5)


def main():
    port = int(os.environ.get('PORT', 8765))
    threading.Thread(target=swarm_loop, daemon=True).start()
    server = ThreadingHTTPServer(('0.0.0.0', port),
                                 partial(dashboard.Handler, directory=dashboard.WEB_DIR))
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    print(f"SwarmLab serving on 0.0.0.0:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()


if __name__ == '__main__':
    main()
