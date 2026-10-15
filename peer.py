"""
peer.py – Main entry point for the BitTorrent P2P simulation.

Orchestrates file sharing between peers with different roles:
  - SEED:     Has the full file, only uploads
  - NORMAL:   Uses TIT_FOR_TAT strategy
  - FREERIDER: Downloads but rarely uploads
  - ADAPTIVE: Uses ML-based strategy switching every 30 seconds

Usage:
  python peer.py --port 5001 --role seed
  python peer.py --port 5002 --role normal
  python peer.py --port 5003 --role freerider
  python peer.py --port 5004 --role adaptive
"""

import argparse
import json
import logging
import os
import random
import signal
import sys
import threading
import time
from collections import deque

from utils import split_file, create_piece_hashes, verify_piece, generate_test_file, PIECE_SIZE
from protocol import (
    MessageType, create_request, create_piece, create_have,
    create_reject, extract_piece_data
)
from network import PeerServer
from metrics import SwarmMetrics
from strategies import Strategy, get_strategy, get_strategy_name
from ml_model import load_model, train_model, predict_with_confidence



DEFAULT_PEERS = [5001, 5002, 5003, 5004]
TEST_FILE = 'testfile.txt'
ADAPTIVE_INTERVAL = 30  # seconds between strategy re-evaluation
FREERIDER_UPLOAD_CHANCE = 0.05  # 5% chance freerider actually serves a piece
DOWNLOAD_LOOP_INTERVAL = 1  # seconds between download attempts
HAVE_BROADCAST_INTERVAL = 10  # seconds between HAVE broadcasts
REQUEST_TIMEOUT = 15.0  # seconds before an outstanding request is abandoned
REJECT_COOLDOWN = 8.0  # seconds to avoid a peer that just rejected us
MAX_INFLIGHT = 3  # concurrent outstanding requests
SWITCH_CONFIDENCE = 0.55  # min model probability required to switch strategy
SWITCH_PATIENCE = 2  # consecutive agreeing predictions required (hysteresis)
TELEMETRY_DIR = 'telemetry'  # per-peer JSON snapshots read by the dashboard
TELEMETRY_INTERVAL = 1.0


class Peer:
    """
    Main peer orchestrator.

    Manages piece storage, download/upload logic, strategy selection,
    and the adaptive ML loop.
    """

    def __init__(self, port, role, known_peers):
        self.port = port
        self.role = role.lower()
        self.known_peers = [p for p in known_peers if p != port]

        # Piece storage: index -> bytes
        self.pieces = {}
        self.piece_hashes = []
        self.total_pieces = 0

       
        self.peer_pieces = {}  # port -> set of piece indices

        
        self.swarm_metrics = SwarmMetrics()

        
        if self.role == 'normal':
            self.current_strategy = Strategy.TIT_FOR_TAT
        elif self.role == 'adaptive':
            self.current_strategy = Strategy.TIT_FOR_TAT  # Start with TFT
        else:
            self.current_strategy = Strategy.TIT_FOR_TAT

        # Unchoked peers (updated by strategy)
        self.unchoked_peers = set()
        self._unchoke_lock = threading.Lock()

        # ML model (loaded lazily for adaptive peers)
        self.ml_model = None

        # Network server
        self.server = PeerServer(
            port=port,
            role=self.role,
            on_message=self._handle_message,
            on_connect=self._on_connect,
            on_disconnect=self._on_disconnect
        )

        # Pending requests tracking
        self._pending_requests = {}  # piece_index -> (timestamp, peer_port)
        self._request_lock = threading.Lock()
        self._reject_until = {}  # peer_port -> time before which we avoid it
        self._pieces_lock = threading.Lock()

        # Telemetry / explainability state
        self._events = deque(maxlen=60)
        self._strategy_history = []  # [(unix_ts, strategy_name)]
        self._last_features = []
        self._last_proba = [0.0, 0.0, 0.0]
        self._pending_switch = None  # (candidate, streak)
        self._started = time.time()
        self._running = True
        self._completed_at = None

        # Setup logging with port prefix
        self.logger = logging.getLogger(f'peer.{port}')

    # ──────────────────────────────────────────
    # Initialization
    # ──────────────────────────────────────────

    def _event(self, kind, text):
        """Record a human-readable event for the dashboard feed."""
        self._events.append({'t': round(time.time(), 2), 'kind': kind, 'text': text})

    def _write_telemetry(self):
        """Atomically write this peer's state for the dashboard."""
        with self._pieces_lock:
            have = sorted(self.pieces.keys())
        with self._unchoke_lock:
            unchoked = sorted(self.unchoked_peers)
        state = {
            'port': self.port,
            'role': self.role,
            'updated': round(time.time(), 2),
            'started': round(self._started, 2),
            'completed_at': self._completed_at,
            'total_pieces': self.total_pieces,
            'pieces': have,
            'connected': sorted(self.server.get_connected_ports()),
            'unchoked': unchoked,
            'strategy': get_strategy_name(self.current_strategy),
            'strategy_history': self._strategy_history[-30:],
            'features': [round(f, 4) for f in self._last_features],
            'proba': [round(p, 3) for p in self._last_proba],
            'peers': self.swarm_metrics.snapshot(),
            'events': list(self._events)[-30:],
        }
        os.makedirs(TELEMETRY_DIR, exist_ok=True)
        path = os.path.join(TELEMETRY_DIR, f'peer_{self.port}.json')
        tmp = f'{path}.tmp'
        with open(tmp, 'w') as f:
            json.dump(state, f)
        os.replace(tmp, path)

    def _telemetry_loop(self):
        while self._running:
            try:
                self._write_telemetry()
            except Exception as e:
                self.logger.debug(f"Telemetry write failed: {e}")
            time.sleep(TELEMETRY_INTERVAL)

    def _load_file(self):
        """Load and split the test file into pieces."""
        filepath = TEST_FILE
        if not os.path.exists(filepath):
            self.logger.info("Test file not found, generating...")
            generate_test_file(filepath)

        all_pieces = split_file(filepath)
        self.piece_hashes = create_piece_hashes(all_pieces)
        self.total_pieces = len(all_pieces)

        if self.role == 'seed':
            # Seed has ALL pieces
            with self._pieces_lock:
                for i, piece_data in enumerate(all_pieces):
                    self.pieces[i] = piece_data
            self.logger.info(f"Loaded all {self.total_pieces} pieces (SEED)")
        else:
            # Other peers start with NO pieces
            self.logger.info(f"File has {self.total_pieces} pieces, starting with 0")

    def _load_ml_model(self):
        """Load or train the ML model for adaptive peers."""
        if self.role != 'adaptive':
            return

        self.ml_model = load_model()
        if self.ml_model is None:
            self.logger.info("No pre-trained model found, training new model...")
            self.ml_model, accuracy = train_model()
            if self.ml_model:
                self.logger.info(f"Model trained (accuracy: {accuracy:.2%})")
            else:
                self.logger.warning("ML model unavailable, using rule-based fallback")

    # ──────────────────────────────────────────
    # Network Callbacks
    # ──────────────────────────────────────────

    def _on_connect(self, peer_port):
        """Called when a peer connects."""
        self.logger.info(f"Peer {peer_port} connected")
        self._event('net', f"peer {peer_port} connected")
        self.swarm_metrics.record_connect(peer_port)
        self.peer_pieces.setdefault(peer_port, set())

        # Announce our pieces to the new peer
        with self._pieces_lock:
            have = list(self.pieces.keys())
        if have:
            self.server.send_to_peer(peer_port, create_have(have))

    def _on_disconnect(self, peer_port):
        """Called when a peer disconnects."""
        self.logger.info(f"Peer {peer_port} disconnected")
        self._event('net', f"peer {peer_port} disconnected")
        self.peer_pieces.pop(peer_port, None)
        self.swarm_metrics.record_disconnect(peer_port)

    def _handle_message(self, peer_port, msg):
        """
        Route incoming messages to the appropriate handler.

        This is the central message dispatcher called by the network layer.
        """
        if not isinstance(msg, dict):
            return
        msg_type = msg.get('type')

        if msg_type == MessageType.REQUEST:
            self._handle_request(peer_port, msg)
        elif msg_type == MessageType.PIECE:
            self._handle_piece(peer_port, msg)
        elif msg_type == MessageType.HAVE:
            self._handle_have(peer_port, msg)
        elif msg_type == MessageType.REJECT:
            self._handle_reject(peer_port, msg)
        else:
            self.logger.debug(f"Unknown message type from {peer_port}: {msg_type}")

    # ──────────────────────────────────────────
    # Message Handlers
    # ──────────────────────────────────────────

    def _handle_request(self, peer_port, msg):
        """
        Handle a REQUEST message from a peer.

        Checks strategy to decide whether to serve the piece:
          - SEED: always serves
          - FREERIDER: serves ~5% of the time
          - NORMAL/ADAPTIVE: checks unchoke list via strategy
        """
        piece_index = msg.get('piece_index')
        if not isinstance(piece_index, int):
            self.logger.debug(f"Malformed REQUEST from {peer_port}: {msg}")
            return

        # Check if we even have this piece
        if piece_index not in self.pieces:
            self.server.send_to_peer(peer_port, create_reject(piece_index))
            return

        # Record the request in metrics
        pm = self.swarm_metrics.get_or_create(peer_port)
        pm.record_request()

        # Decide whether to serve
        should_serve = False

        if self.role == 'seed':
            # Seeds ALWAYS serve
            should_serve = True
        elif self.role == 'freerider':
            # Freeriders rarely serve (5% chance)
            should_serve = random.random() < FREERIDER_UPLOAD_CHANCE
            if not should_serve:
                self.logger.info(f"FREERIDER: Rejecting request for piece {piece_index} from {peer_port}")
        else:
            # Normal/Adaptive: check unchoke list
            with self._unchoke_lock:
                should_serve = peer_port in self.unchoked_peers

            if not should_serve:
                self.logger.debug(f"Peer {peer_port} is choked, rejecting piece {piece_index}")

        if should_serve:
            # Send the piece
            piece_data = self.pieces[piece_index]
            if self.server.send_to_peer(
                peer_port, create_piece(piece_index, piece_data)
            ):
                pm.record_upload(len(piece_data))  # only count delivered data
                self._event('up', f"piece {piece_index} → {peer_port}")
                self.logger.info(f"Uploaded piece {piece_index} to {peer_port} ({len(piece_data)} bytes)")
        else:
            self.server.send_to_peer(peer_port, create_reject(piece_index))

    def _handle_piece(self, peer_port, msg):
        """
        Handle a received PIECE message.

        Verifies SHA1 hash and stores the piece if valid.
        """
        piece_index = msg.get('piece_index')
        try:
            piece_data = extract_piece_data(msg)
        except Exception:
            piece_data = None
        pm = self.swarm_metrics.get_or_create(peer_port)
        if not isinstance(piece_index, int) or piece_data is None:
            pm.record_failure()
            self.logger.warning(f"Malformed PIECE from {peer_port}")
            return

        # Remove from pending
        with self._request_lock:
            self._pending_requests.pop(piece_index, None)

        # Verify piece integrity
        if 0 <= piece_index < len(self.piece_hashes):
            expected_hash = self.piece_hashes[piece_index]
            if verify_piece(piece_data, expected_hash):
                with self._pieces_lock:
                    is_new = piece_index not in self.pieces
                    self.pieces[piece_index] = piece_data
                    n_have = len(self.pieces)
                if not is_new:
                    return  # duplicate delivery (e.g. endgame); ignore
                pm.record_download(len(piece_data))
                self._event('down', f"piece {piece_index} ← {peer_port}")
                self.logger.info(
                    f"Downloaded piece {piece_index} from {peer_port} "
                    f"({n_have}/{self.total_pieces} complete)"
                )

                # Announce the new piece to all peers
                self.server.broadcast(create_have([piece_index]))

                # Check if download is complete
                if n_have == self.total_pieces:
                    self._completed_at = round(time.time(), 2)
                    self._event('done', f"download complete in {time.time() - self._started:.1f}s")
                    self.logger.info("*** FILE DOWNLOAD COMPLETE! ***")
            else:
                pm.record_failure()
                self._event('fail', f"piece {piece_index} from {peer_port} failed hash")
                self.logger.warning(
                    f"Piece {piece_index} from {peer_port} FAILED hash verification!"
                )
        else:
            self.logger.warning(f"Received unknown piece index {piece_index} from {peer_port}")

    def _handle_have(self, peer_port, msg):
        """Handle a HAVE message: update our knowledge of what the peer has."""
        piece_indices = [i for i in msg.get('piece_indices', []) if isinstance(i, int)]
        self.peer_pieces.setdefault(peer_port, set())
        self.peer_pieces[peer_port].update(piece_indices)
        self.logger.debug(f"Peer {peer_port} has pieces: {piece_indices}")

    def _handle_reject(self, peer_port, msg):
        """Handle a REJECT message: peer refused our request."""
        piece_index = msg.get('piece_index')
        with self._request_lock:
            self._pending_requests.pop(piece_index, None)
        # Don't hammer a peer that just said no
        self._reject_until[peer_port] = time.time() + REJECT_COOLDOWN
        self.logger.debug(f"Peer {peer_port} rejected request for piece {piece_index}")

    # ──────────────────────────────────────────
    # Download Loop
    # ──────────────────────────────────────────

    def _download_loop(self):
        """
        Continuously request missing pieces from peers.

        For non-seed peers: identifies missing pieces, finds peers
        that have them, and sends REQUEST messages.
        """
        if self.role == 'seed':
            return  # Seeds don't download

        while self._running:
            try:
                missing = self._get_missing_pieces()
                if not missing:
                    time.sleep(DOWNLOAD_LOOP_INTERVAL)
                    continue

                with self._request_lock:
                    now = time.time()
                    # Time out stale requests and charge the peer for it
                    for idx, (ts, port) in list(self._pending_requests.items()):
                        if now - ts > REQUEST_TIMEOUT:
                            del self._pending_requests[idx]
                            self.swarm_metrics.get_or_create(port).record_timeout()
                            self._reject_until[port] = now + REJECT_COOLDOWN
                            self._event('fail', f"request for piece {idx} to {port} timed out")
                    slots = MAX_INFLIGHT - len(self._pending_requests)
                    requestable = [
                        idx for idx in missing
                        if idx not in self._pending_requests
                    ]

                if not requestable or slots <= 0:
                    time.sleep(DOWNLOAD_LOOP_INTERVAL)
                    continue

                # Rarest-first: prefer pieces held by the fewest peers
                # (random tie-break) so scarce pieces spread through the swarm.
                random.shuffle(requestable)
                requestable.sort(key=self._availability)

                for piece_index in requestable:
                    if slots <= 0:
                        break
                    peer_port = self._find_peer_with_piece(piece_index)
                    if peer_port is None:
                        continue
                    with self._request_lock:
                        self._pending_requests[piece_index] = (time.time(), peer_port)
                    if self.server.send_to_peer(peer_port, create_request(piece_index)):
                        self.swarm_metrics.get_or_create(peer_port).record_request_sent()
                        slots -= 1
                        self.logger.debug(
                            f"Requested piece {piece_index} from {peer_port}"
                        )
                    else:
                        with self._request_lock:
                            self._pending_requests.pop(piece_index, None)

                time.sleep(DOWNLOAD_LOOP_INTERVAL)

            except Exception as e:
                self.logger.error(f"Download loop error: {e}")
                time.sleep(DOWNLOAD_LOOP_INTERVAL)

    def _get_missing_pieces(self):
        """Get list of piece indices we don't have yet."""
        with self._pieces_lock:
            return [i for i in range(self.total_pieces) if i not in self.pieces]

    def _availability(self, piece_index):
        """Number of known peers advertising this piece."""
        return sum(1 for pcs in list(self.peer_pieces.values()) if piece_index in pcs)

    def _find_peer_with_piece(self, piece_index):
        """Find a connected peer that has a specific piece."""
        candidates = []
        connected = set(self.server.get_connected_ports())

        now = time.time()
        for port, pieces in list(self.peer_pieces.items()):
            if piece_index in pieces and port in connected:
                candidates.append(port)

        if not candidates:
            return None
        # Prefer peers that aren't cooling off; fall back to any holder
        ready = [p for p in candidates if self._reject_until.get(p, 0) <= now]
        return random.choice(ready or candidates)

    # ──────────────────────────────────────────
    # Strategy & Unchoke Management
    # ──────────────────────────────────────────

    def _unchoke_loop(self):
        """Periodically re-evaluate which peers to unchoke using the active strategy."""
        if self.role == 'seed':
            return  # Seeds serve everyone

        while self._running:
            try:
                strategy = get_strategy(self.current_strategy)
                selected = strategy.select_peers(self.swarm_metrics)

                with self._unchoke_lock:
                    old = self.unchoked_peers.copy()
                    self.unchoked_peers = set(selected)

                # Log changes
                newly_unchoked = self.unchoked_peers - old
                newly_choked = old - self.unchoked_peers
                if newly_unchoked:
                    self.logger.debug(f"Unchoked peers: {newly_unchoked}")
                if newly_choked:
                    self.logger.debug(f"Choked peers: {newly_choked}")

            except Exception as e:
                self.logger.error(f"Unchoke loop error: {e}")

            time.sleep(10)  # Re-evaluate every 10 seconds

    # ──────────────────────────────────────────
    # Adaptive ML Strategy Loop
    # ──────────────────────────────────────────

    def _adaptive_loop(self):
        """
        [ADAPTIVE PEER ONLY] Every 30 seconds:
          1. Compute swarm-level feature vector
          2. Run ML model prediction
          3. Switch active strategy if different
        """
        if self.role != 'adaptive':
            return

        while self._running:
            time.sleep(ADAPTIVE_INTERVAL)
            try:
                # Step 1: Compute feature vector
                features = self.swarm_metrics.feature_vector()
                self.logger.info(f"Swarm features: {[f'{f:.3f}' for f in features]}")

                # Step 2: ML prediction (with class probabilities)
                self._last_features = features
                candidate, proba = predict_with_confidence(self.ml_model, features)
                self._last_proba = proba
                confidence = proba[candidate]

                # Step 3: Switch only on confident, repeated agreement
                # (hysteresis prevents flapping on noisy swarm metrics)
                old_strategy = self.current_strategy
                if candidate == old_strategy:
                    self._pending_switch = None
                    self.logger.info(
                        f"Strategy unchanged: {get_strategy_name(candidate)} "
                        f"(confidence {confidence:.0%})"
                    )
                elif confidence < SWITCH_CONFIDENCE:
                    self._pending_switch = None
                    self.logger.info(
                        f"Ignoring low-confidence suggestion {get_strategy_name(candidate)} "
                        f"({confidence:.0%} < {SWITCH_CONFIDENCE:.0%})"
                    )
                else:
                    prev, streak = self._pending_switch or (candidate, 0)
                    streak = streak + 1 if prev == candidate else 1
                    self._pending_switch = (candidate, streak)
                    if streak >= SWITCH_PATIENCE:
                        self.current_strategy = Strategy(candidate)
                        self._pending_switch = None
                        self._strategy_history.append(
                            (round(time.time(), 2), get_strategy_name(candidate)))
                        self._event('ml', f"switched {get_strategy_name(old_strategy)} → "
                                          f"{get_strategy_name(candidate)} ({confidence:.0%})")
                        self.logger.info(
                            f"*** Switched strategy: {get_strategy_name(old_strategy)} "
                            f"→ {get_strategy_name(candidate)} ***"
                        )
                    else:
                        self.logger.info(
                            f"Candidate {get_strategy_name(candidate)} "
                            f"({streak}/{SWITCH_PATIENCE} confirmations)"
                        )

                # Log contributor analysis
                for port, pm in list(self.swarm_metrics.peer_metrics.items()):
                    if pm.is_freerider():
                        self._event('ml', f"peer {port} flagged as freerider "
                                          f"(reciprocity {pm.reciprocity():.2f})")
                        self.logger.info(
                            f"Peer {port} classified as freerider "
                            f"(reciprocity={pm.reciprocity():.2f})")

            except Exception as e:
                self.logger.error(f"Adaptive loop error: {e}")

    # ──────────────────────────────────────────
    # HAVE Broadcast Loop
    # ──────────────────────────────────────────

    def _have_broadcast_loop(self):
        """Periodically broadcast our piece list to all connected peers."""
        while self._running:
            time.sleep(HAVE_BROADCAST_INTERVAL)
            with self._pieces_lock:
                have = list(self.pieces.keys())
            if have:
                self.server.broadcast(create_have(have))

    # ──────────────────────────────────────────
    # Main Run
    # ──────────────────────────────────────────

    def run(self):
        """Start the peer: load file, start server, connect to peers, run loops."""
        self.logger.info(f"Starting peer on port {self.port} with role={self.role.upper()}")

        # Graceful shutdown on SIGTERM too (run_simulation uses terminate())
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        self._strategy_history.append(
            (round(time.time(), 2), get_strategy_name(self.current_strategy)))

        # Step 1: Load/split test file
        self._load_file()

        # Step 2: Load ML model (adaptive only)
        self._load_ml_model()

        # Step 3: Start network server
        self.server.start()

        # Step 4: Wait briefly for other peers to start, then connect
        time.sleep(2)
        self.logger.info(f"Connecting to known peers: {self.known_peers}")
        for peer_port in self.known_peers:
            threading.Thread(
                target=self.server.connect_to_peer,
                args=(peer_port,),
                daemon=True
            ).start()

        # Step 5: Start background loops
        threads = []

        # Download loop (non-seed peers)
        if self.role != 'seed':
            t = threading.Thread(target=self._download_loop, daemon=True)
            t.start()
            threads.append(t)

        # Unchoke loop (non-seed peers)
        if self.role != 'seed':
            t = threading.Thread(target=self._unchoke_loop, daemon=True)
            t.start()
            threads.append(t)

        # Adaptive strategy loop
        if self.role == 'adaptive':
            t = threading.Thread(target=self._adaptive_loop, daemon=True)
            t.start()
            threads.append(t)

        # Telemetry for the dashboard
        t = threading.Thread(target=self._telemetry_loop, daemon=True)
        t.start()
        threads.append(t)

        # HAVE broadcast loop
        t = threading.Thread(target=self._have_broadcast_loop, daemon=True)
        t.start()
        threads.append(t)

        # Step 6: Run until interrupted
        self.logger.info(f"Peer {self.port} ({self.role.upper()}) is running. Press Ctrl+C to stop.")
        try:
            while True:
                # Periodic status log
                time.sleep(15)
                connected = self.server.get_connected_ports()
                self.logger.info(
                    f"Status: {len(self.pieces)}/{self.total_pieces} pieces, "
                    f"{len(connected)} connected peers {connected}, "
                    f"strategy={get_strategy_name(self.current_strategy)}"
                )
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
            self._running = False
            try:
                self._write_telemetry()
            except Exception:
                pass
            self.server.stop()


def setup_logging(port):
    """Configure logging with peer port prefix."""
    formatter = logging.Formatter(
        f'[{port}] %(asctime)s %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)

    # Quiet down noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(
        description='BitTorrent P2P Simulation - Peer Node',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python peer.py --port 5001 --role seed
  python peer.py --port 5002 --role normal
  python peer.py --port 5003 --role freerider
  python peer.py --port 5004 --role adaptive

Roles:
  seed       Has full file, only uploads
  normal     Downloads and uploads using TIT_FOR_TAT
  freerider  Downloads but rarely uploads (~5%% chance)
  adaptive   Uses ML-based strategy switching every 30s
        """
    )

    parser.add_argument(
        '--port', type=int, required=True,
        help='Port to listen on (e.g., 5001)'
    )
    parser.add_argument(
        '--role', type=str, required=True,
        choices=['seed', 'normal', 'freerider', 'adaptive'],
        help='Peer role: seed, normal, freerider, or adaptive'
    )
    parser.add_argument(
        '--peers', type=str, default=None,
        help='Comma-separated list of known peer ports (default: 5001,5002,5003,5004)'
    )

    args = parser.parse_args()

    # Parse known peers
    if args.peers:
        try:
            known_peers = [int(p.strip()) for p in args.peers.split(',') if p.strip()]
        except ValueError:
            parser.error('--peers must be a comma-separated list of integer ports')
    else:
        known_peers = DEFAULT_PEERS

    # Setup logging
    setup_logging(args.port)

    # Create and run peer
    peer = Peer(port=args.port, role=args.role, known_peers=known_peers)
    peer.run()


if __name__ == '__main__':
    main()
