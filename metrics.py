"""
metrics.py – Per-peer and swarm-level metric tracking.

PeerMetrics: Tracks upload/download rates, ratios, and failure stats per peer.
SwarmMetrics: Aggregates peer-level metrics into a feature vector for ML model.
"""

import time
import threading
from collections import defaultdict


class PeerMetrics:
    """
    Track metrics for a single remote peer.

    Tracks:
      - Bytes uploaded to this peer
      - Bytes downloaded from this peer
      - Upload/download rates (computed over sliding window)
      - Request counts, timeouts, and piece failures
    """

    def __init__(self, peer_port):
        self.peer_port = peer_port
        self.bytes_uploaded = 0
        self.bytes_downloaded = 0
        self.request_count = 0       # requests RECEIVED from this peer
        self.requests_sent = 0       # requests WE sent to this peer
        self.timeout_count = 0
        self.failure_count = 0

        # Timestamps for rate computation
        self._upload_history = []    # list of (timestamp, bytes)
        self._download_history = []  # list of (timestamp, bytes)
        self._lock = threading.Lock()

        # Rate window in seconds
        self.rate_window = 60.0

    def record_upload(self, num_bytes):
        """Record bytes uploaded TO this peer."""
        with self._lock:
            now = time.time()
            self.bytes_uploaded += num_bytes
            self._upload_history.append((now, num_bytes))
            self._prune_history()

    def record_download(self, num_bytes):
        """Record bytes downloaded FROM this peer."""
        with self._lock:
            now = time.time()
            self.bytes_downloaded += num_bytes
            self._download_history.append((now, num_bytes))
            self._prune_history()

    def record_request(self):
        """Record a piece request received FROM this peer."""
        with self._lock:
            self.request_count += 1

    def record_request_sent(self):
        """Record a piece request WE sent to this peer."""
        with self._lock:
            self.requests_sent += 1

    def record_timeout(self):
        """Record a request timeout from this peer."""
        with self._lock:
            self.timeout_count += 1

    def record_failure(self):
        """Record a piece verification failure from this peer."""
        with self._lock:
            self.failure_count += 1

    def upload_rate(self):
        """Compute upload rate (bytes/sec) over the rate window."""
        return self._compute_rate(self._upload_history)

    def download_rate(self):
        """Compute download rate (bytes/sec) over the rate window."""
        return self._compute_rate(self._download_history)

    def upload_download_ratio(self):
        """
        Compute upload/download ratio.
        Returns 0.0 if no downloads, 1.0 if balanced.
        A freerider will have a very low ratio (downloads >> uploads).
        """
        if self.bytes_downloaded == 0:
            return 1.0 if self.bytes_uploaded > 0 else 0.0
        return self.bytes_uploaded / self.bytes_downloaded

    def reciprocity(self):
        """
        How much this peer gives us per byte we give them
        (bytes_downloaded / bytes_uploaded, our perspective).

        1.0 when we have not given them anything yet and they have given us
        something; 0.0 when there is no history at all. Low values with a
        non-trivial upload to them indicate a freerider.
        """
        if self.bytes_uploaded == 0:
            return 1.0 if self.bytes_downloaded > 0 else 0.0
        return self.bytes_downloaded / self.bytes_uploaded

    def is_freerider(self, threshold=0.3, min_uploaded=1024):
        """True if we gave this peer real data and got (almost) nothing back."""
        return self.bytes_uploaded >= min_uploaded and self.reciprocity() < threshold

    def snapshot(self):
        """JSON-serialisable view for telemetry."""
        return {
            'port': self.peer_port,
            'uploaded': self.bytes_uploaded,
            'downloaded': self.bytes_downloaded,
            'reciprocity': round(self.reciprocity(), 3),
            'down_rate': round(self.download_rate(), 1),
            'timeouts': self.timeout_count,
            'failures': self.failure_count,
            'freerider': self.is_freerider(),
        }

    def _compute_rate(self, history):
        """Compute transfer rate from recent history."""
        with self._lock:
            now = time.time()
            cutoff = now - self.rate_window
            recent = [(t, b) for t, b in history if t > cutoff]
            if not recent:
                return 0.0
            total_bytes = sum(b for _, b in recent)
            elapsed = now - recent[0][0]
            if elapsed < 0.1:
                return float(total_bytes)
            return total_bytes / elapsed

    def _prune_history(self):
        """Remove entries older than rate_window."""
        cutoff = time.time() - self.rate_window * 2
        self._upload_history = [
            (t, b) for t, b in self._upload_history if t > cutoff
        ]
        self._download_history = [
            (t, b) for t, b in self._download_history if t > cutoff
        ]


class SwarmMetrics:
    """
    Aggregate peer-level metrics into swarm-level features.

    Used by the adaptive peer to compute the feature vector
    for the ML strategy selector.
    """

    def __init__(self):
        self.peer_metrics = {}  # port -> PeerMetrics
        self._lock = threading.Lock()

        # Churn tracking
        self._connect_events = []   # timestamps of connect events
        self._disconnect_events = []  # timestamps of disconnect events
        self.churn_window = 120.0    # 2-minute window for churn rate

    def get_or_create(self, peer_port):
        """Get or create PeerMetrics for a specific peer."""
        with self._lock:
            if peer_port not in self.peer_metrics:
                self.peer_metrics[peer_port] = PeerMetrics(peer_port)
            return self.peer_metrics[peer_port]

    def record_connect(self, peer_port):
        """Record a peer connection event."""
        self.get_or_create(peer_port)
        with self._lock:
            self._connect_events.append(time.time())

    def record_disconnect(self, peer_port):
        """Record a peer disconnection event."""
        with self._lock:
            self._disconnect_events.append(time.time())

    def avg_upload_ratio(self):
        """Average reciprocity (what peers give us per byte we give them)."""
        with self._lock:
            ratios = [
                pm.reciprocity()
                for pm in self.peer_metrics.values()
            ]
        if not ratios:
            return 0.0
        return sum(ratios) / len(ratios)

    def low_contributor_pct(self, threshold=0.3):
        """
        Percentage of peers with upload/download ratio below threshold.
        These are suspected freeriders.
        """
        with self._lock:
            peers = list(self.peer_metrics.values())
        if not peers:
            return 0.0
        low = sum(1 for pm in peers if pm.is_freerider(threshold))
        return low / len(peers)

    def churn_rate(self):
        """
        Peer churn rate: (connects + disconnects) per minute
        over the churn window.
        """
        now = time.time()
        cutoff = now - self.churn_window
        with self._lock:
            recent_connects = sum(1 for t in self._connect_events if t > cutoff)
            recent_disconnects = sum(1 for t in self._disconnect_events if t > cutoff)
        window_minutes = self.churn_window / 60.0
        return (recent_connects + recent_disconnects) / window_minutes

    def timeout_rate(self):
        """Average timeout rate across all peers (timeouts / requests)."""
        with self._lock:
            peers = list(self.peer_metrics.values())
        total_timeouts = sum(pm.timeout_count for pm in peers)
        total_requests = sum(pm.requests_sent for pm in peers)
        if total_requests == 0:
            return 0.0
        return total_timeouts / total_requests

    def failure_rate(self):
        """Average piece failure rate across all peers."""
        with self._lock:
            peers = list(self.peer_metrics.values())
        total_failures = sum(pm.failure_count for pm in peers)
        total_requests = sum(pm.requests_sent for pm in peers)
        if total_requests == 0:
            return 0.0
        return total_failures / total_requests

    def snapshot(self):
        """JSON-serialisable per-peer view for telemetry."""
        with self._lock:
            peers = list(self.peer_metrics.values())
        return [pm.snapshot() for pm in sorted(peers, key=lambda p: p.peer_port)]

    def feature_vector(self):
        """
        Compute the 5-feature vector for the ML strategy selector.

        Returns:
            [avg_ratio, low_contributor_pct, churn_rate, timeout_rate, failure_rate]
        """
        return [
            self.avg_upload_ratio(),
            self.low_contributor_pct(),
            self.churn_rate(),
            self.timeout_rate(),
            self.failure_rate()
        ]


if __name__ == '__main__':
    # Quick self-test
    swarm = SwarmMetrics()

    # Simulate some peers
    pm1 = swarm.get_or_create(5001)
    pm1.record_upload(2048)
    pm1.record_download(1024)

    pm2 = swarm.get_or_create(5002)
    pm2.record_download(4096)  # Freerider: downloads but doesn't upload
    pm2.record_request_sent()
    pm2.record_request_sent()
    pm2.record_timeout()

    pm3 = swarm.get_or_create(5003)
    pm3.record_upload(1024)
    pm3.record_download(1024)

    swarm.record_connect(5001)
    swarm.record_connect(5002)
    swarm.record_connect(5003)

    print(f"Peer 5001 ratio: {pm1.upload_download_ratio():.2f}")
    print(f"Peer 5002 ratio: {pm2.upload_download_ratio():.2f}")
    print(f"Peer 5003 ratio: {pm3.upload_download_ratio():.2f}")
    print(f"Avg upload ratio: {swarm.avg_upload_ratio():.2f}")
    print(f"Low contributor %: {swarm.low_contributor_pct():.2f}")
    print(f"Churn rate: {swarm.churn_rate():.2f}")
    print(f"Timeout rate: {swarm.timeout_rate():.2f}")
    print(f"Feature vector: {swarm.feature_vector()}")
