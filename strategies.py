"""
strategies.py – Three peer selection strategies for unchoking decisions.

Strategies:
  1. TIT_FOR_TAT     – Reciprocal: top N by upload rate + 1 optimistic unchoke
  2. STRICT_MODE     – Anti-freerider: only peers with ratio > threshold
  3. EXPLORATION_MODE – Random selection with multiple optimistic unchokes

Each strategy implements select_peers() → list of peer ports to unchoke.

METRIC CONVENTION (important):
  PeerMetrics tracks from OUR perspective:
    - bytes_uploaded   = bytes WE sent TO this peer
    - bytes_downloaded = bytes WE received FROM this peer
  So from the strategy's point of view:
    - A peer's "contribution to us" = pm.bytes_downloaded (they gave us data)
    - A peer's "consumption from us" = pm.bytes_uploaded (they took our data)
    - Reciprocity ratio = pm.bytes_downloaded / pm.bytes_uploaded
      (how much they give back per unit we give them)
"""

import random
import logging
from enum import IntEnum
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from metrics import SwarmMetrics

logger = logging.getLogger(__name__)


class Strategy(IntEnum):
    """Strategy enum matching ML model output classes."""
    TIT_FOR_TAT = 0
    STRICT_MODE = 1
    EXPLORATION_MODE = 2


# ──────────────────────────────────────────────
# Strategy 1: TIT_FOR_TAT
# ──────────────────────────────────────────────

class TitForTat:
    """
    Reciprocal strategy: reward peers who contribute the most.

    Algorithm:
      1. Sort peers by how much they upload TO US
         (= bytes_downloaded in our PeerMetrics, i.e. what we received)
      2. Select top N-1 peers
      3. Add exactly 1 optimistic unchoke (random peer not in top N-1)

    This promotes fairness – peers who share more get served more.
    """

    def __init__(self, max_unchoke: int = 4):
        self.max_unchoke = max_unchoke

    def select_peers(self, swarm_metrics: 'SwarmMetrics') -> List[int]:
        """
        Select peers to unchoke based on their contribution to us.

        Args:
            swarm_metrics: SwarmMetrics instance with per-peer data.

        Returns:
            List of peer ports to unchoke.
        """
        peers = list(swarm_metrics.peer_metrics.values())
        if not peers:
            return []

        # Sort by how much they give US (download_rate = rate we receive from them)
        sorted_peers = sorted(
            peers,
            key=lambda p: p.download_rate(),
            reverse=True
        )

        # Top N-1 peers (leave room for 1 optimistic unchoke)
        n = max(1, self.max_unchoke - 1)
        top_peers = sorted_peers[:n]
        top_ports = {p.peer_port for p in top_peers}

        # Optimistic unchoke: 1 random peer NOT in top N-1
        remaining = [p for p in peers if p.peer_port not in top_ports]
        unchoked_ports = [p.peer_port for p in top_peers]

        if remaining:
            optimistic = random.choice(remaining)
            unchoked_ports.append(optimistic.peer_port)
            logger.info(f"  TIT_FOR_TAT optimistic unchoke: peer {optimistic.peer_port}")

        return unchoked_ports


# ──────────────────────────────────────────────
# Strategy 2: STRICT_MODE (Anti-freerider)
# ──────────────────────────────────────────────

class StrictMode:
    """
    Anti-freerider strategy: only serve peers who contribute.

    Algorithm:
      1. Compute reciprocity ratio for each peer:
         reciprocity = bytes_downloaded / bytes_uploaded
         (how much they give us back relative to what we give them)
      2. Only unchoke peers with reciprocity > threshold
      3. NO optimistic unchoke – strict enforcement

    This punishes freeriders by completely denying them service.
    """

    def __init__(self, max_unchoke: int = 4, ratio_threshold: float = 0.8):
        self.max_unchoke = max_unchoke
        self.ratio_threshold = ratio_threshold

    def _reciprocity_ratio(self, pm) -> float:
        """Reciprocity ratio (see PeerMetrics.reciprocity)."""
        return pm.reciprocity()

    def select_peers(self, swarm_metrics: 'SwarmMetrics') -> List[int]:
        """
        Select peers to unchoke based on their reciprocity ratio.

        Args:
            swarm_metrics: SwarmMetrics instance with per-peer data.

        Returns:
            List of peer ports to unchoke (only reciprocating peers).
        """
        peers = list(swarm_metrics.peer_metrics.values())
        if not peers:
            return []

        # Only unchoke peers with sufficient reciprocity
        qualified = []
        for p in peers:
            ratio = self._reciprocity_ratio(p)
            if ratio > self.ratio_threshold:
                qualified.append((p, ratio))

        # Sort by ratio (best reciprocators first)
        qualified.sort(key=lambda x: x[1], reverse=True)

        # Take top N – NO optimistic unchoke
        unchoked = qualified[:self.max_unchoke]
        unchoked_ports = [p.peer_port for p, _ in unchoked]
        unchoked_set = set(unchoked_ports)

        # Log rejected peers
        for p in peers:
            if p.peer_port not in unchoked_set:
                ratio = self._reciprocity_ratio(p)
                if p.bytes_downloaded > 0 or p.bytes_uploaded > 0:
                    logger.info(
                        f"  STRICT_MODE: Peer {p.peer_port} rejected "
                        f"(reciprocity={ratio:.2f} < {self.ratio_threshold})"
                    )

        return unchoked_ports


# ──────────────────────────────────────────────
# Strategy 3: EXPLORATION_MODE
# ──────────────────────────────────────────────

class ExplorationMode:
    """
    Exploration strategy: discover new potentially good peers.

    Algorithm:
      1. Ignore historical contribution scores entirely
      2. Randomly select N peers for unchoking
      3. All selections are optimistic unchokes (2-3+ peers)
      4. Rotate selection frequently

    Useful when the swarm has many new peers or low information.
    """

    def __init__(self, max_unchoke: int = 4):
        self.max_unchoke = max_unchoke

    def select_peers(self, swarm_metrics: 'SwarmMetrics') -> List[int]:
        """
        Randomly select peers to unchoke (exploration).

        Args:
            swarm_metrics: SwarmMetrics instance with per-peer data.

        Returns:
            List of randomly selected peer ports.
        """
        peers = list(swarm_metrics.peer_metrics.values())
        if not peers:
            return []

        # Random sample of up to max_unchoke peers
        k = min(self.max_unchoke, len(peers))
        selected = random.sample(peers, k)
        unchoked_ports = [p.peer_port for p in selected]

        logger.info(
            f"  EXPLORATION_MODE: Randomly selected {len(unchoked_ports)} peers: "
            f"{unchoked_ports}"
        )
        return unchoked_ports


# ──────────────────────────────────────────────
# Strategy Factory
# ──────────────────────────────────────────────

# Singleton instances
_strategies = {
    Strategy.TIT_FOR_TAT: TitForTat(),
    Strategy.STRICT_MODE: StrictMode(),
    Strategy.EXPLORATION_MODE: ExplorationMode(),
}


def get_strategy(strategy_enum):
    """
    Get a strategy instance by enum value.

    Args:
        strategy_enum: Strategy enum value (0, 1, or 2).

    Returns:
        Strategy instance with select_peers() method.
    """
    return _strategies[Strategy(strategy_enum)]


def get_strategy_name(strategy_enum):
    """Get human-readable strategy name."""
    return Strategy(strategy_enum).name


if __name__ == '__main__':
    from metrics import SwarmMetrics

    # Quick self-test
    swarm = SwarmMetrics()

    # Simulate peer 5001: we uploaded 1024 to them, they gave us 2048
    pm1 = swarm.get_or_create(5001)
    pm1.record_upload(1024)     # We sent 1024 to them
    pm1.record_download(2048)   # They sent 2048 to us  → reciprocity = 2.0

    # Simulate peer 5002: we uploaded 4096 to them, they gave us 0
    pm2 = swarm.get_or_create(5002)
    pm2.record_upload(4096)     # We sent 4096 to them
    # No download from them    → reciprocity = 0.0 (freerider)

    # Simulate peer 5003: balanced exchange
    pm3 = swarm.get_or_create(5003)
    pm3.record_upload(1024)
    pm3.record_download(1024)   # reciprocity = 1.0

    print("=== Peer Metrics ===")
    for port in [5001, 5002, 5003]:
        pm = swarm.peer_metrics[port]
        print(f"  Peer {port}: uploaded={pm.bytes_uploaded}, "
              f"downloaded={pm.bytes_downloaded}, "
              f"ratio={pm.reciprocity():.2f}")

    print("\n=== TIT_FOR_TAT ===")
    tft = get_strategy(Strategy.TIT_FOR_TAT)
    print(f"Unchoked: {tft.select_peers(swarm)}")

    print("\n=== STRICT_MODE ===")
    strict = get_strategy(Strategy.STRICT_MODE)
    result = strict.select_peers(swarm)
    print(f"Unchoked: {result}")
    print(f"  (Peer 5002 should be rejected as freerider)")

    print("\n=== EXPLORATION_MODE ===")
    explore = get_strategy(Strategy.EXPLORATION_MODE)
    print(f"Unchoked: {explore.select_peers(swarm)}")
