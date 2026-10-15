# BitTorrent P2P Simulation with AI Adaptive Strategy

A minimal but functional BitTorrent-inspired peer-to-peer simulation demonstrating ML-based adaptive strategy selection.

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Generate Test File & Train Model

```bash
cd BitC
python utils.py        # Generates testfile.txt and verifies piece splitting
python ml_model.py     # Trains the RandomForest model → strategy_model.pkl
```

### Run the Simulation

Open 4 separate terminals and run each peer:

```bash
# Terminal 1 – Seed (has full file, only uploads)
python peer.py --port 5001 --role seed

# Terminal 2 – Normal peer (TIT_FOR_TAT strategy)
python peer.py --port 5002 --role normal

# Terminal 3 – Freerider (downloads but rarely uploads)
python peer.py --port 5003 --role freerider

# Terminal 4 – Adaptive peer (ML-based strategy switching)
python peer.py --port 5004 --role adaptive
```

> **Tip:** Start the seed first, wait 2 seconds, then start the others.

### Custom Peer List

```bash
python peer.py --port 5005 --role normal --peers 5001,5002,5003,5004,5005
```

---

## SwarmLab dashboard

```bash
python run_simulation.py --dashboard --duration 0   # runs 4 peers until Ctrl+C + UI at http://127.0.0.1:8765
python dashboard.py                       # UI only, reads ./telemetry written by peers
python serve.py                           # hosting entry point (used by Render; honours $PORT)
```

Open `web/index.html` via the server, or add `?demo` to the URL for a built-in
simulation that needs no backend. The UI shows swarm topology with moving
pieces, per-peer piece maps and reciprocity tables, the adaptive peer's model
class probabilities / features / strategy timeline, and an event feed.

## Robustness & novelty notes

- **Fixed metric bugs**: reciprocity was inverted (generous peers looked like freeriders); timeouts were never recorded; failure/timeout rates were divided by the wrong counter.
- **Rarest-first piece selection**, bounded in-flight requests, request timeouts charged to the peer, and reject cool-downs.
- **Confidence-gated, hysteretic strategy switching**: the adaptive peer only switches after 2 consecutive predictions above 55% probability, with the class probabilities exposed for explainability.
- Message validation (malformed messages no longer kill connections), spurious-disconnect churn fix, SIGTERM-safe shutdown, atomic model/telemetry writes.
- The model pickle is generated locally and not committed (never unpickle untrusted files).
- Tests: `python -m unittest discover -s tests`

## Architecture

```
peer.py          Main entry, CLI, orchestration
├── network.py   TCP socket server, peer connections
├── protocol.py  Length-prefixed JSON message framing
├── strategies.py Three unchoking strategies
├── metrics.py   Per-peer and swarm-level metrics
├── ml_model.py  RandomForest strategy selector
├── utils.py     File splitting, SHA1 hashing
└── dashboard.py + web/  SwarmLab live UI (telemetry JSON → HTML/Canvas)
```

## Peer Roles

| Role | Behavior |
|------|----------|
| `seed` | Has full file, serves all requests unconditionally |
| `normal` | Uses TIT_FOR_TAT strategy for fair reciprocal exchange |
| `freerider` | Downloads pieces but uploads only ~5% of the time |
| `adaptive` | Switches strategy every 30s based on ML analysis of swarm metrics |

## Strategies

| Strategy | Logic |
|----------|-------|
| `TIT_FOR_TAT` | Top N peers by upload rate + 1 optimistic unchoke |
| `STRICT_MODE` | Only peers with upload/download ratio > 0.8, no optimistic unchoke |
| `EXPLORATION_MODE` | Random peer selection, multiple optimistic unchokes |

## ML Feature Vector

The adaptive peer computes these 5 swarm-level features every 30 seconds:

```
[avg_upload_ratio, low_contributor_pct, churn_rate, timeout_rate, failure_rate]
```

The RandomForest model maps this to one of the 3 strategies.

## Example Logs

```
[5001] 15:04:12 INFO: Starting peer on port 5001 with role=SEED
[5001] 15:04:12 INFO: Loaded all 6 pieces (SEED)
[5001] 15:04:14 INFO: Accepted connection from peer 5002 (role=normal)
[5001] 15:04:15 INFO: Uploaded piece 0 to 5002 (1024 bytes)

[5004] 15:04:42 INFO: Swarm features: ['0.450', '0.333', '2.000', '0.000', '0.000']
[5004] 15:04:42 INFO: *** Switched strategy: TIT_FOR_TAT → STRICT_MODE ***
[5004] 15:04:42 INFO: Peer 5003 classified as low contributor (ratio=0.05)
```
