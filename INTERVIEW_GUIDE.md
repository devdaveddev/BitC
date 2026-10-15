# BitC – Interview Guide: BitTorrent P2P Simulation with AI-Adaptive Strategy

---

## 1. One-Liner Pitch

> A Python-based BitTorrent-inspired peer-to-peer simulation where four distinct peer roles exchange file pieces over TCP, and an **adaptive peer** uses a **RandomForest ML model** to dynamically switch unchoking strategies in real time based on swarm health metrics.

---

## 2. Why I Built This

- To demonstrate understanding of **distributed systems** concepts (peer discovery, message framing, choking/unchoking).
- To show how **machine learning** can be embedded inside a systems-level application for real-time decision-making.
- To explore the BitTorrent protocol's **incentive mechanisms** (Tit-for-Tat, freerider punishment) in a controlled environment.

---

## 3. Architecture Overview

```
run_simulation.py          (orchestrator – launches all 4 peers)
  └── peer.py              (main entry, CLI, orchestration)
        ├── network.py      (TCP server, peer connections, thread-per-connection)
        ├── protocol.py     (length-prefixed JSON message framing)
        ├── strategies.py   (3 unchoking strategies: TFT, Strict, Exploration)
        ├── metrics.py      (per-peer + swarm-level metric tracking)
        ├── ml_model.py     (RandomForest strategy selector + rule-based fallback)
        └── utils.py        (file splitting, SHA1 hashing, verification)
```

**Key design choice:** Every module is a standalone file with its own `if __name__ == '__main__'` self-test, making each component independently testable and debuggable.

---

## 4. Module-by-Module Deep Dive

### 4.1 `utils.py` – File Splitting & Hashing

**What it does:**
- Splits any file into fixed-size **1024-byte pieces**.
- Computes **SHA1 hashes** for each piece for integrity verification.
- Provides `verify_piece()` to check a received piece against its expected hash.
- Generates a ~5KB test file (`testfile.txt`) for the simulation.

**Key functions:**
| Function | Purpose |
|---|---|
| `split_file(filepath)` | Reads file → returns list of byte chunks |
| `create_piece_hashes(pieces)` | SHA1 hex digest for every piece |
| `verify_piece(data, expected_hash)` | Integrity check on download |
| `reassemble_file(pieces, output_path)` | Reconstruct original file from pieces |
| `generate_test_file(filepath)` | Create test content for simulation |

**Interview talking point:**
> "I chose SHA1 because real BitTorrent uses it for piece verification. The file is split into 1024-byte chunks – small enough for quick transfer in local simulation, while still demonstrating the piece-based architecture."

---

### 4.2 `protocol.py` – Message Framing

**What it does:**
- Implements a **length-prefixed JSON** wire protocol over raw TCP sockets.
- Format: `[4 bytes big-endian length][UTF-8 JSON payload]`
- Defines 5 message types and factory functions for each.

**Message types:**
| Type | Fields | Purpose |
|---|---|---|
| `HANDSHAKE` | `port`, `role` | Peer identification on connection |
| `REQUEST` | `piece_index` | Ask a peer for a specific piece |
| `PIECE` | `piece_index`, `data` (base64) | Deliver piece bytes |
| `HAVE` | `piece_indices` (list) | Announce which pieces you own |
| `REJECT` | `piece_index` | Decline a request (peer is choked) |

**Why length-prefixed JSON?**
> "TCP is a byte stream, not a message stream. Without framing, you can't tell where one message ends and the next begins. I prepend a 4-byte big-endian integer (the payload length) so `recv_message()` knows exactly how many bytes to read. JSON is used for readability and debugging ease – in production you'd use protobuf or a binary format for efficiency."

**Wire-level detail:**
```
┌──────────────┬─────────────────────────────────────┐
│ 4 bytes (BE) │         JSON payload (UTF-8)         │
│  msg length  │  {"type":"REQUEST","piece_index":3}  │
└──────────────┴─────────────────────────────────────┘
```

**Key implementation detail – `_recv_exact()`:**
```python
def _recv_exact(sock, num_bytes):
    data = b''
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            return None
        data += chunk
    return data
```
> "A single `recv()` call may return fewer bytes than requested (partial read). This loop ensures we always get exactly the number of bytes we expect. This is a common pitfall in socket programming."

---

### 4.3 `network.py` – TCP Server & Connection Management

**What it does:**
- `PeerServer` manages both **inbound** (listening) and **outbound** (connecting) TCP connections.
- Uses a **thread-per-connection** model – each connection gets its own recv loop thread.
- Handles connection lifecycle: accept, handshake, recv loop, disconnect cleanup.

**Key design decisions:**

1. **Duplicate connection prevention:**
   ```
   Peer A (port 5001) connects to Peer B (port 5002)
   Peer B (port 5002) also connects to Peer A (port 5001)
   → Two connections for the same pair!
   ```
   > "I use a port-number tiebreaker: the peer with the lower port keeps its outbound connection, and the higher-port peer closes its duplicate. This avoids double message delivery."

2. **Retry logic with backoff:**
   - `connect_to_peer()` retries 3 times with a 2-second delay between attempts.
   - Handles `ConnectionRefusedError` gracefully (peer may not be up yet).

3. **Thread-safe connection registry:**
   - `self.connections: Dict[int, PeerConnection]` protected by `threading.Lock()`.
   - Every `send()` is also lock-protected per-connection.

4. **Callback-based architecture:**
   ```python
   PeerServer(
       on_message=self._handle_message,   # called on every received message
       on_connect=self._on_connect,       # called when a peer connects
       on_disconnect=self._on_disconnect  # called when a peer disconnects
   )
   ```
   > "This decouples network I/O from application logic. `peer.py` only deals with message semantics; `network.py` handles all the socket plumbing."

---

### 4.4 `metrics.py` – Per-Peer & Swarm-Level Metrics

**What it does:**
- `PeerMetrics` tracks per-peer stats: bytes uploaded/downloaded, request counts, timeouts, failures.
- `SwarmMetrics` aggregates across all peers into a **5-feature vector** for the ML model.

**Per-peer metrics (`PeerMetrics`):**
| Metric | Description |
|---|---|
| `bytes_uploaded` | Bytes WE sent TO this peer |
| `bytes_downloaded` | Bytes WE received FROM this peer |
| `upload_rate()` | Bytes/sec over a 60-second sliding window |
| `download_rate()` | Bytes/sec over a 60-second sliding window |
| `upload_download_ratio()` | `uploaded / downloaded` (freeriders have low ratio) |
| `timeout_count` | Number of timed-out requests |
| `failure_count` | Number of hash verification failures |

**Swarm-level features (`SwarmMetrics.feature_vector()`):**
```python
[avg_upload_ratio, low_contributor_pct, churn_rate, timeout_rate, failure_rate]
```

| Feature | Meaning | Impact |
|---|---|---|
| `avg_upload_ratio` | Mean up/down ratio across all peers | Low → peers aren't reciprocating |
| `low_contributor_pct` | % of peers with ratio < 0.3 | High → many freeriders |
| `churn_rate` | (connects + disconnects) / minute over 2-min window | High → unstable swarm |
| `timeout_rate` | timeouts / total requests | High → peers are unresponsive |
| `failure_rate` | hash failures / total requests | High → data corruption |

**Rate computation detail:**
> "I use a **sliding window** approach: only transfers within the last 60 seconds are considered for rate calculation. Old entries are pruned periodically. This gives a responsive, up-to-date rate rather than a lifetime average."

---

### 4.5 `strategies.py` – Unchoking Strategies

**What it does:**
- Implements 3 peer-selection strategies that decide **which peers to serve** (unchoke).
- Each strategy has a `select_peers(swarm_metrics) → list[port]` interface.

**Strategy 1: TIT_FOR_TAT (cooperative)**
```
Algorithm:
  1. Sort all peers by download_rate (how much they give US)
  2. Pick top N-1 (best contributors)
  3. Add 1 random "optimistic unchoke" (gives new peers a chance)
```
> "This mirrors real BitTorrent: reward reciprocators. The optimistic unchoke prevents the network from becoming a closed clique – new peers always get at least one chance to prove themselves."

**Strategy 2: STRICT_MODE (anti-freerider)**
```
Algorithm:
  1. Compute reciprocity ratio per peer = bytes_they_gave_us / bytes_we_gave_them
  2. Only unchoke peers with ratio > 0.8
  3. NO optimistic unchoke – strict enforcement
```
> "When the swarm has many freeriders, you need to stop subsidizing them. This strategy is aggressive: no optimistic unchoke means freeriders get zero service."

**Strategy 3: EXPLORATION_MODE (discovery)**
```
Algorithm:
  1. Ignore all historical scores
  2. Randomly select N peers to unchoke
  3. All selections are optimistic (exploratory)
```
> "Used when the swarm is full of new or unknown peers. By ignoring history and randomly sampling, we quickly gather data about who's a good contributor. Think of it as the 'exploration' phase in an explore-exploit tradeoff."

**Factory pattern:**
```python
_strategies = {
    Strategy.TIT_FOR_TAT: TitForTat(),
    Strategy.STRICT_MODE: StrictMode(),
    Strategy.EXPLORATION_MODE: ExplorationMode(),
}
```
> "Singleton instances indexed by an IntEnum. The ML model outputs 0/1/2, which maps directly to the enum. New strategies can be added without touching `peer.py`."

---

### 4.6 `ml_model.py` – AI Strategy Selector

**What it does:**
- Trains a **RandomForestClassifier** (50 trees, max depth 8) on 500 synthetic samples.
- Maps the 5-feature vector → one of 3 strategies.
- Provides a **rule-based fallback** when sklearn is unavailable.

**Training data generation (synthetic):**
```python
Features: [avg_ratio, low_pct, churn, timeout, failure]  # uniformly sampled

Labeling rules:
  if low_pct > 0.4 or failure > 0.3  → STRICT_MODE
  if avg_ratio < 0.4 and low_pct < 0.2  → EXPLORATION_MODE
  if churn > 5.0 and low_pct < 0.3  → EXPLORATION_MODE
  else  → TIT_FOR_TAT
```

> "I use synthetic data because I don't have real-world swarm telemetry. The labeling heuristics encode domain knowledge: 'if freeriders are rampant, crack down; if contributions are low but no clear freeriders, explore for better peers.' The RandomForest learns the decision boundaries from these rules, including cases near the thresholds where the heuristic alone might be brittle."

**Why RandomForest?**
- Works well with small datasets (500 samples).
- Handles non-linear decision boundaries.
- Naturally handles the multi-class output (3 strategies).
- Fast inference (< 1ms per prediction).
- No feature scaling needed.

**Model persistence:**
```python
# Save
pickle.dump(model, open('strategy_model.pkl', 'wb'))
# Load
model = pickle.load(open('strategy_model.pkl', 'rb'))
```

**Fallback mechanism:**
> "If sklearn isn't installed or the model file is missing, `predict_strategy()` automatically falls back to `rule_based_fallback()`, which uses the exact same heuristic rules as the training data. The system is never broken – it just degrades gracefully."

---

### 4.7 `peer.py` – Main Orchestrator

**What it does:**
- Central entry point; creates a `Peer` object with all the components wired together.
- Runs 4-5 concurrent background threads depending on role.

**Peer roles:**
| Role | Behavior | Threads |
|---|---|---|
| `seed` | Has all pieces, serves unconditionally | HAVE broadcast |
| `normal` | Tit-for-Tat strategy | Download + unchoke + HAVE broadcast |
| `freerider` | Downloads but uploads only 5% of the time | Download + unchoke + HAVE broadcast |
| `adaptive` | Switches strategy every 30s via ML | Download + unchoke + adaptive + HAVE broadcast |

**Thread model:**
```
Main Thread: periodic status logging every 15s
  ├── Download Loop (3s interval): request missing pieces
  ├── Unchoke Loop (10s interval): re-evaluate unchoked peers via strategy
  ├── Adaptive Loop (30s interval): [adaptive only] ML-based strategy switch
  └── HAVE Broadcast Loop (10s interval): announce piece ownership
```

**Message dispatch:**
```python
def _handle_message(self, peer_port, msg):
    if msg_type == 'REQUEST':  → _handle_request()   # decide to serve or reject
    elif msg_type == 'PIECE':  → _handle_piece()      # verify hash, store, broadcast HAVE
    elif msg_type == 'HAVE':   → _handle_have()       # update peer_pieces map
    elif msg_type == 'REJECT': → _handle_reject()     # clear pending request
```

**Piece request pipeline:**
```
1. Missing pieces = total_pieces - owned_pieces - pending_requests
2. For each missing piece (up to 3 at a time):
   a. Find a connected peer who has it (from peer_pieces map)
   b. Send REQUEST message
   c. Add to pending_requests with timestamp
3. If pending request > 15 seconds → timeout + retry
```

---

### 4.8 `run_simulation.py` – Simulation Runner

**What it does:**
- Launches all 4 peers as subprocesses with a 2-second stagger.
- Captures stdout to individual log files (`peer_5001.log`, etc.).
- Runs for a configurable duration (default 90 seconds).
- Terminates all processes and prints combined logs at the end.

---

## 5. End-to-End Data Flow

```
                          STARTUP
                            │
    utils.py ── split testfile.txt into 1024B pieces + SHA1 hashes
    ml_model.py ── load strategy_model.pkl (or train if missing)
                            │
                       NETWORKING
                            │
    network.py ── TCP bind on 127.0.0.1:port
                ── connect outbound to known peers
                ── exchange HANDSHAKE messages
                ── broadcast initial HAVE messages
                            │
                     STEADY STATE
                            │
    ┌───────── Download Loop (every 3s) ─────────┐
    │  identify missing pieces                     │
    │  find peer with piece (from HAVE data)       │
    │  send REQUEST → receive PIECE or REJECT      │
    │  verify SHA1 → store + broadcast HAVE        │
    └──────────────────────────────────────────────┘
                            │
    ┌───────── Unchoke Loop (every 10s) ──────────┐
    │  run current strategy's select_peers()       │
    │  update unchoked_peers set                   │
    │  (controls who gets served on REQUEST)       │
    └──────────────────────────────────────────────┘
                            │
    ┌──────── Adaptive Loop (every 30s) ──────────┐  ← adaptive peer only
    │  metrics.py → 5-feature vector               │
    │  ml_model.py → predict_strategy()            │
    │  switch strategy if prediction differs       │
    │  log contributor analysis                    │
    └──────────────────────────────────────────────┘
```

---

## 6. Concurrency Model

| Concern | Mechanism |
|---|---|
| Multiple connections | Thread-per-connection (daemon threads) |
| Connection registry | `threading.Lock()` on `PeerServer.connections` |
| Send operations | Per-connection `threading.Lock()` |
| Unchoked peers set | `threading.Lock()` (`_unchoke_lock`) |
| Pending requests | `threading.Lock()` (`_request_lock`) |
| Metric recording | `threading.Lock()` per `PeerMetrics` instance |

> "I used **locks** over queues for simplicity. Each shared data structure has its own lock for fine-grained locking – this avoids a single global lock bottleneck while keeping the code readable."

---

## 7. Key Design Decisions

| Decision | Rationale |
|---|---|
| **Thread-per-connection** (not async) | Simpler to reason about for a 4-peer simulation; async would be overkill |
| **Length-prefixed JSON** (not raw TCP) | Handles partial reads; JSON is debug-friendly |
| **Synthetic training data** | No real-world data available; heuristic labels encode domain knowledge |
| **RandomForest** (not neural net) | Small dataset, fast inference, no scaling needed, interpretable |
| **SHA1 piece verification** | Mirrors real BitTorrent; prevents corrupted pieces from propagating |
| **Optimistic unchoke in TFT** | Prevents starvation of new peers (bootstrap problem) |
| **Callback-based network layer** | Decouples I/O from business logic; easy to test |
| **Rule-based fallback** | Graceful degradation when ML dependencies are missing |
| **IntEnum for strategies** | ML model outputs 0/1/2 → maps directly; extensible |
| **Singleton strategy instances** | Avoid re-creation overhead; strategies are stateless |

---

## 8. Interview Q&A

### Architecture & Systems

**Q: How do peers discover each other?**
> Peers are configured with a static list of known ports (default: 5001–5004). On startup, each peer attempts outbound TCP connections to all known peers. There's no tracker or DHT – this keeps the simulation simple while still demonstrating peer-to-peer communication.

**Q: How do you handle the TCP stream boundary problem?**
> TCP is a byte stream, not a message stream. I use a 4-byte big-endian length prefix before each JSON payload. The receiver first reads exactly 4 bytes to get the message length, then reads exactly that many bytes for the payload. The `_recv_exact()` function loops until all bytes are received, handling partial reads.

**Q: What happens if two peers try to connect to each other simultaneously?**
> Both peers initiate outbound connections, resulting in two TCP connections for the same pair. On inbound handshake, I check if a connection already exists. If it does, I use a **port-number tiebreaker**: the peer with the lower port number keeps its outbound connection, and the other peer's duplicate inbound is closed. This deterministically resolves to exactly one connection per pair.

**Q: Why threads instead of async/await?**
> For a 4-peer simulation, threading is simpler and more readable. The `thread-per-connection` model maps naturally to the problem (each connection independently reads messages). Async would add complexity (event loops, coroutine management) without meaningful performance benefit at this scale. In production BitTorrent, you'd use async I/O to handle thousands of connections.

**Q: How do you ensure thread safety?**
> Each shared data structure has its own `threading.Lock()`: the connection registry, the unchoked-peers set, the pending-requests dict, and each `PeerMetrics` instance. I use fine-grained locking to avoid bottlenecks. All daemon threads stop when the main thread exits.

---

### ML / Adaptive Strategy

**Q: Why did you use a RandomForest classifier?**
> RandomForest works well with small datasets (500 samples), requires no feature scaling, handles multi-class classification naturally, and is fast at inference (< 1ms). It also captures non-linear decision boundaries between the 3 strategies. A neural network would be overkill and harder to debug for this use case.

**Q: Isn't training on synthetic data a problem?**
> Yes, it's a limitation. The synthetic data encodes expert heuristics (e.g., "high freerider % → strict mode"). The model learns these boundaries, including soft transitions near thresholds. In production, I'd collect real swarm telemetry and retrain. The rule-based fallback ensures the system works even without the ML model.

**Q: What are the 5 features and why those?**
> 1. **avg_upload_ratio** – overall health of reciprocity in the swarm
> 2. **low_contributor_pct** – fraction of suspected freeriders
> 3. **churn_rate** – connects + disconnects per minute (swarm stability)
> 4. **timeout_rate** – unresponsive peers
> 5. **failure_rate** – data corruption / malicious peers
>
> These capture the three threats the adaptive peer needs to respond to: freeriding, instability, and unreliability.

**Q: How does the adaptive loop work end-to-end?**
> Every 30 seconds: (1) `SwarmMetrics.feature_vector()` computes 5 floats from live peer data, (2) `predict_strategy(model, features)` runs the RandomForest prediction, (3) if the predicted strategy differs from the current one, the peer switches. The new strategy immediately affects the unchoke loop (every 10s), which controls which peers get served.

**Q: What's the fallback if sklearn isn't installed?**
> `predict_strategy()` catches the case where `model is None` and calls `rule_based_fallback()`. This function applies the same `if/elif/else` heuristic rules used to generate the training labels. It's deterministic and dependency-free. The system gracefully degrades from ML to rules.

---

### BitTorrent Concepts

**Q: What is choking/unchoking?**
> In BitTorrent, "choking" means refusing to upload to a peer. "Unchoking" means allowing uploads. A peer periodically evaluates which connected peers to unchoke based on a strategy. In my simulation, the unchoke loop runs every 10 seconds and updates the `unchoked_peers` set. When a REQUEST arrives, the peer checks if the requester is in this set before serving.

**Q: Explain the Tit-for-Tat mechanism.**
> TFT follows a simple principle: *help those who help you*. I sort peers by how much they upload to me (their download_rate in my metrics). The top N-1 are unchoked. Additionally, I unchoke 1 random peer (optimistic unchoke) to let newcomers prove themselves. This creates a virtuous cycle: cooperative peers get served, which gives them more data to share back.

**Q: How does the freerider get punished?**
> A freerider downloads pieces but only uploads 5% of the time. Over time, its upload/download ratio drops below 0.3. In **TFT**, it won't rank in the top peers by upload rate, so it only gets served during occasional optimistic unchokes. In **STRICT_MODE**, it's completely blocked (ratio < 0.8 threshold). The adaptive peer detects high freerider % and switches to STRICT_MODE automatically.

**Q: What's the optimistic unchoke and why is it important?**
> Without it, new peers face a chicken-and-egg problem: they have nothing to trade, so nobody unchokes them, so they can never get any data to trade. The optimistic unchoke randomly gives one unchoked slot to a non-top peer, allowing newcomers to download a few pieces and start contributing. It's the bootstrap mechanism of BitTorrent.

---

### Error Handling & Edge Cases

**Q: What if a piece fails hash verification?**
> The peer logs a warning and discards the piece. It **does not** store it. The piece remains in the "missing" set and will be re-requested in the next download loop iteration (3 seconds later), potentially from a different peer. The failure is recorded in `PeerMetrics` and contributes to the `failure_rate` feature for the ML model.

**Q: What if a peer crashes mid-simulation?**
> The TCP connection drops, triggering a `recv()` return of empty bytes. The recv loop exits, `_on_disconnect()` is called, and the peer is removed from the connections dict. Other peers continue operating. The churn_rate metric increases, potentially triggering the adaptive peer to switch to EXPLORATION_MODE.

**Q: How are request timeouts handled?**
> Each pending request is timestamped. Every download loop iteration (3s), requests older than 15 seconds are cleared from `pending_requests`, making those piece indices available for re-request. This handles cases where a peer was choked and never responded.

---

## 9. Potential Improvements (If Asked)

| Improvement | Description |
|---|---|
| **Piece selection** | Implement "rarest-first" instead of sequential to improve piece availability |
| **Real training data** | Log actual swarm metrics during simulations and retrain the model |
| **Async I/O** | Replace threads with `asyncio` for scalability to hundreds of peers |
| **DHT/Tracker** | Implement peer discovery instead of hard-coded port list |
| **Encryption** | Add TLS for peer connections |
| **Pipelining** | Request multiple pieces per peer simultaneously instead of 3 global |
| **Feature engineering** | Add time-series features (trend of freerider %, rate of change) |
| **Online learning** | Update the model during runtime using reinforcement learning |

---

## 10. How to Run

```bash
# Step 1: Generate test file & train ML model
python utils.py
python ml_model.py

# Step 2: Option A – Automated (all 4 peers at once)
python run_simulation.py --duration 90

# Step 2: Option B – Manual (4 terminals)
python peer.py --port 5001 --role seed
python peer.py --port 5002 --role normal
python peer.py --port 5003 --role freerider
python peer.py --port 5004 --role adaptive
```

---

## 11. Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.x |
| Networking | Raw TCP sockets (`socket` module) |
| Concurrency | `threading` (thread-per-connection) |
| Wire format | Length-prefixed JSON |
| Hashing | SHA1 (`hashlib`) |
| ML | scikit-learn `RandomForestClassifier` |
| Serialization | `pickle` (model), `json` (messages), `base64` (piece data) |
