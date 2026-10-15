"""
network.py – TCP socket handling for peer-to-peer communication.

PeerServer:
  - Listens on 127.0.0.1:port for incoming connections
  - Connects outbound to known peers
  - Routes messages to a callback handler
  - Thread-per-connection model for simplicity
"""

import socket
import threading
import logging
import time
from typing import Optional, Callable, Dict, List, Any

from protocol import send_message, recv_message, create_handshake

logger = logging.getLogger(__name__)


class PeerConnection:
    """Represents a single TCP connection to/from a remote peer."""

    def __init__(self, sock: socket.socket, peer_port: Optional[int] = None,
                 direction: str = 'inbound'):
        self.sock = sock
        self.peer_port = peer_port  # Set after HANDSHAKE for inbound
        self.direction = direction
        self.connected = True
        self._lock = threading.Lock()

    def send(self, msg_dict: Dict[str, Any]) -> bool:
        """Thread-safe message send."""
        with self._lock:
            if not self.connected:
                return False
            try:
                send_message(self.sock, msg_dict)
                return True
            except (OSError, BrokenPipeError, ConnectionResetError) as e:
                logger.debug(f"Send error to {self.peer_port}: {e}")
                self.connected = False
                return False

    def close(self) -> None:
        """Close the connection safely."""
        self.connected = False
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass


class PeerServer:
    """
    TCP server for peer-to-peer communication.

    Manages both inbound (listening) and outbound (connecting) connections.
    Each connection runs in its own thread with a recv_message loop.
    """

    def __init__(self, port: int, role: str,
                 on_message: Optional[Callable] = None,
                 on_connect: Optional[Callable] = None,
                 on_disconnect: Optional[Callable] = None):
        """
        Args:
            port: Local port to listen on.
            role: Peer role string (seed/normal/freerider/adaptive).
            on_message: Callback(peer_port, message_dict) for incoming messages.
            on_connect: Callback(peer_port) when a peer connects.
            on_disconnect: Callback(peer_port) when a peer disconnects.
        """
        self.port = port
        self.role = role
        self.on_message = on_message or (lambda p, m: None)
        self.on_connect = on_connect or (lambda p: None)
        self.on_disconnect = on_disconnect or (lambda p: None)

        self.connections: Dict[int, PeerConnection] = {}
        self._lock = threading.Lock()
        self._server_sock: Optional[socket.socket] = None
        self._running = False

    def start(self) -> None:
        """Start the server listener in a background thread."""
        self._running = True
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind(('127.0.0.1', self.port))
        self._server_sock.listen(10)
        self._server_sock.settimeout(1.0)  # For clean shutdown

        thread = threading.Thread(target=self._accept_loop, daemon=True)
        thread.start()
        logger.info(f"Server listening on 127.0.0.1:{self.port}")

    def _accept_loop(self) -> None:
        """Accept incoming connections."""
        while self._running:
            try:
                if self._server_sock is None:
                    break
                conn_sock, addr = self._server_sock.accept()
                thread = threading.Thread(
                    target=self._handle_connection,
                    args=(conn_sock,),
                    daemon=True
                )
                thread.start()
            except socket.timeout:
                continue
            except OSError:
                break

    def connect_to_peer(self, peer_port: int, retries: int = 3,
                        delay: float = 2.0) -> bool:
        """
        Connect outbound to a known peer.

        Args:
            peer_port: Port of the remote peer on localhost.
            retries: Number of connection attempts.
            delay: Seconds between retries.

        Returns:
            True if connected successfully.
        """
        # Don't connect to ourselves
        if peer_port == self.port:
            return False

        # Don't duplicate connections
        with self._lock:
            if peer_port in self.connections and self.connections[peer_port].connected:
                return True

        for attempt in range(retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)
                sock.connect(('127.0.0.1', peer_port))

                # Re-check under lock before registering (another thread may have connected)
                with self._lock:
                    if peer_port in self.connections and self.connections[peer_port].connected:
                        sock.close()
                        return True

                # Send handshake immediately
                send_message(sock, create_handshake(self.port, self.role))

                conn = PeerConnection(sock, peer_port, 'outbound')
                with self._lock:
                    self.connections[peer_port] = conn

                # Start receive thread
                thread = threading.Thread(
                    target=self._recv_loop,
                    args=(conn,),
                    daemon=True
                )
                thread.start()

                logger.info(f"Connected to peer {peer_port}")
                self.on_connect(peer_port)
                return True

            except (ConnectionRefusedError, socket.timeout, OSError) as e:
                if attempt < retries - 1:
                    logger.debug(
                        f"Connection to {peer_port} failed "
                        f"(attempt {attempt + 1}/{retries}): {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.warning(
                        f"Could not connect to peer {peer_port} "
                        f"after {retries} attempts"
                    )
                    return False
        return False

    def _handle_connection(self, sock: socket.socket) -> None:
        """Handle an inbound connection: wait for handshake then recv loop."""
        conn = PeerConnection(sock, direction='inbound')
        try:
            # Wait for HANDSHAKE
            sock.settimeout(10.0)
            msg = recv_message(sock)
            if msg is None or msg.get('type') != 'HANDSHAKE':
                logger.debug("Inbound connection failed handshake, closing")
                conn.close()
                return

            peer_port = msg.get('port')
            if not isinstance(peer_port, int) or not (1 <= peer_port <= 65535) \
                    or peer_port == self.port:
                logger.debug(f"Rejecting handshake with invalid port: {peer_port!r}")
                conn.close()
                return
            peer_role = msg.get('role', '?')
            conn.peer_port = peer_port

            with self._lock:
                existing = self.connections.get(peer_port)
                if existing and existing.connected:
                    # Tiebreaker: keep the connection initiated by the peer
                    # with the lower port number to avoid duplicates
                    if self.port < peer_port:
                        # We initiated an outbound first; reject this inbound
                        conn.close()
                        return
                    else:
                        # Accept inbound, close our outbound
                        existing.close()
                self.connections[peer_port] = conn

            logger.info(
                f"Accepted connection from peer {peer_port} (role={peer_role})"
            )
            self.on_connect(peer_port)

            # Enter recv loop
            self._recv_loop(conn)

        except Exception as e:
            logger.debug(f"Inbound connection error: {e}")
            conn.close()

    def _recv_loop(self, conn: PeerConnection) -> None:
        """Continuously receive messages from a connected peer."""
        if not conn.connected:
            return
        try:
            conn.sock.settimeout(None)  # Blocking reads
        except OSError:
            return

        while self._running and conn.connected:
            try:
                msg = recv_message(conn.sock)
                if msg is None:
                    break
                self.on_message(conn.peer_port, msg)
            except Exception as e:
                logger.debug(f"Recv error from {conn.peer_port}: {e}")
                break

        # Clean up
        peer_port = conn.peer_port
        conn.close()

        was_registered = False
        with self._lock:
            if self.connections.get(peer_port) is conn:
                del self.connections[peer_port]
                was_registered = True

        # Only report a disconnect if this was the live connection; a
        # superseded duplicate (tie-break) is not real churn.
        if peer_port and was_registered:
            logger.info(f"Peer {peer_port} disconnected")
            self.on_disconnect(peer_port)

    def send_to_peer(self, peer_port: int, msg_dict: Dict[str, Any]) -> bool:
        """
        Send a message to a specific connected peer.

        Args:
            peer_port: Port of the target peer.
            msg_dict: Message dictionary to send.

        Returns:
            True if sent successfully, False otherwise.
        """
        with self._lock:
            conn = self.connections.get(peer_port)

        if conn is None or not conn.connected:
            logger.debug(f"No active connection to peer {peer_port}")
            return False

        return conn.send(msg_dict)

    def broadcast(self, msg_dict: Dict[str, Any],
                  exclude: Optional[set] = None) -> None:
        """
        Broadcast a message to all connected peers.

        Args:
            msg_dict: Message dictionary to broadcast.
            exclude: Set of peer ports to exclude.
        """
        exclude = exclude or set()
        with self._lock:
            targets = [
                (port, conn) for port, conn in self.connections.items()
                if port not in exclude and conn.connected
            ]

        for port, conn in targets:
            conn.send(msg_dict)

    def get_connected_ports(self) -> List[int]:
        """Get list of currently connected peer ports."""
        with self._lock:
            return [
                port for port, conn in self.connections.items()
                if conn.connected
            ]

    def stop(self) -> None:
        """Stop the server and close all connections."""
        self._running = False

        if self._server_sock:
            try:
                self._server_sock.close()
            except OSError:
                pass
            self._server_sock = None

        with self._lock:
            for conn in self.connections.values():
                conn.close()
            self.connections.clear()

        logger.info("Server stopped")
