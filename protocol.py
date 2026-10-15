"""
protocol.py – Message formats and TCP framing for peer communication.

Uses a simple length-prefixed JSON protocol over TCP:
  [4 bytes: message length][JSON payload]

Message Types:
  HANDSHAKE  – Identify peer on connection
  REQUEST    – Ask for a specific piece
  PIECE      – Send piece data (base64-encoded)
  HAVE       – Announce owned piece indices
  REJECT     – Decline a piece request (peer is choked)
"""

import json
import struct
import base64
import socket
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# 4-byte big-endian length prefix
HEADER_FORMAT = '!I'
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

# Maximum message size (16 MB safety limit)
MAX_MESSAGE_SIZE = 16 * 1024 * 1024


class MessageType:
    """Constants for message types."""
    HANDSHAKE = 'HANDSHAKE'
    REQUEST = 'REQUEST'
    PIECE = 'PIECE'
    HAVE = 'HAVE'
    REJECT = 'REJECT'


def create_handshake(port: int, role: str) -> Dict[str, Any]:
    """Create a HANDSHAKE message to identify this peer."""
    return {
        'type': MessageType.HANDSHAKE,
        'port': port,
        'role': role
    }


def create_request(piece_index: int) -> Dict[str, Any]:
    """Create a REQUEST message for a specific piece."""
    return {
        'type': MessageType.REQUEST,
        'piece_index': piece_index
    }


def create_piece(piece_index: int, data: bytes) -> Dict[str, Any]:
    """
    Create a PIECE message with base64-encoded data.

    Args:
        piece_index: Index of the piece.
        data: Raw bytes of the piece.
    """
    return {
        'type': MessageType.PIECE,
        'piece_index': piece_index,
        'data': base64.b64encode(data).decode('ascii')
    }


def create_have(piece_indices: List[int]) -> Dict[str, Any]:
    """Create a HAVE message announcing owned pieces."""
    return {
        'type': MessageType.HAVE,
        'piece_indices': piece_indices
    }


def create_reject(piece_index: int) -> Dict[str, Any]:
    """Create a REJECT message declining a piece request."""
    return {
        'type': MessageType.REJECT,
        'piece_index': piece_index
    }


def extract_piece_data(message: Dict[str, Any]) -> bytes:
    """
    Extract raw bytes from a PIECE message.

    Args:
        message: Decoded message dict with base64 'data' field.

    Returns:
        Raw bytes of the piece.
    """
    return base64.b64decode(message['data'])


def send_message(sock: socket.socket, msg_dict: Dict[str, Any]) -> None:
    """
    Send a length-prefixed JSON message over a socket.

    Protocol: [4 bytes big-endian length][JSON payload bytes]

    Args:
        sock: Connected socket.
        msg_dict: Dictionary to serialize and send.

    Raises:
        ValueError: If message exceeds MAX_MESSAGE_SIZE.
        OSError: If socket write fails.
    """
    payload = json.dumps(msg_dict).encode('utf-8')
    length = len(payload)
    if length > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message too large: {length} bytes")
    header = struct.pack(HEADER_FORMAT, length)
    sock.sendall(header + payload)


def recv_message(sock: socket.socket) -> Optional[Dict[str, Any]]:
    """
    Receive a length-prefixed JSON message from a socket.

    Returns:
        Decoded message dict, or None if connection closed/error.

    Raises:
        ValueError: If received message header indicates size > MAX_MESSAGE_SIZE.
    """
    # Read 4-byte header
    header_data = _recv_exact(sock, HEADER_SIZE)
    if header_data is None:
        return None

    length = struct.unpack(HEADER_FORMAT, header_data)[0]
    if length > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message too large: {length} bytes")
    if length == 0:
        return None

    # Read payload
    payload_data = _recv_exact(sock, length)
    if payload_data is None:
        return None

    try:
        return json.loads(payload_data.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"Failed to decode message payload: {e}")
        return None


def _recv_exact(sock: socket.socket, num_bytes: int) -> Optional[bytes]:
    """
    Receive exactly num_bytes from a socket.

    Returns:
        bytes object of exactly num_bytes, or None if connection closed.
    """
    data = b''
    while len(data) < num_bytes:
        try:
            chunk = sock.recv(num_bytes - len(data))
            if not chunk:
                return None
            data += chunk
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            return None
        except OSError as e:
            # EBADF (9) = socket already closed
            # ENOTCONN (107) = not connected
            # Other OS errors are also fatal for this connection
            logger.debug(f"Socket recv error: {e}")
            return None
    return data


if __name__ == '__main__':
    # Quick self-test using socket pair
    a, b = socket.socketpair()
    try:
        # Test HANDSHAKE
        send_message(a, create_handshake(5001, 'seed'))
        msg = recv_message(b)
        print(f"Handshake: {msg}")
        assert msg['type'] == 'HANDSHAKE'
        assert msg['port'] == 5001

        # Test REQUEST
        send_message(a, create_request(3))
        msg = recv_message(b)
        print(f"Request: {msg}")
        assert msg['type'] == 'REQUEST'
        assert msg['piece_index'] == 3

        # Test PIECE with data integrity
        test_data = b'hello world' * 100
        send_message(a, create_piece(3, test_data))
        msg = recv_message(b)
        extracted = extract_piece_data(msg)
        assert extracted == test_data
        print(f"Piece: index={msg['piece_index']}, data_len={len(extracted)}")

        # Test HAVE
        send_message(a, create_have([0, 1, 2, 3]))
        msg = recv_message(b)
        print(f"Have: {msg}")
        assert msg['piece_indices'] == [0, 1, 2, 3]

        # Test REJECT
        send_message(a, create_reject(5))
        msg = recv_message(b)
        print(f"Reject: {msg}")
        assert msg['piece_index'] == 5

        print("\nAll protocol tests passed!")
    finally:
        a.close()
        b.close()
