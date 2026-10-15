"""
utils.py – File splitting, hashing, and verification utilities.

Splits a file into fixed-size pieces, computes SHA1 hashes for integrity,
and provides verification for received pieces.
"""

import hashlib
import os

# Fixed piece size in bytes
PIECE_SIZE = 1024


def split_file(filepath, piece_size=PIECE_SIZE):
    """
    Read a file and split it into fixed-size byte chunks.

    Args:
        filepath: Path to the file to split.
        piece_size: Size of each piece in bytes (default 1024).

    Returns:
        List of bytes objects, each up to piece_size bytes.
    """
    pieces = []
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(piece_size)
            if not chunk:
                break
            pieces.append(chunk)
    return pieces


def hash_piece(data):
    """
    Compute SHA1 hex digest for a piece of data.

    Args:
        data: bytes object to hash.

    Returns:
        Hex string of SHA1 hash.
    """
    return hashlib.sha1(data).hexdigest()


def create_piece_hashes(pieces):
    """
    Compute SHA1 hashes for all pieces.

    Args:
        pieces: List of bytes objects.

    Returns:
        List of SHA1 hex digest strings.
    """
    return [hash_piece(p) for p in pieces]


def verify_piece(data, expected_hash):
    """
    Verify a piece's integrity by comparing its SHA1 hash.

    Args:
        data: bytes object to verify.
        expected_hash: Expected SHA1 hex digest.

    Returns:
        True if hash matches, False otherwise.
    """
    return hash_piece(data) == expected_hash


def reassemble_file(pieces, output_path):
    """
    Reassemble a complete file from ordered pieces.

    Args:
        pieces: Dict mapping piece_index -> bytes data.
        output_path: Path to write the reassembled file.
    """
    with open(output_path, 'wb') as f:
        for i in sorted(pieces.keys()):
            f.write(pieces[i])


def generate_test_file(filepath, size_bytes=5120):
    """
    Generate a test file with repeated content for simulation.

    Args:
        filepath: Path to create the test file.
        size_bytes: Approximate size in bytes.
    """
    content = (
        "BitTorrent P2P Simulation Test File\n"
        "====================================\n"
        "This file is used to test the peer-to-peer file sharing simulation.\n"
        "Each peer will split this file into 1024-byte pieces and share them.\n"
        "The adaptive peer uses ML-based strategy selection to optimize.\n"
        "\n"
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do\n"
        "eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim\n"
        "ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut\n"
        "aliquip ex ea commodo consequat. Duis aute irure dolor in\n"
        "reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla\n"
        "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in\n"
        "culpa qui officia deserunt mollit anim id est laborum.\n\n"
    )

    with open(filepath, 'w') as f:
        written = 0
        block_num = 0
        while written < size_bytes:
            block = f"[Block {block_num}] {content}"
            f.write(block)
            written += len(block.encode('utf-8'))
            block_num += 1


if __name__ == '__main__':
    # Quick self-test
    test_path = 'testfile.txt'
    generate_test_file(test_path)
    pieces = split_file(test_path)
    hashes = create_piece_hashes(pieces)
    print(f"File split into {len(pieces)} pieces")
    for i, h in enumerate(hashes):
        assert verify_piece(pieces[i], h), f"Piece {i} verification failed!"
        print(f"  Piece {i}: {len(pieces[i])} bytes, hash={h[:16]}...")
    print("All pieces verified successfully!")
