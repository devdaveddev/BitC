"""Unit tests for the deterministic core. Run: python -m unittest discover -s tests"""
import os
import socket
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metrics import SwarmMetrics, PeerMetrics
from ml_model import predict_with_confidence, rule_based_fallback
from protocol import send_message, recv_message, create_piece, extract_piece_data
from strategies import Strategy, get_strategy
from utils import hash_piece, verify_piece


class MetricsTests(unittest.TestCase):
    def test_freerider_detection_uses_reciprocity(self):
        pm = PeerMetrics(1)
        pm.record_upload(4096)  # we gave 4 KiB, got nothing back
        self.assertTrue(pm.is_freerider())
        pm.record_download(4096)
        self.assertFalse(pm.is_freerider())

    def test_generous_peer_not_flagged(self):
        pm = PeerMetrics(2)
        pm.record_download(8192)
        self.assertFalse(pm.is_freerider())
        self.assertEqual(pm.reciprocity(), 1.0)

    def test_timeout_rate_uses_requests_sent(self):
        sw = SwarmMetrics()
        pm = sw.get_or_create(1)
        for _ in range(4):
            pm.record_request_sent()
        pm.record_timeout()
        self.assertAlmostEqual(sw.timeout_rate(), 0.25)


class StrategyTests(unittest.TestCase):
    def test_strict_excludes_freerider(self):
        sw = SwarmMetrics()
        good, bad = sw.get_or_create(1), sw.get_or_create(2)
        good.record_upload(1024); good.record_download(2048)
        bad.record_upload(4096)
        self.assertEqual(get_strategy(Strategy.STRICT_MODE).select_peers(sw), [1])


class MLTests(unittest.TestCase):
    def test_fallback_and_probabilities(self):
        feats = [0.3, 0.6, 2.0, 0.1, 0.1]
        self.assertEqual(rule_based_fallback(feats), 1)
        choice, proba = predict_with_confidence(None, feats)
        self.assertEqual(choice, 1)
        self.assertAlmostEqual(sum(proba), 1.0)


class ProtocolTests(unittest.TestCase):
    def test_piece_roundtrip_and_hash(self):
        a, b = socket.socketpair()
        try:
            data = os.urandom(1024)
            send_message(a, create_piece(7, data))
            msg = recv_message(b)
            self.assertEqual(extract_piece_data(msg), data)
            self.assertTrue(verify_piece(data, hash_piece(data)))
            self.assertFalse(verify_piece(data + b'x', hash_piece(data)))
        finally:
            a.close(); b.close()


if __name__ == '__main__':
    unittest.main()
