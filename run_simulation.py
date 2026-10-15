"""
run_simulation.py – Launch all 4 peers as subprocesses for testing.

Runs the simulation for a configurable duration, then captures all logs.
Usage: python run_simulation.py [--duration 90]
"""

import subprocess
import sys
import time
import argparse
import signal
import os
import glob

from utils import generate_test_file, PIECE_SIZE


def main():
    parser = argparse.ArgumentParser(description='Run P2P simulation with 4 peers')
    parser.add_argument('--duration', type=int, default=90,
                        help='Simulation duration in seconds (default: 90)')
    parser.add_argument('--pieces', type=int, default=48,
                        help='Number of 1 KiB pieces in the shared file (default: 48)')
    parser.add_argument('--dashboard', action='store_true',
                        help='Also serve the live dashboard on http://127.0.0.1:8765')
    args = parser.parse_args()

    peers = [
        {'port': 5001, 'role': 'seed'},
        {'port': 5002, 'role': 'normal'},
        {'port': 5003, 'role': 'freerider'},
        {'port': 5004, 'role': 'adaptive'},
    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    processes = []
    log_files = []

    # Fresh shared file + clean telemetry so the dashboard shows only this run
    generate_test_file(os.path.join(script_dir, 'testfile.txt'),
                       size_bytes=args.pieces * PIECE_SIZE)
    for stale in glob.glob(os.path.join(script_dir, 'telemetry', 'peer_*.json')):
        os.remove(stale)

    if args.dashboard:
        dash = subprocess.Popen([sys.executable, os.path.join(script_dir, 'dashboard.py')],
                                cwd=script_dir)
        processes.append(dash)
        print("  Dashboard: http://127.0.0.1:8765")

    print(f"=== BitTorrent P2P Simulation ===")
    print(f"Duration: {args.duration} seconds\n")

    # Start each peer with a 2-second stagger
    for i, peer in enumerate(peers):
        log_path = os.path.join(script_dir, f"peer_{peer['port']}.log")
        log_file = open(log_path, 'w')
        log_files.append((log_path, log_file))

        cmd = [
            sys.executable, os.path.join(script_dir, 'peer.py'),
            '--port', str(peer['port']),
            '--role', peer['role']
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=script_dir
        )
        processes.append(proc)
        print(f"  Started peer {peer['port']} ({peer['role'].upper()}) [PID: {proc.pid}]")

        if i < len(peers) - 1:
            time.sleep(2)  # Stagger starts

    print(f"\nAll peers running. Waiting {args.duration} seconds...")
    print("(Press Ctrl+C to stop early)\n")

    try:
        for elapsed in range(args.duration):
            time.sleep(1)
            if elapsed > 0 and elapsed % 15 == 0:
                print(f"  ... {elapsed}s elapsed")

            for proc in processes:
                if proc.poll() not in (None, 0):
                    print(f"  ! process {proc.pid} exited with code {proc.returncode}")
    except KeyboardInterrupt:
        print("\nStopping early...")

    # Terminate all processes
    print("\nStopping all peers...")
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()

    time.sleep(2)

    for proc in processes:
        if proc.poll() is None:
            proc.kill()
        proc.wait()

    # Close log files
    for _, lf in log_files:
        lf.close()

    # Print combined logs
    print("\n" + "=" * 60)
    print("SIMULATION LOGS")
    print("=" * 60)

    for log_path, _ in log_files:
        if os.path.exists(log_path):
            print(f"\n--- {os.path.basename(log_path)} ---")
            with open(log_path, 'r') as f:
                content = f.read()
                if content:
                    print(content)
                else:
                    print("  (empty)")

    print("\n=== Simulation Complete ===")


if __name__ == '__main__':
    main()
