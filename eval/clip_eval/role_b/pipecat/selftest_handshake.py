"""M3 sub-step 1 self-test: prove the Python adapter completes the orchestrator
/signal WebRTC handshake.

Starts the real shared/orchestrator.js, connects a dummy Role A and our Role B
via OrchestratorConnection, and asserts both reach connection state "connected"
and that Role B receives an incoming audio track (bidirectional negotiation).
No OpenAI, no clips, no model — just the audio-leg handshake.

Run (needs node >=22.12 on PATH):
    .venv/bin/python selftest_handshake.py
"""

import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import time

from orchestrator_transport import OrchestratorConnection

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ORCH = f"{REPO}/shared/orchestrator.js"


def free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_for_line(path: str, needle: str, timeout: float = 15) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if needle in open(path).read():
                return True
        except FileNotFoundError:
            pass
        time.sleep(0.2)
    return False


async def main() -> int:
    port = free_port()
    rec = tempfile.mkdtemp(prefix="handshake_")
    log_path = os.path.join(rec, "orch.log")
    # orchestrator.js lives in shared/ but its deps are in v2/node_modules; the real
    # runner makes this resolve via NODE_PATH (set in pipeline/scripts/paths.sh), so we
    # replicate that here. (Our Python adapter doesn't need this — only the Node orchestrator.)
    env = {**os.environ, "SIGNAL_PORT": str(port), "RECORD_DIR": rec,
           "WIRE_SAMPLE_RATE": "48000", "NODE_PATH": f"{REPO}/v2/node_modules"}
    with open(log_path, "w") as logf:
        orch = subprocess.Popen(["node", ORCH], cwd=f"{REPO}/v2", env=env, stdout=logf, stderr=subprocess.STDOUT)

    try:
        if not wait_for_line(log_path, f"signaling on port {port}", 15):
            print("FAIL: orchestrator did not start"); print(open(log_path).read()); return 1
        print(f"orchestrator up on port {port}")

        url = f"ws://localhost:{port}/signal"
        a = OrchestratorConnection(url, "A")
        b = OrchestratorConnection(url, "B")
        await a.connect()
        await b.connect()
        try:
            await asyncio.gather(a.wait_connected(20), b.wait_connected(20))
            await b.wait_incoming(20)
            print("PASS: A and B both connected; B received the customer audio track")
            ok = 0
        except asyncio.TimeoutError:
            print("FAIL: timed out waiting for connection/incoming track")
            print("--- orchestrator log ---"); print(open(log_path).read())
            ok = 1
        await a.close()
        await b.close()
        return ok
    finally:
        orch.kill()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
