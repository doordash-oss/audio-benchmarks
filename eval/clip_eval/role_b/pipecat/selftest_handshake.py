# Copyright 2026 DoorDash, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prove the Python adapter completes the audio bridge
/signal WebRTC handshake.

Starts runtime/audio_bridge.js, connects a dummy Role A and our Role B
via OrchestratorConnection, and asserts both reach connection state "connected"
and that Role B receives an incoming audio track (bidirectional negotiation).
No OpenAI, no clips, no model — just the audio-leg handshake.

Run (needs node >=22.12 on PATH):
    .venv/bin/python selftest_handshake.py
"""

import asyncio
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import wave

from orchestrator_transport import OrchestratorConnection

EVAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
ORCH = os.path.join(EVAL_ROOT, "clip_eval", "runtime", "audio_bridge.js")


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
    # Node dependencies are installed at the eval root; mirror the runner's NODE_PATH.
    env = {**os.environ, "SIGNAL_PORT": str(port), "RECORD_DIR": rec,
           "WIRE_SAMPLE_RATE": "48000", "NODE_PATH": f"{EVAL_ROOT}/node_modules"}
    with open(log_path, "w") as logf:
        orch = subprocess.Popen(["node", ORCH], cwd=EVAL_ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)

    result = 1
    try:
        if not wait_for_line(log_path, f"listening on port {port}", 15):
            print("FAIL: audio bridge did not start")
            print(open(log_path).read())
            return 1
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
            result = 0
        except asyncio.TimeoutError:
            print("FAIL: timed out waiting for connection/incoming track")
            print("--- orchestrator log ---"); print(open(log_path).read())
            result = 1
        await a.close()
        await b.close()
    finally:
        if orch.poll() is None:
            orch.send_signal(signal.SIGINT)
            try:
                orch.wait(timeout=10)
            except subprocess.TimeoutExpired:
                orch.kill()
                orch.wait()

    if result == 0:
        frame_counts = {}
        for role in ("A", "B"):
            wav_path = os.path.join(rec, f"{role}.wav")
            try:
                with wave.open(wav_path, "rb") as recording:
                    frames = recording.getnframes()
                    assert recording.getframerate() == 48000
                    assert recording.getnchannels() == 1
                    assert 0 < frames < 48000 * 60
                    frame_counts[role] = frames
            except Exception as exc:  # noqa: BLE001 - report a useful self-test failure
                print(f"FAIL: invalid finalized {role}.wav: {exc!r}")
                print("--- audio bridge log ---"); print(open(log_path).read())
                return 1
        print("PASS: A.wav and B.wav have finalized RIFF headers")
        combined_path = os.path.join(rec, "combined.wav")
        try:
            with wave.open(combined_path, "rb") as recording:
                assert recording.getframerate() == 48000
                assert recording.getnchannels() == 2
                assert abs(recording.getnframes() - max(frame_counts.values())) <= 1
        except Exception as exc:  # noqa: BLE001 - report a useful self-test failure
            print(f"FAIL: invalid combined.wav: {exc!r}")
            print("--- audio bridge log ---"); print(open(log_path).read())
            return 1
        print("PASS: combined.wav preserves the longer channel")
    return result


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
