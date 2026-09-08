#!/usr/bin/env python3
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

"""
Pipecat-based Role B adapter for the Order Up benchmark.

Role B is the agent under test. This adapter satisfies the swappable Role B
contract (clip_eval/docs/contracts/30_role_b_protocol.md): one long-running
process the runtime drives over (1) CLI args, (2) stdin JSON lines
(configure_tools / inject), (3) stdout JSON lines (agent_stop / transcripts),
(4) HTTP tool calls -> POST {serverUrl}/api/tool, and (5) a WebRTC audio leg to
the orchestrator. The model leg is implemented with Pipecat so additional S2S
providers can be added behind the same shape.

Built up across milestones (see PLAN.md):
  M2 (THIS FILE): CLI parsing, stdin reader + queue-until-ready, structured
                  logging, clean shutdown on SIGTERM/SIGINT/stdin-close.
                  NO WebRTC, NO model yet.
  M3: orchestrator WebRTC transport (aiortc + /signal offerer).
  M4: Pipecat pipeline + OpenAI Realtime model leg.
  M5: agent_stop + transcripts on stdout. (`inject` is accepted + logged but NOT
      wired into the live session yet — the Order Up runner never sends it; TODO.)
  M6: tool calls -> POST /api/tool.
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import threading
import time

from aiortc.mediastreams import MediaStreamError
from av.audio.resampler import AudioResampler

from orchestrator_transport import OrchestratorConnection
from pipecat_pipeline import run_model_pipeline


def log(msg: str) -> None:
    """Free-form diagnostics -> stderr (captured to $RECORD_DIR/adapter_B.log).
    Per the contract, stderr is NOT part of the protocol; it's for the dashboard."""
    print(f"[{time.strftime('%H:%M:%S')}] [pipecat-roleB] {msg}", file=sys.stderr, flush=True)


def emit(obj: dict) -> None:
    """Contract events -> stdout, one JSON object per line. The runtime parses these."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def arm_hard_exit(grace: float) -> None:
    """Backstop: force process exit `grace` seconds after shutdown begins.

    aiortc + the Pipecat/websocket runner can leave stray tasks/threads that stall
    asyncio.run()'s loop teardown, leaving the process lingering until the runtime
    SIGKILLs it. This daemon timer guarantees a prompt exit regardless. (B.wav is
    written by the orchestrator, not by us, so forcing our exit truncates nothing.)"""
    def _kill():
        log(f"hard-exit backstop fired after {grace}s — forcing exit")
        os._exit(0)
    t = threading.Timer(grace, _kill)
    t.daemon = True
    t.start()


def parse_args(argv=None) -> argparse.Namespace:
    """Accept the full Role B CLI surface (contract §2). Unknown args are tolerated
    for forward-compatibility with future runner changes."""
    p = argparse.ArgumentParser(description="Pipecat Role B adapter (Order Up examinee)")
    p.add_argument("--role", required=True, choices=["B"], help="contract: fixed to B")
    p.add_argument("--signalUrl", required=True, help="orchestrator /signal WS endpoint")
    p.add_argument("--tokenServer", default=None,
                   help="accepted but unused: Pipecat's OpenAI Realtime uses the API key directly")
    p.add_argument("--model", default=None, help="provider-specific model id")
    p.add_argument("--voice", default=None, help="synth voice name (provider-specific)")
    p.add_argument("--systemPrompt", default="", help="examinee system prompt")
    p.add_argument("--turnMode", choices=["server_vad", "none"], default="server_vad")
    p.add_argument("--vadMode", choices=["slow", "fast"], default="slow")
    p.add_argument("--speakFirst", action="store_true", help="greet first instead of waiting")
    p.add_argument("--enableTranscription", action="store_true", help="emit transcript events")
    p.add_argument("--prefill", default="", help="optional priming text")
    p.add_argument("--prefillFile", default="", help="optional priming JSON file")
    p.add_argument("--provider", default=os.environ.get("ROLE_B_PROVIDER") or "openai",
                   help="S2S provider (additive): openai|gemini_live|ultravox|grok_realtime|cascaded|... "
                        "Defaults to $ROLE_B_PROVIDER (the runner exports it), else openai.")
    p.add_argument("--echo", action="store_true",
                   help="debug: echo the customer audio instead of running the model (M3 loopback)")
    args, unknown = p.parse_known_args(argv)
    if unknown:
        log(f"ignoring unknown args (forward-compat): {unknown}")
    return args


class RoleBAdapter:
    """Holds session state and the stdin/lifecycle plumbing shared by all milestones.
    M3/M4 will attach the WebRTC transport + Pipecat pipeline and set `ready`."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.tools: list = []                 # tool catalogue from configure_tools
        self.tool_server_url: str | None = None
        self._pending_stdin: list[dict] = []  # messages received before `ready`
        self.ready = False                    # True once the model channel is live
        self._tools_locked = False            # True once the pipeline captured self.tools
        self._stop = asyncio.Event()
        self.conn: OrchestratorConnection | None = None  # the WebRTC audio leg (M3)

    # ── stdin message handling ──────────────────────────────────────────────
    def handle_message(self, msg: dict) -> None:
        """Apply a single stdin message to the live session. In the M2 skeleton we
        only parse + log; M5/M6 wire these into the model + tool server."""
        mtype = msg.get("type")
        if mtype == "configure_tools":
            self.tools = msg.get("tools", []) or []
            self.tool_server_url = msg.get("serverUrl")
            names = ", ".join(t.get("name", "?") for t in self.tools)
            log(f"configure_tools: {len(self.tools)} tool(s) [{names}] serverUrl={self.tool_server_url}")
            # Tools are read once when the pipeline is built. The runner sends
            # configure_tools before that (queued, then flushed), so this is fine;
            # warn if it arrives later, since the live model won't pick it up.
            if self._tools_locked:
                log("WARNING: configure_tools arrived after pipeline build — NOT applied to the live model")
        elif mtype == "inject":
            # NOTE: accepted + logged but NOT wired into the live session (TODO).
            # The Order Up runner never sends inject; only configure_tools.
            log(f"inject: text={msg.get('text')!r} "
                f"trigger_response={msg.get('trigger_response')} "
                f"delay={msg.get('trigger_response_delay')}")
            # M5: create a user-role item in the session and optionally trigger a response.
        else:
            log(f"unknown stdin message type: {mtype!r}")

    def dispatch_or_queue(self, msg: dict) -> None:
        """Per contract §3: apply even when messages arrive before the model channel
        is open — so queue until `ready`, then replay (see flush_pending)."""
        if self.ready:
            self.handle_message(msg)
        else:
            self._pending_stdin.append(msg)
            log(f"queued stdin message (type={msg.get('type')!r}) — not ready yet")

    def flush_pending(self) -> None:
        if self._pending_stdin:
            log(f"flushing {len(self._pending_stdin)} queued stdin message(s)")
        pending, self._pending_stdin = self._pending_stdin, []
        for msg in pending:
            self.handle_message(msg)

    def start_stdin_reader(self, loop: asyncio.AbstractEventLoop) -> None:
        """Read newline-delimited JSON from stdin on a daemon thread, bridging each
        message onto the event loop via call_soon_threadsafe.

        Why a thread instead of asyncio's StreamReader: blocking `readline()` has no
        64 KiB line limit and no pipe-backpressure deadlock, so a pathological
        oversized line can't hang us; and it returns each line immediately (the
        `for line in sys.stdin` iterator's read-ahead buffering would otherwise delay
        the first configure_tools line until its buffer filled). EOF => request stop.
        The thread is a daemon, so it never blocks process exit on SIGTERM."""

        def _reader() -> None:
            try:
                while True:
                    raw = sys.stdin.readline()
                    if not raw:  # EOF
                        break
                    text = raw.strip()
                    if not text:
                        continue
                    try:
                        msg = json.loads(text)
                    except json.JSONDecodeError:
                        loop.call_soon_threadsafe(log, f"ignoring non-JSON stdin line: {text[:120]!r}")
                        continue
                    loop.call_soon_threadsafe(self.dispatch_or_queue, msg)
            finally:
                loop.call_soon_threadsafe(log, "stdin closed (EOF) — shutting down")
                loop.call_soon_threadsafe(self._stop.set)

        threading.Thread(target=_reader, name="stdin-reader", daemon=True).start()

    def request_stop(self, reason: str) -> None:
        log(f"received {reason} — stopping")
        self._stop.set()

    async def echo_loop(self) -> None:
        """M3 stand-in for the model: echo the customer's audio back as B's audio,
        proving the bidirectional pipe works (B.wav should contain A's speech). The
        Pipecat model pipeline replaces this in M4. Incoming frames are normalized to
        48 kHz mono PCM16 (the orchestrator wire format) and pushed to the outgoing track."""
        resampler = AudioResampler(format="s16", layout="mono", rate=48000)
        track = self.conn.incoming_track
        frames = 0
        while not self._stop.is_set():
            try:
                frame = await track.recv()
            except MediaStreamError:
                log("incoming track ended — stopping echo")
                self._stop.set()
                break
            for f in resampler.resample(frame):
                self.conn.outgoing.push_bytes(f.to_ndarray().tobytes())
            frames += 1
            if frames % 500 == 0:
                log(f"echo: forwarded {frames} frames of customer audio")

    # ── main run loop ─────────────────────────────────────────────────────────
    async def run(self) -> None:
        a = self.args
        log(f"starting: role={a.role} provider={a.provider} model={a.model} voice={a.voice} "
            f"vad={a.vadMode} turn={a.turnMode} speakFirst={a.speakFirst} "
            f"enableTranscription={a.enableTranscription} signalUrl={a.signalUrl}")
        log("connecting orchestrator audio leg" + (" + echo (debug)" if a.echo else " + Pipecat model"))

        self.start_stdin_reader(asyncio.get_running_loop())

        work_task: asyncio.Task | None = None
        try:
            self.conn = OrchestratorConnection(a.signalUrl, role="B", sample_rate=48000, on_log=log)
            await self.conn.connect()
            await self.conn.wait_incoming(timeout=30)   # wait for the customer audio track
            log("orchestrator connected; customer audio track up")
        except asyncio.TimeoutError:
            log("timed out waiting for the customer audio track — Role A likely never "
                "connected to the orchestrator; shutting down")
            self._stop.set()
        except Exception as e:  # noqa: BLE001 — never hang if the audio leg fails to come up
            log(f"failed to set up orchestrator audio leg: {e!r} — shutting down")
            self._stop.set()
        else:
            # Real ready signal now (transport + customer track up), replacing M2's
            # simulated-ready placeholder. Replay any queued stdin (e.g. configure_tools).
            self.ready = True
            self.flush_pending()
            if a.echo:
                work_task = asyncio.create_task(self.echo_loop())
            else:
                self._tools_locked = True  # tools captured below; later configure_tools won't apply
                work_task = asyncio.create_task(run_model_pipeline(
                    self.conn, a, self._stop, log, emit,
                    tools=self.tools, tool_server_url=self.tool_server_url))

        await self._stop.wait()
        # Shutting down. Arm the hard-exit backstop in case aiortc/Pipecat teardown
        # stalls the event loop (see arm_hard_exit). Graceful cleanup races it; if it
        # finishes first, main() exits immediately and the timer never fires.
        arm_hard_exit(5.0)
        if work_task:
            work_task.cancel()
            await asyncio.gather(work_task, return_exceptions=True)
        if self.conn:
            await self.conn.close()
        self.flush_pending()  # drain anything queued during shutdown
        log("shutdown complete.")


async def main_async(args: argparse.Namespace) -> None:
    adapter = RoleBAdapter(args)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, adapter.request_stop, sig.name)
    await adapter.run()


def main() -> None:
    args = parse_args()
    if args.role != "B":
        log(f"error: --role must be B (got {args.role!r})")
        sys.exit(2)
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass
    # Guarantee a prompt exit even if loop teardown left stray non-daemon threads.
    os._exit(0)


if __name__ == "__main__":
    main()
