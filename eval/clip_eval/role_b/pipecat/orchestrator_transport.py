"""Orchestrator audio leg — the vendor-neutral half of the Role B adapter.

Connects to the benchmark orchestrator (shared/orchestrator.js) over its custom
WebSocket /signal protocol as the WebRTC *offerer*, establishes one bidirectional
48 kHz mono PCM16 audio track, and exposes:
  - .outgoing       : OutgoingAudioTrack — push B's audio bytes here (-> B.wav)
  - .incoming_track : the remote MediaStreamTrack carrying A's (customer) audio
This mirrors connectOrchestrator() in shared/role_b/reference/gptRealtime_adapter.js,
ported to Python/aiortc. It knows nothing about the model — that's the M4 leg.

/signal protocol (see shared/orchestrator.js):
  - on connect the orchestrator sends {"type":"ready"}
  - we send  {"role","type":"offer","sdp"} then ICE {"role","type":"candidate","candidate":{...}}
  - we recv  {"role","type":"answer","sdp"} and {"role","type":"candidate","candidate":{...}}
    (broadcast; we filter on our role)
"""

import asyncio
import json

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp
from websockets.asyncio.client import connect as ws_connect

from raw_audio_track import OutgoingAudioTrack


class OrchestratorConnection:
    def __init__(self, signal_url: str, role: str, sample_rate: int = 48000, on_log=print):
        self.signal_url = signal_url
        self.role = role
        self.sample_rate = sample_rate
        self._log = on_log
        self.pc: RTCPeerConnection | None = None
        self.ws = None
        self.outgoing = OutgoingAudioTrack(sample_rate)
        self.incoming_track = None
        self._incoming = asyncio.Event()
        self._connected = asyncio.Event()
        self._remote_set = False
        self._pending_candidates: list[dict] = []
        self._recv_task: asyncio.Task | None = None

    async def connect(self) -> None:
        # All traffic is localhost (adapter <-> orchestrator on the same host), so
        # use no ICE servers: host candidates connect directly, and we avoid external
        # STUN round-trips (faster) and aioice's STUN-retry-after-teardown noise.
        self.pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
        self.pc.addTrack(self.outgoing)

        @self.pc.on("track")
        def _on_track(track):
            if track.kind == "audio" and self.incoming_track is None:
                self.incoming_track = track
                self._incoming.set()
                self._log(f"[{self.role}] incoming audio track received")

        @self.pc.on("connectionstatechange")
        async def _on_state():
            state = self.pc.connectionState
            self._log(f"[{self.role}] connection state = {state}")
            if state == "connected":
                self._connected.set()

        # NOTE: aiortc gathers ICE non-trickle inside setLocalDescription(), so the
        # full candidate set ships embedded in the offer SDP — there is no
        # "icecandidate" event to forward. We still ACCEPT trickled candidates from
        # the orchestrator below (it uses wrtc, which does trickle).
        self.ws = await ws_connect(self.signal_url)
        self._recv_task = asyncio.create_task(self._recv_loop())

        offer = await self.pc.createOffer()
        await self.pc.setLocalDescription(offer)  # aiortc gathers ICE here (non-trickle)
        await self.ws.send(json.dumps({
            "role": self.role, "type": "offer", "sdp": self.pc.localDescription.sdp,
        }))
        self._log(f"[{self.role}] offer sent")

    async def _recv_loop(self) -> None:
        try:
            async for raw in self.ws:
                data = json.loads(raw)
                mtype, mrole = data.get("type"), data.get("role")
                if mtype == "ready":
                    continue
                if mrole != self.role:
                    continue
                if mtype == "answer":
                    await self.pc.setRemoteDescription(RTCSessionDescription(sdp=data["sdp"], type="answer"))
                    self._remote_set = True
                    self._log(f"[{self.role}] answer applied")
                    for c in self._pending_candidates:
                        await self._add_candidate(c)
                    self._pending_candidates.clear()
                elif mtype == "candidate":
                    c = data.get("candidate")
                    if not c or not c.get("candidate"):
                        continue
                    if self._remote_set:
                        await self._add_candidate(c)
                    else:
                        self._pending_candidates.append(c)  # can't add before remote desc
        except Exception as e:  # noqa: BLE001 — surface signaling errors, don't hang
            self._log(f"[{self.role}] signaling loop ended: {e!r}")

    async def _add_candidate(self, c: dict) -> None:
        sdp = c["candidate"]
        if sdp.startswith("candidate:"):
            sdp = sdp[len("candidate:"):]
        try:
            cand = candidate_from_sdp(sdp)
            cand.sdpMid = c.get("sdpMid")
            cand.sdpMLineIndex = c.get("sdpMLineIndex")
            await self.pc.addIceCandidate(cand)
        except Exception as e:  # noqa: BLE001
            self._log(f"[{self.role}] addIceCandidate failed: {e!r}")

    async def wait_connected(self, timeout: float = 20) -> None:
        await asyncio.wait_for(self._connected.wait(), timeout)

    async def wait_incoming(self, timeout: float = 20) -> None:
        await asyncio.wait_for(self._incoming.wait(), timeout)

    async def close(self) -> None:
        # Stop the signaling loop and join it before tearing down the PC, so a
        # message can't arrive mid-setRemoteDescription during teardown. Close ws
        # before pc so no new signaling can be received.
        if self._recv_task:
            self._recv_task.cancel()
            await asyncio.gather(self._recv_task, return_exceptions=True)
        if self.ws:
            await self.ws.close()
        if self.pc:
            await self.pc.close()
