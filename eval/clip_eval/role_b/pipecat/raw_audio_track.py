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

"""OutgoingAudioTrack — an aiortc audio source fed from a byte buffer.

Adapted from Pipecat's smallwebrtc RawAudioTrack. We push PCM16 mono bytes in
(from the model / an echo loop); the track hands aiortc one 10 ms frame at a
time, paced to the wall clock, emitting silence whenever the buffer is empty so
the track never stalls. This is the B->orchestrator ("downlink") side of the
adapter's audio leg.
"""

import asyncio
import fractions
import time
from collections import deque

import numpy as np
from aiortc.mediastreams import AudioStreamTrack
from av import AudioFrame

_PTIME = 0.01            # 10 ms frames — matches the orchestrator's 10 ms pacing
_TARGET_FRAMES = 5       # ~50 ms target buffer depth (matches the reference adapter's BUFFER_TARGET)
# Max CONSECUTIVE silence to preserve (a natural pause after a period / between sentences).
# The old drain dropped EVERY silent frame once depth>target — but burst delivery keeps the
# buffer deep for the whole turn, so it stripped ALL intra-utterance pauses, jamming
# sentences together (the "still sounds fast / no pause after periods" symptom). We now keep
# pauses up to this length and only collapse the EXCESS (e.g. long trailing silence), which
# still bounds drift. Native preserves pauses naturally because WebRTC paces it at the source.
_MAX_PAUSE_FRAMES = 80   # ~800 ms — comfortably covers a sentence-boundary pause
# Runaway backstop only — NOT the primary drift control (that's the silence-drain
# below, which is inaudible). It must be large enough to hold a whole agent turn,
# because Pipecat delivers the model's audio over a WebSocket in a BURST (a multi-
# second utterance arrives near-instantly), unlike the native adapter's WebRTC leg
# which is already real-time-paced at the source. With the old 1 s (100-frame) cap,
# every frame past 1 s was dropped as "oldest" — i.e. REAL SPEECH — truncating each
# turn to ~1 s and roughly halving the agent's audible audio (measured 8.9 s active
# vs the native adapter's 22.9 s on the same task), which also garbled Role A's
# whisper transcription. 30 s comfortably holds any single turn; the silence-drain
# still returns the buffer to target between turns so latency never accumulates.
_HARD_CAP_FRAMES = 3000  # ~30 s backstop (was 100 = 1 s, which truncated speech)
_SILENCE_RMS = 184       # ~ -45 dBFS int16 (matches the reference adapter's silence threshold)


class OutgoingAudioTrack(AudioStreamTrack):
    kind = "audio"

    def __init__(self, sample_rate: int = 48000):
        super().__init__()
        self._sr = sample_rate
        self._samples_per_frame = int(sample_rate * _PTIME)   # 480 @ 48 kHz
        self._bytes_per_frame = self._samples_per_frame * 2   # PCM16 mono
        self._queue: deque[bytes] = deque()
        self._buf = bytearray()
        self._timestamp = 0
        self._start: float | None = None
        self._silence_run = 0    # consecutive silent frames currently being enqueued

    def push_bytes(self, pcm: bytes) -> None:
        """Append PCM16 mono bytes; sliced into exact 10 ms frames internally.

        Drift control: a producer running faster than 48 kHz (e.g. OpenAI Realtime
        TTS, ~+0.4%) would otherwise grow the queue without bound, pushing B's audio
        progressively later. When the buffer is above target we DROP silent frames so
        it drains back toward target during pauses (inaudible) — mirroring the native
        adapter's silence-drain. A hard cap drops oldest frames as a last resort.
        Sub-frame remainders linger in self._buf and combine with the next push (no audio
        lost mid-stream); flush() emits the final remainder at end-of-stream."""
        self._buf.extend(pcm)
        while len(self._buf) >= self._bytes_per_frame:
            frame = bytes(self._buf[: self._bytes_per_frame])
            del self._buf[: self._bytes_per_frame]
            self._enqueue(frame)

    def flush(self) -> None:
        """Emit any sub-frame remainder, zero-padded to a full 10 ms frame, so the tail of
        the last utterance isn't dropped. Call when the output transport stops."""
        if self._buf:
            frame = bytes(self._buf).ljust(self._bytes_per_frame, b"\x00")[: self._bytes_per_frame]
            self._buf.clear()
            self._enqueue(frame)

    def _enqueue(self, frame: bytes) -> None:
        depth = len(self._queue)
        if self._is_silence(frame):
            self._silence_run += 1
            # Keep natural pauses; collapse only EXCESS consecutive silence, and only when
            # above target (genuine backlog). This preserves after-period pauses that the
            # old "drop any silent frame above target" rule stripped under burst delivery.
            if depth >= _TARGET_FRAMES and self._silence_run > _MAX_PAUSE_FRAMES:
                return
        else:
            self._silence_run = 0
        if depth >= _HARD_CAP_FRAMES:
            self._queue.popleft()  # bound worst-case latency
        self._queue.append(frame)

    @staticmethod
    def _is_silence(frame: bytes) -> bool:
        a = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        return float(np.sqrt(np.mean(a * a))) < _SILENCE_RMS

    async def recv(self) -> AudioFrame:
        # Pace output to real time: frame N is due at start + N*ptime.
        if self._start is None:
            self._start = time.time()
            self._timestamp = 0
        else:
            self._timestamp += self._samples_per_frame
        delay = (self._start + self._timestamp / self._sr) - time.time()
        if delay > 0:
            await asyncio.sleep(delay)

        chunk = self._queue.popleft() if self._queue else b"\x00" * self._bytes_per_frame
        samples = np.frombuffer(chunk, dtype=np.int16).reshape(1, -1)  # (channels, samples)
        frame = AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = self._sr
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, self._sr)
        return frame
