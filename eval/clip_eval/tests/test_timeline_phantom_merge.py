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

"""Guard: the phantom-merge coalescer must not fuse two real utterances.

`PHANTOM_MERGE_S` exists to collapse a Gemini-only artifact — Pipecat emits a
near-zero-length phantom agent_start/agent_stop pair (~3-60ms) around the real
one. The merge condition used to compare a span's real `agent_start` against
the PREVIOUS span's audio-duration-*projected* stop (start + audio_duration_s)
rather than its raw wall-clock stop marker. Streaming TTS queues audio faster
than realtime, so an ordinary pair of quick back-to-back OpenAI responses
routinely has a first span whose projected stop lands slightly after the
second span's real start — a negative "gap" that is always under the 0.25s
threshold regardless of how large it is. Two genuinely separate utterances
("Okay, adding that dessert now." / "Got it, churros added. Anything else?",
0.822s apart in real wall-clock time) were silently fused into one displayed
timeline bar whose start/end matched neither utterance's real timing.

Found by comparing a dashboard timeline bar against what was actually audible
in a run's combined.wav — the bar's text was correct (both utterances,
correctly ordered) but its start/end did not correspond to either one.
"""

from pathlib import Path

from clip_eval.lib.timeline import _parse_b_log_contract

LOG_LINE = '[{ts}] [roleB-viz] {body}'


def _write_log(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "adapter_B.log"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_back_to_back_utterances_are_not_merged(tmp_path):
    """Two real, separately-synthesized turns with a genuine ~0.8s raw gap
    must stay as two spans, even though the first span's projected stop
    (audio_duration_s past its start) lands after the second span's real
    agent_start."""
    lines = [
        LOG_LINE.format(ts="2026-01-01T00:00:39.903Z", body="agent_start"),
        LOG_LINE.format(ts="2026-01-01T00:00:40.432Z",
                         body='agent_transcript {"text": "Okay, adding that dessert now.", "chars": 30, "words": 5}'),
        LOG_LINE.format(ts="2026-01-01T00:00:40.433Z",
                         body='agent_stop {"audio_duration_s": 1.88, "audio_frames": 90240}'),
        LOG_LINE.format(ts="2026-01-01T00:00:41.255Z", body="agent_start"),
        LOG_LINE.format(ts="2026-01-01T00:00:41.842Z",
                         body='agent_transcript {"text": "Got it, churros added. Anything else?", "chars": 37, "words": 6}'),
        LOG_LINE.format(ts="2026-01-01T00:00:41.843Z",
                         body='agent_stop {"audio_duration_s": 3.16, "audio_frames": 151680}'),
    ]
    rows = _parse_b_log_contract(_write_log(tmp_path, lines), anchor=0.0)
    speech = [r for r in rows if r["kind"] == "agent_speech"]
    assert len(speech) == 2, f"expected 2 separate spans, got {len(speech)}: {speech}"
    assert speech[0]["text"] == "Okay, adding that dessert now."
    assert speech[1]["text"] == "Got it, churros added. Anything else?"
    # Each span's own width must match its own audio_duration_s, not some
    # value borrowed from the other span.
    assert round(speech[0]["t_end"] - speech[0]["t"], 3) == 1.88
    assert round(speech[1]["t_end"] - speech[1]["t"], 3) == 3.16


def test_genuine_gemini_phantom_is_still_merged(tmp_path):
    """A true near-zero-length phantom span (leading-edge artifact) must
    still coalesce into the real span that follows it."""
    lines = [
        LOG_LINE.format(ts="2026-01-01T00:00:10.000Z", body="agent_start"),
        LOG_LINE.format(ts="2026-01-01T00:00:10.003Z", body="agent_stop {}"),
        LOG_LINE.format(ts="2026-01-01T00:00:10.010Z", body="agent_start"),
        LOG_LINE.format(ts="2026-01-01T00:00:10.510Z",
                         body='agent_transcript {"text": "Got it, one taco.", "chars": 17, "words": 4}'),
        LOG_LINE.format(ts="2026-01-01T00:00:10.520Z",
                         body='agent_stop {"audio_duration_s": 0.51, "audio_frames": 24480}'),
    ]
    rows = _parse_b_log_contract(_write_log(tmp_path, lines), anchor=0.0)
    speech = [r for r in rows if r["kind"] == "agent_speech"]
    assert len(speech) == 1, f"expected the phantom to merge into 1 span, got {len(speech)}: {speech}"
    assert speech[0]["text"] == "Got it, one taco."


def test_mid_utterance_phantom_concatenates_text(tmp_path):
    """Gemini's two-text-chunk-one-continuous-audio case: both chunks must
    still land on one merged span in order."""
    lines = [
        LOG_LINE.format(ts="2026-01-01T00:00:20.000Z", body="agent_start"),
        LOG_LINE.format(ts="2026-01-01T00:00:20.050Z",
                         body='agent_transcript {"text": "Got it, lamb vindaloo.", "chars": 22, "words": 4}'),
        LOG_LINE.format(ts="2026-01-01T00:00:20.100Z",
                         body='agent_stop {"audio_duration_s": 0.1, "audio_frames": 4800}'),
        LOG_LINE.format(ts="2026-01-01T00:00:20.105Z", body="agent_start"),
        LOG_LINE.format(ts="2026-01-01T00:00:20.400Z",
                         body='agent_transcript {"text": "Would you like that mild or spicy?", "chars": 34, "words": 7}'),
        LOG_LINE.format(ts="2026-01-01T00:00:20.900Z",
                         body='agent_stop {"audio_duration_s": 0.795, "audio_frames": 38160}'),
    ]
    rows = _parse_b_log_contract(_write_log(tmp_path, lines), anchor=0.0)
    speech = [r for r in rows if r["kind"] == "agent_speech"]
    assert len(speech) == 1
    assert speech[0]["text"] == "Got it, lamb vindaloo. Would you like that mild or spicy?"
