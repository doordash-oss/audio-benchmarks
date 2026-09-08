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

"""Build a unified timeline JSON for one eval run, used by the visualization
dashboard.

Inputs (from a run dir like outputs/OrderUp/<task_id>/<ts>/):
  adapter_A.log    — clipPlayer events (with ISO timestamps after our patch)
  adapter_B.log    — gptRealtime events (with ISO timestamps)
  tool_log.json    — every tool call with absolute epoch ms
  final_cart.json  — final cart state
  score.json       — eval score
  combined.wav     — stereo recording (A=left, B=right)

Output:
  <record_dir>/timeline.json — unified per-event list with recording-relative
                                t (seconds) plus metadata.

────────────────────────────────────────────────────────────────────────
TIMING MODEL — read this before changing any anchor / offset logic
────────────────────────────────────────────────────────────────────────

Three timebases exist in any eval run, and the dashboard's job is to
collapse them onto ONE (the wav file's intrinsic timeline) so event
markers visually align with the audio that plays at that moment.

1) WALL CLOCK — `Date.now()` / `new Date().toISOString()` on the host.
   Used by every adapter to timestamp every log line. Same physical
   clock across all processes (single host); strictly monotonic.
   Inputs: adapter_A.log, adapter_B.log, orchestrator.log, tool_log.json.

2) WAV-FILE TIMELINE — A.wav, B.wav, combined.wav all share this. Wav
   position N (seconds) ≡ sample_index_N / 48000. The wav timeline
   advances at exactly the orchestrator's wav-writer's pace, which is
   one sample per arriving audio frame from the orchestrator's
   per-role sink callback.

   Critically: WAV TIMELINE != WALL CLOCK.
   The two diverge for two reasons:
     a. STARTUP OFFSET (constant, ~0.8s): the orchestrator's wav
        writer is created when pc.ontrack fires for a role, but actual
        samples don't start flowing until the WebRTC media path is
        fully up — measured empirically as the writer's first emitted
        byte landing ~0.8s after the `sink(<role>)#1` log line. We
        calibrate this constant out per-run (see anchor calibration
        below).
     b. CONTENT DELAY (variable, used to grow without bound): for B,
        provider audio and the local 48 kHz bridge can run on slightly
        different clocks. Role B's transport resamples and paces provider
        output before it reaches the bridge, keeping this delay bounded.
        Older recordings produced by other adapters may still contain
        accumulated drift.

3) WEBRTC AUDIO SAMPLE TIMELINE — internal to each WebRTC track,
   driven by the SENDER's audio clock. We don't observe this directly;
   it manifests only as the rate at which samples arrive at the
   orchestrator's sink. For A's track, the sender is A's local pacer
   (clip files → fixed 100ms-tick); rate is locked to host clock so
   wav and wall agree (modulo the constant startup offset). For B's
   track, the sender is OpenAI's TTS audio output via two WebRTC hops
   (OpenAI → B's adapter → orchestrator). The drift this causes is
   what (1.b) above describes.

The dashboard's audio player plays combined.wav at native rate; wav
position == dashboard time, by definition. So every event marker the
dashboard renders MUST be in wav-file-timeline coordinates, not raw
wall coordinates.

WHAT THIS FILE DOES:
  - Pick a wall-clock anchor (`recording_start_iso`) — the
    `sink(A)#1` log line in orchestrator.log. This is the wall time at
    which the first audio byte arrived at the orchestrator from A.
  - Calibrate the anchor against A.wav's actual content (the first
    injected clip cross-correlation). This absorbs the constant
    startup offset.
  - Compute every event's t-value as (event_wall - calibrated_anchor),
    yielding wav-file-timeline coordinates.

WHAT THIS FILE DOES NOT DO (yet):
  - Per-role drift correction for B-side. The current provider transport
    keeps residual drift bounded, but exact behavior is provider-dependent.
    If you need sub-50ms B-side accuracy, the right
    fix is an offline B.wav RMS scan to find each agent_speech's
    actual onset position and use that as t. The orchestrator's
    inline `audio_onset role=B` log is also available but lags the
    wav file by 200-600ms due to write-stream queueing — useful as
    telemetry, not as the timing source of truth.

EFFECT ON PERCEIVED AUDIO AT RUNTIME (separate from dashboard):
  The dashboard's job is post-hoc visualization. But the same drift
  affected what A's routing oracle SAW during the eval: A receives B's
  audio via the orchestrator, which forwards from B's adapter at the
  pacer's rate — so A heard B with the same growing delay that landed
  in B.wav. Before the silence-drain fix, A's clip-injection decisions
  were reacting to audio progressively staler than wall (up to ~1s by
  end of session), and the conversation effectively played in slow
  motion. The fix bounds A's perceived delay to ~50ms — A's routing
  becomes reactive in near-real-time again.
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path

from clip_eval.pipeline.paths import REPO_ROOT, ARTIFACTS_PATH
ISO_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\]")


def parse_iso(s: str) -> float:
    """Parse ISO-8601 'Z' timestamp -> epoch seconds (float)."""
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
        tzinfo=timezone.utc
    ).timestamp()


def first_iso_in_log(path: Path) -> float | None:
    if not path.exists():
        return None
    for line in path.read_text(errors="ignore").splitlines():
        m = ISO_RE.match(line.strip())
        if m:
            return parse_iso(m.group(1))
    return None


def first_inject_iso(adapter_a_log: Path) -> float | None:
    """Find the wall-clock time of the first `[ClipPlayer] Injected ...` line
    in A's log. Returns epoch seconds (float)."""
    if not adapter_a_log.exists():
        return None
    for line in adapter_a_log.read_text(errors="ignore").splitlines():
        m = ISO_RE.match(line.strip())
        if not m:
            continue
        if "[ClipPlayer] Injected " in line:
            return parse_iso(m.group(1))
    return None


def recording_start_iso(orch_log: Path) -> float | None:
    """Find the ISO time of the orchestrator's `sink(A)#1 bytesIn=...` line.

    This is the orchestrator's first audio-frame callback for role A — i.e.
    the moment A.wav begins recording. Far more reliable than scanning the
    wav for audible content, which can latch onto background noise. Returns
    None for old orchestrator logs without ISO timestamps.
    """
    if not orch_log.exists():
        return None
    for line in orch_log.read_text(errors="ignore").splitlines():
        m = ISO_RE.match(line.strip())
        if not m:
            continue
        if "sink(A)#1 " in line:
            return parse_iso(m.group(1))
    return None


def calibrate_inject_offset_s(
    a_wav: Path, clip_wav: Path, expected_t: float, search_pad_s: float = 4.0
) -> float | None:
    """Cross-correlate `clip_wav` against `a_wav` around `expected_t` (the
    inject-log-based predicted position) and return `expected_t - true_t`.

    Positive return = log time is LATER than the actual byte-in-A.wav
    position by this many seconds. Applying it as an offset to every clip's
    `t` value (i.e. subtracting it) shifts blocks back so they align with
    where the audio actually plays.

    Returns None on any error (missing files, sample-rate mismatch, low
    correlation score).
    """
    try:
        import numpy as np
        from scipy.signal import correlate
        import wave
    except Exception as e:
        # Surface the import failure loudly — silently swallowing it
        # disables anchor calibration without any warning, leaving every
        # event ~0.5-0.8s mis-aligned with the audio playback.
        import sys
        print(
            f"[viz] WARNING: anchor calibration disabled — "
            f"numpy/scipy not importable ({type(e).__name__}: {e}). "
            f"Timeline t-values will be ~0.5-0.8s LATE relative to audio "
            f"playback. Launch with `--with numpy --with scipy` or add "
            f"them to the env.",
            file=sys.stderr,
        )
        return None
    if not a_wav.exists() or not clip_wav.exists():
        return None
    try:
        with wave.open(str(a_wav)) as w:
            sr_a = w.getframerate()
            n_a = w.getnframes()
            raw_a = w.readframes(n_a)
        A = np.frombuffer(raw_a, dtype=np.int16).astype(np.float32)
        with wave.open(str(clip_wav)) as w:
            sr_c = w.getframerate()
            n_c = w.getnframes()
            raw_c = w.readframes(n_c)
        C = np.frombuffer(raw_c, dtype=np.int16).astype(np.float32)
    except Exception:
        return None
    if sr_a != sr_c or len(C) < 1024:
        return None
    # When the inject log predates sink(A)#1 (negative expected_t), the
    # pacer queued the bytes for `-expected_t` seconds before A.wav
    # actually started recording. The audio content then lands very
    # close to A.wav position 0. Search from 0 in that case, widening
    # `hi` enough to fit the clip plus the pad.
    if expected_t < 0:
        lo = 0
        hi = min(len(A), int((max(0.0, expected_t) + search_pad_s) * sr_a) + len(C))
    else:
        lo = max(0, int((expected_t - search_pad_s) * sr_a))
        hi = min(len(A), int((expected_t + search_pad_s) * sr_a) + len(C))
    window = A[lo:hi]
    if len(window) < len(C):
        return None
    c_norm = C - C.mean()
    nrm = float(np.linalg.norm(c_norm))
    if nrm < 1e-6:
        return None
    c_norm = c_norm / nrm
    corr = correlate(window, c_norm, mode="valid", method="fft")
    best = int(np.argmax(np.abs(corr)))
    # Normalize correlation score so we can sanity-check
    seg = window[best:best + len(C)]
    seg_norm = float(np.linalg.norm(seg))
    score = abs(float(corr[best])) / (seg_norm + 1e-9)
    if score < 0.3:
        # Poor match — probably a fallback clip or the wrong audio region.
        return None
    true_sample = lo + best
    true_t = true_sample / sr_a
    return expected_t - true_t


def first_audible_sample_s(
    wav_path: Path,
    sustained_threshold_dbfs: float = -25.0,
    onset_threshold_dbfs: float = -35.0,
    win_ms: int = 50,
    min_sustained_ms: int = 200,
    walkback_max_s: float = 1.5,
) -> float | None:
    """Find the audio position (s) where speech begins in a PCM16 mono wav.

    Two-pass detection:
      Pass 1: scan forward for `min_sustained_ms` of sustained windowed RMS
              above `sustained_threshold_dbfs` (~-25 dBFS). This robustly
              latches onto a clip but lands on its loudest centre, ~0.3-0.5s
              past the actual speech onset.
      Pass 2: walk backwards (up to `walkback_max_s`) and return the EARLIEST
              window with RMS above `onset_threshold_dbfs` (~-35 dBFS, well
              above the background-noise floor of ~-44 dBFS RMS but low
              enough to catch a quiet "I" or "Can" before the louder part).
              Stops the walkback if a window falls below the noise floor —
              that's a clean separator between the previous clip/noise and
              the current speech onset.

    This is more accurate than just "sustained-loud" because speech onsets
    are typically quieter than the loudest mid-utterance vowels.
    """
    if not wav_path.exists():
        return None
    with wave.open(str(wav_path)) as w:
        sr = w.getframerate()
        n_to_read = min(w.getnframes(), sr * 60)
        frames = w.readframes(n_to_read)
    n_samples = len(frames) // 2  # int16
    samples = struct.unpack(f"<{n_samples}h", frames)
    win_n = max(1, int(sr * win_ms / 1000))
    min_sustained_wins = max(1, min_sustained_ms // win_ms)
    sustained_sq = (32768 * (10 ** (sustained_threshold_dbfs / 20))) ** 2
    onset_sq = (32768 * (10 ** (onset_threshold_dbfs / 20))) ** 2
    # Pre-compute mean-squared per non-overlapping window (faster than nesting
    # the Python loop on every walk-back).
    win_count = (n_samples - win_n + 1) // win_n + 1
    win_msq = []
    for i in range(0, n_samples - win_n + 1, win_n):
        s = 0.0
        for j in range(i, i + win_n):
            v = samples[j]
            s += v * v
        win_msq.append((i, s / win_n))
    # Pass 1: forward-scan for sustained-loud.
    consec = 0
    first_above = None
    sustained_window_idx = None
    for idx, (i, msq) in enumerate(win_msq):
        if msq > sustained_sq:
            if first_above is None:
                first_above = idx
            consec += 1
            if consec >= min_sustained_wins:
                sustained_window_idx = first_above
                break
        else:
            first_above = None
            consec = 0
    if sustained_window_idx is None:
        return None
    # Pass 2: walk back from `sustained_window_idx` up to `walkback_max_s`
    # looking for the earliest window above the onset threshold. Stop at
    # any clearly-below-noise window (treated as a silence separator).
    walkback_wins = max(1, int((walkback_max_s * 1000) // win_ms))
    earliest_onset_idx = sustained_window_idx
    floor_sq = onset_sq * 0.3  # ~5dB below onset = clear silence separator
    for idx in range(sustained_window_idx - 1, max(-1, sustained_window_idx - walkback_wins - 1), -1):
        msq = win_msq[idx][1]
        if msq < floor_sq:
            # Crossed back into silence — stop walking back.
            break
        if msq > onset_sq:
            earliest_onset_idx = idx
    return win_msq[earliest_onset_idx][0] / sr


def iter_log_lines(path: Path):
    if not path.exists():
        return
    for raw in path.read_text(errors="ignore").splitlines():
        line = raw.strip()
        m = ISO_RE.match(line)
        ts = parse_iso(m.group(1)) if m else None
        body = line[m.end():].strip() if m else line
        yield ts, body


def parse_clip_player_events(path: Path, anchor: float) -> list[dict]:
    """Pick out the events we care about from adapter_A.log.

    Returns a list of:
      - customer_clip_injected events (clip injections)
      - routing events (A-side router decisions with reasoning)
      - agent_speech_a_heard ranges (A's whisper transcription of B's
        audio, used to populate the "A heard" block on agent_speech rows)
    """
    events = []
    if not path.exists():
        return events
    # Pass 1: collect inject_start times per node_id as ordered queues — the
    # same clip can be injected multiple times (e.g. fallback repeats), and
    # each Injected event must pair with its OWN inject_start.
    inject_start_queue: dict[str, list[float]] = {}
    for ts, body in iter_log_lines(path):
        if ts is None:
            continue
        m = re.search(r"\[ClipPlayer\] inject_start (\S+)$", body)
        if m:
            inject_start_queue.setdefault(m.group(1), []).append(ts)

    # A's input-audio transcription (whisper-1 transcribing B's audio).
    # Key: item_id → text. Joined with A-side speech ranges in pass 2.
    a_input_transcripts: dict[str, str] = {}
    # A's perception of agent speech ranges. Each entry: (t_start, t_end,
    # item_id). Used to attribute transcripts to time ranges, which
    # downstream attaches to B-side agent_speech rows.
    a_speech_ranges: list[dict] = []
    _a_started: float | None = None
    _a_started_iid: str | None = None
    for ts, body in iter_log_lines(path):
        if ts is None:
            continue
        if "conversation.item.input_audio_transcription.completed" in body:
            mm = re.search(r"\{.*\}\s*$", body)
            if mm:
                try:
                    evt = json.loads(mm.group(0))
                    iid = evt.get("item_id")
                    txt = (evt.get("transcript") or "").strip()
                    if iid and txt:
                        a_input_transcripts[iid] = txt
                except Exception:
                    pass
            continue
        if "input_audio_buffer.speech_started" in body:
            mm = re.search(r'"item_id"\s*:\s*"([^"]+)"', body)
            _a_started = ts
            _a_started_iid = mm.group(1) if mm else None
            continue
        if "input_audio_buffer.speech_stopped" in body and _a_started is not None:
            mm = re.search(r'"item_id"\s*:\s*"([^"]+)"', body)
            iid = (mm.group(1) if mm else _a_started_iid)
            a_speech_ranges.append({
                "t": round(_a_started - anchor, 3),
                "t_end": round(ts - anchor, 3),
                "item_id": iid,
            })
            _a_started = None
            _a_started_iid = None
            continue

    # Resolve A's transcripts onto A's speech ranges. The completed event
    # often arrives AFTER both started+stopped so we have to do this in a
    # second pass.
    for r in a_speech_ranges:
        r["text"] = a_input_transcripts.get(r["item_id"] or "", "")
    # Emit as agent_a_heard markers — downstream merges them onto
    # agent_speech rows AND surfaces them as gap-band VAD markers in the
    # mini-timeline. Emit EVERY pair (not just transcribed ones) so the
    # gap renderer sees the full VAD timeline; transcript-less pairs
    # still represent real start/stop detections from A's session.
    for r in a_speech_ranges:
        events.append({
            "t": r["t"],
            "t_end": r["t_end"],
            "kind": "agent_a_heard",
            "text": r.get("text") or "",
        })

    # Pass 2: emit events. Pop the front of the per-id queue for each
    # Injected line so the Nth injection gets the Nth inject_start time.
    for ts, body in iter_log_lines(path):
        if ts is None:
            continue
        t = round(ts - anchor, 3)
        # Interrupt flag — emitted by clipPlayer_adapter when the agent's
        # RMS VAD on orchSink fires within INTERRUPT_FLAG_WINDOW_MS of an
        # inject. Sole source of unintended_interrupt events.
        m = re.search(
            r"\[ClipPlayer\] interrupt_flag clip_id=(\S+) delta_ms=(-?\d+)",
            body,
        )
        if m:
            events.append({
                "t": t,
                "kind": "unintended_interrupt",
                "clip_id": m.group(1),
                "delta_ms": int(m.group(2)),
            })
            continue
        # Drop any legacy inline-detector log lines so older runs that
        # still carry them don't double-render alongside the new flag.
        if re.search(r"\[ClipPlayer\] UNINTENDED_INTERRUPT clip_id=", body):
            continue
        # Injection
        m = re.search(
            r"\[ClipPlayer\] Injected (\S+) \((\d+) bytes @ (\d+)Hz\)",
            body,
        )
        if m:
            node_id = m.group(1)
            n_bytes = int(m.group(2))
            sr = int(m.group(3)) or 1
            # The clipPlayer pushes int16 mono PCM to the WebRTC source.
            # `bytes / 2 / sr` is the TRUE played duration at eval time.
            # Manifest's audio_end - audio_start can drift from this
            # whenever the clip wav was regenerated after the eval (the
            # refiner narrows the window in place; the older eval played
            # the wider wav). Prefer this over source_end-source_start
            # when sizing the timeline bar.
            played_dur_s = round(n_bytes / 2.0 / sr, 3)
            q = inject_start_queue.get(node_id)
            inj_ts = q.pop(0) if q else ts
            events.append({
                "t": round(inj_ts - anchor, 3),
                "kind": "customer_clip_injected",
                "node_id": node_id,
                "played_dur_s": played_dur_s,
            })
            continue
        # Legacy `Injected` lines without the bytes/sr metadata (very
        # old runs). Fall back to node-id-only capture.
        m = re.search(r"\[ClipPlayer\] Injected (\S+) \(", body)
        if m:
            node_id = m.group(1)
            q = inject_start_queue.get(node_id)
            inj_ts = q.pop(0) if q else ts
            events.append({
                "t": round(inj_ts - anchor, 3),
                "kind": "customer_clip_injected",
                "node_id": node_id,
            })
            continue
        # Routing (legacy select_clip)
        m = re.search(r"\[ClipPlayer\] select_clip → \"([^\"]+)\" \| (.*)$", body)
        if m:
            events.append({
                "t": t, "kind": "routing", "decision": "select_clip",
                "node_id": m.group(1), "reasoning": m.group(2),
            })
            continue
        # Routing (legacy select_fallback)
        m = re.search(r"\[ClipPlayer\] select_fallback → (?:fallback_clip \"([^\"]+)\" on \"([^\"]+)\"|\"([^\"]+)\") \| (.*)$", body)
        if m:
            node_id = m.group(2) or m.group(3)
            events.append({
                "t": t, "kind": "routing", "decision": "select_fallback",
                "node_id": node_id, "reasoning": m.group(4),
            })
            continue
        # Routing (new `play(...)` format). The choice is one of "next_item"
        # (formerly "advance"), "yes", "no", "fallback", or a specific
        # clip_id. We extract the resulting node_id (where available) so
        # the reasoning can be attached to the next customer_speech row.
        # next_item: advance-to-next variant
        m = re.search(r'\[ClipPlayer\] play\((?:next_item|advance)\) → s\d+ i\d+ (\S+)\s*\|\s*(.*)$', body)
        if m:
            events.append({
                "t": t, "kind": "routing", "decision": "play_next_item",
                "node_id": m.group(1), "reasoning": m.group(2),
            })
            continue
        # initial-turn auto-play (no model decision; fired by doInitialTurn)
        m = re.search(r'\[ClipPlayer\] play\(initial\) → s\d+ i\d+ (\S+)\s*\|\s*(.*)$', body)
        if m:
            events.append({
                "t": t, "kind": "routing", "decision": "play_initial",
                "node_id": m.group(1), "reasoning": m.group(2),
            })
            continue
        # next_item: auto-correction variant — adapter auto-plays a queued
        # _corr clip when next_item is called and a correction is pending
        m = re.search(r'\[ClipPlayer\] play\((?:next_item|advance)\) → auto-correction (\S+)\s*\|\s*(.*)$', body)
        if m:
            events.append({
                "t": t, "kind": "routing", "decision": "play_next_item_corr",
                "node_id": m.group(1), "reasoning": m.group(2),
            })
            continue
        # next_item: conversation_ended (no clip to play)
        m = re.search(r'\[ClipPlayer\] play\((?:next_item|advance)\) — no more items[^|]*\|\s*(.*)$', body)
        if m:
            events.append({
                "t": t, "kind": "routing", "decision": "play_next_item_end",
                "node_id": None, "reasoning": m.group(1),
            })
            continue
        m = re.search(r'\[ClipPlayer\] play\((yes|no)\)[^(]*\((\S+)\)\s*\|\s*(.*)$', body)
        if m:
            events.append({
                "t": t, "kind": "routing", "decision": f"play_{m.group(1)}",
                "node_id": m.group(2), "reasoning": m.group(3),
            })
            continue
        m = re.search(r'\[ClipPlayer\] play\(fallback\)[^|]*replaying (\S+)[^|]*\|\s*(.*)$', body)
        if m:
            events.append({
                "t": t, "kind": "routing", "decision": "play_fallback",
                "node_id": m.group(1), "reasoning": m.group(2),
            })
            continue
        # `play("<clip_id>") | reasoning` — allow an optional trailing
        # annotation between the `)` and the `|` (e.g.
        # `play("i0_main") — compound replay swapped to sub_i0 (…) | reason`)
        # emitted when the adapter swaps a compound main pick for a
        # per-item sub-clip. Without the wildcard tolerance, the swap
        # events wouldn't be captured as routing rows and the sub-clip
        # customer_speech would fall back to pairing with an older
        # routing event (typically the initial-turn auto-play).
        m = re.search(r'\[ClipPlayer\] play\("([^"]+)"\)[^|]*\|\s*(.*)$', body)
        if m:
            events.append({
                "t": t, "kind": "routing", "decision": "play_clip",
                "node_id": m.group(1), "reasoning": m.group(2),
            })
            continue
        # Refusals / rejections (state didn't advance).
        m = re.search(r'\[ClipPlayer\] (REFUSED|REJECTED) [^|]*?\|\s*(.*)$', body)
        if m:
            events.append({
                "t": t, "kind": "routing", "decision": m.group(1).lower(),
                "node_id": None, "reasoning": m.group(2),
            })
            continue
        # Tool-response mirror. The adapter emits the exact payload it
        # returned to the model after every clipPlayer pick (line tag
        # added in _emitFunctionCallOutput). Attach it to the most
        # recent routing event so the dashboard can render it on hover.
        m = re.search(r"\[ClipPlayer:tool_response\]\s+(\{.*\})\s*$", body)
        if m:
            try:
                payload = json.loads(m.group(1))
            except Exception:
                payload = None
            if payload is not None:
                # Walk backwards for the nearest routing event; attach the
                # payload as `tool_response` (the routing event has the
                # decision/node_id, and the payload is what the model
                # actually saw in response).
                for prev in reversed(events):
                    if prev.get("kind") == "routing":
                        prev["tool_response"] = payload
                        break
            continue
        # Ack ingestion trace. Emitted by the adapter every time the
        # router reports a non-empty `acknowledged` array on a play(...)
        # call. Attaches to the routing event that FOLLOWS (the pick
        # that piggybacks the ack) — parser can't tell the two apart
        # without more context, so we buffer and let the next routing
        # event claim it via a scratch attribute picked up in the merge
        # pass.
        m = re.search(
            r"\[ClipPlayer:acks\]\s+accepted=(\[.*?\])\s+rejected=(\[.*?\])\s+unacked_after=(\[.*?\])\s*$",
            body,
        )
        if m:
            try:
                acks = {
                    "t": t,
                    "kind": "acks",
                    "accepted": json.loads(m.group(1)),
                    "rejected": json.loads(m.group(2)),
                    "unacked_after": json.loads(m.group(3)),
                }
                events.append(acks)
            except Exception:
                pass
            continue
        # Premature-close substitution. Emitted when the router picked
        # thats_all while unacked > 0 and the adapter swapped in a
        # restatement/main clip. Surface as a distinct event so the
        # dashboard can flag the substitution in the timeline.
        m = re.search(
            r"\[ClipPlayer:premature_close\]\s+thats_all\s+→\s+(\S+)\s+\(unacked=(\[.*?\])\)\s*$",
            body,
        )
        if m:
            try:
                unacked = json.loads(m.group(2))
            except Exception:
                unacked = []
            events.append({
                "t": t,
                "kind": "premature_close",
                "substituted_node_id": m.group(1),
                "unacked": unacked,
            })
            continue
    return events


def _parse_b_log_contract(path: Path, anchor: float) -> list[dict]:
    """Agent-speech rows for non-OpenAI (Pipecat contract) runs.

    OpenAI-native runs log the realtime API's own events (parsed by
    parse_b_log_events below). Other providers — Gemini today, xai/ultravox
    later — don't speak that dialect, so the Pipecat adapter's _StdoutEvents
    instead emits `[roleB-viz]` markers (agent_start / agent_transcript /
    agent_stop) with ISO-UTC timestamps on the same wall-clock as the native
    logs. We pair start→stop into spans and attach the turn's transcript.

    Only the AGENT side is built here; the customer side comes from
    adapter_A.log (parse_clip_player_events), identical for every provider
    because Role A is always the OpenAI clip player.

    Pairing is ORDINAL (turn N's text ↔ turn N's speech span), not time-window:
    the agent_transcript marker (LLMFullResponseEnd) can fire just AFTER agent_stop
    on some providers (Gemini), so attaching text within the [start,stop] window
    drops it and shifts every later turn by one (the symptom: first turn shows
    'transcript missing' and each line lands on the wrong turn). Collecting spans
    and transcripts separately and zipping by index is robust to that skew."""
    # Build one record per audio span. Each span owns its OWN chunk list,
    # so transcripts get bound to their containing span at emission time
    # (not via a fragile post-hoc ordinal zip). This matters when a span
    # gets phantom-merged into its neighbor (see below): merging two
    # spans also concatenates their chunks, so a continuous utterance
    # split across two agent_start/agent_stop pairs reassembles into one
    # row with the full text intact.
    spans: list[dict] = []  # each: {"start": float, "stop": float|None, "chunks": list[str]}
    open_span: dict | None = None
    last_ts: float | None = None
    # Tracks whether an `agent_start` (the audio-side event) has fired
    # since the most recent `agent_transcript`. Cascaded providers can
    # finalize an LLM response (emit `agent_transcript`) and then
    # SUPERSEDE it with a fresh LLM call before TTS ever plays a single
    # frame — leaving two transcripts within ONE span where the second
    # replaces the first. Without this flag the merge below would
    # concatenate the superseded text alongside the real text (symptom
    # on Ember_Oak.g3.s0: span carries "Sure! Would you like cheese... "
    # prepended to the actual line). Initial True so the first
    # transcript appends; thereafter only the LATEST transcript in each
    # (start→next-start) window survives within a single span.
    agent_start_since_transcript: bool = True
    # Pending transcript queued ahead of its own span: the cascaded provider
    # (OpenAI STT → LLM → TTS) finalizes the LLM response BEFORE TTS begins
    # playback, so `agent_transcript` arrives ~1s ahead of the matching
    # `agent_start`. Without buffering it here the parser would attach this
    # text to the PREVIOUS span (or drop it entirely on the first utterance,
    # because spans=[]), shifting every line by one — the symptom: turn 0
    # shows turn 1's text, the greeting transcript disappears, etc.
    # Supersede semantics are preserved: a second transcript arriving while
    # `pending_transcript` is still set replaces the first (the LLM
    # re-finalized before TTS ever started).
    pending_transcript: str | None = None
    # B's VAD perception of customer audio. Mirrors the OpenAI native
    # adapter's customer_speech rows derived from
    # input_audio_buffer.speech_started/stopped, so the dashboard renders
    # the same VAD bars for Gemini that it does for OpenAI runs.
    cust_open: float | None = None
    customer_rows: list[dict] = []
    for ts, body in iter_log_lines(path):
        if ts is None or "[roleB-viz]" not in body:
            continue
        last_ts = ts
        if "agent_start" in body:
            open_span = {"start": ts, "stop": None, "raw_stop": None, "chunks": []}
            if pending_transcript:
                open_span["chunks"].append(pending_transcript)
                pending_transcript = None
                # Pending text counts as this span's "first transcript", so
                # any further transcript that arrives while the span is open
                # should SUPERSEDE it (not append) — same semantics as the
                # within-span supersede case below.
                agent_start_since_transcript = False
            else:
                agent_start_since_transcript = True
            spans.append(open_span)
        elif "agent_stop" in body:
            # Prefer the cumulative-audio-bytes duration the Pipecat
            # adapter emits (`audio_duration_s`) over the marker's
            # wall-clock timestamp. Pipecat's BotStoppedSpeakingFrame
            # fires when TTS SYNTHESIS finishes — streaming TTS produces
            # samples faster than realtime, so several seconds of audio
            # can still be queued for playback when the marker fires.
            # Counting OutputAudioRawFrame.num_frames in the adapter and
            # dividing by sample_rate gives the actual span length.
            audio_dur = None
            m = re.search(r"\{.*\}\s*$", body)
            if m:
                try:
                    payload = json.loads(m.group(0))
                    v = payload.get("audio_duration_s")
                    if isinstance(v, (int, float)) and v > 0:
                        audio_dur = float(v)
                except Exception:
                    pass
            if open_span is not None:
                start = open_span["start"]
                open_span["stop"] = (
                    (start + audio_dur) if audio_dur is not None else ts
                )
                # The marker's own wall-clock time, kept separately from the
                # audio-duration-projected `stop` above. The phantom-merge
                # gap check below needs this: two REAL back-to-back
                # utterances routinely have a first span whose projected
                # `stop` (start + audio_duration_s) lands slightly AFTER the
                # second span's real `agent_start`, because streaming TTS
                # queues audio faster than realtime. Measuring the merge gap
                # against the projected stop turns that ordinary overlap
                # into a large *negative* gap, which is well under the 0.25s
                # phantom threshold and wrongly merges two distinct turns.
                open_span["raw_stop"] = ts
                open_span = None
        elif "agent_transcript" in body:
            m = re.search(r"\{.*\}\s*$", body)
            if m:
                try:
                    txt = (json.loads(m.group(0)).get("text") or "").strip()
                    if txt:
                        # Routing rules for which span this text belongs to:
                        #   1. A span is currently open (OpenAI Realtime
                        #      pattern: start → transcript → stop) → attach
                        #      to the open span.
                        #   2. No open span, but the last-closed span has
                        #      no transcript yet (Pipecat/Gemini pattern:
                        #      start → stop → transcript-fires-just-after)
                        #      → attach to that span.
                        #   3. Otherwise (cascaded pattern: transcript
                        #      arrives BEFORE its agent_start; the
                        #      previous span already has its own
                        #      transcript) → queue as pending_transcript
                        #      for the next agent_start to consume.
                        if open_span is not None:
                            chunks = open_span["chunks"]
                            if agent_start_since_transcript or not chunks:
                                chunks.append(txt)
                            else:
                                # Cascaded supersede: a fresh LLM
                                # finalization arrived without a new
                                # agent_start (no TTS played for the
                                # earlier text). Replace, don't append.
                                chunks[-1] = txt
                            agent_start_since_transcript = False
                        elif spans and not spans[-1]["chunks"]:
                            # Post-stop transcript for the just-closed span.
                            spans[-1]["chunks"].append(txt)
                        else:
                            # Pre-fire transcript for the next span.
                            # Overwriting any prior pending text yields the
                            # supersede behavior described above.
                            pending_transcript = txt
                except Exception:
                    pass
        elif "user_started" in body:
            cust_open = ts
        elif "user_stopped" in body and cust_open is not None:
            customer_rows.append({
                "t": round(cust_open - anchor, 3),
                "t_end": round(ts - anchor, 3),
                "kind": "customer_speech",
            })
            cust_open = None
    # An open span never closed by a stop (call cut off mid-turn) → close
    # at the last marker.
    if open_span is not None and last_ts is not None:
        open_span["stop"] = last_ts
        open_span = None
    if cust_open is not None and last_ts is not None:
        customer_rows.append({
            "t": round(cust_open - anchor, 3),
            "t_end": round(last_ts - anchor, 3),
            "kind": "customer_speech",
        })

    # Pipecat's Gemini path emits a phantom stop→start pair in two
    # situations:
    #   (a) Leading-edge phantom: BotStartedSpeakingFrame fires before
    #       the first audio packet, a transient BotStoppedSpeakingFrame
    #       fires when that initial silence elapses, then a real
    #       BotStartedSpeakingFrame fires when audio actually starts.
    #       Two spans for one turn (a ~3ms phantom + the real audio),
    #       one transcript on the real span.
    #   (b) Mid-utterance phantom: Gemini Live finalizes its LLM response
    #       in two text chunks, producing two stop→start pairs even
    #       though the audio plays continuously. Two spans, one
    #       transcript on EACH span — they should reassemble into one
    #       row with the full concatenated text.
    # Coalesce adjacent spans separated by a gap shorter than this
    # threshold AND concatenate their chunks. 0.25s comfortably exceeds
    # any phantom we've observed (~3ms-60ms) and is smaller than the
    # smallest legitimate mid-turn pause.
    if len(spans) > 1:
        PHANTOM_MERGE_S = 0.25
        merged: list[dict] = [spans[0]]
        for sp in spans[1:]:
            prev = merged[-1]
            prev_stop = prev["stop"] if prev["stop"] is not None else prev["start"]
            # Gap test uses the RAW marker timestamp, not the audio-duration-
            # projected stop: a genuine Gemini phantom has near-zero raw gap
            # (the stop/start markers fire back-to-back in real wall-clock
            # time), so this still catches it. Two ordinary sequential
            # utterances instead have a real ~0.5-1s raw gap even when their
            # projected durations overlap, so they no longer look like a
            # negative (and therefore always-qualifying) gap.
            prev_raw_stop = prev["raw_stop"] if prev["raw_stop"] is not None else prev_stop
            if sp["start"] - prev_raw_stop < PHANTOM_MERGE_S:
                # Take the LATER stop (in case the merged-in span runs
                # longer than the leader) and concatenate chunks.
                later_stop = sp["stop"] if sp["stop"] is not None else prev_stop
                prev["stop"] = max(prev_stop, later_stop)
                prev["raw_stop"] = sp["raw_stop"]
                prev["chunks"].extend(sp["chunks"])
            else:
                merged.append(sp)
        spans = merged

    rows: list[dict] = []
    for sp in spans:
        # Join chunks with a single space and dedupe runs of whitespace.
        # Each chunk was stripped on read, so a Gemini Live continuation
        # like "Got it, lamb vindaloo. Would you like that mild" +
        # " or spicy?..." (the second carried a leading space pre-strip)
        # joins cleanly to "Got it, lamb vindaloo. Would you like that
        # mild or spicy?...".
        txt = re.sub(r"\s+", " ", " ".join(c for c in sp["chunks"] if c)).strip()
        start = sp["start"]
        stop = sp["stop"] if sp["stop"] is not None else start
        rows.append({
            "t": round(start - anchor, 3),
            "t_end": round(stop - anchor, 3),
            "kind": "agent_speech",
            "text": txt,
            # Response-length hints used by the dashboard timeline so a
            # reviewer can spot one-word agent turns vs full paragraphs
            # without expanding each row.
            "chars": len(txt),
            "words": len(txt.split()) if txt else 0,
        })
    rows.extend(customer_rows)
    return rows


def parse_b_log_events(path: Path, anchor: float) -> list[dict]:
    """Extract dialog timing + transcripts from adapter_B.log.

    B's session sees BOTH sides of the conversation:
      - input_audio_buffer.speech_started / .speech_stopped → customer speech (from A)
      - output_audio_buffer.started / .stopped → agent's own speech
      - response.output_audio_transcript.done → text of the agent's last turn

    We pair started/stopped events to produce {t_start, t_end} ranges, and
    attach the most recent agent transcript to each agent_speech range.
    """
    if not path.exists():
        return []

    # Non-OpenAI providers (Gemini, and future xai/ultravox) emit the
    # model-agnostic `[roleB-viz]` contract markers instead of native OpenAI
    # realtime events. Detect and branch; the native parser below is unchanged.
    if "[roleB-viz]" in path.read_text(errors="ignore"):
        return _parse_b_log_contract(path, anchor)

    # Customer side: input_audio_buffer.speech_started/stopped pairs.
    customer_started: float | None = None
    customer_started_item_id: str | None = None
    customer_ranges: list[dict] = []
    # B's own transcription of A's input audio (heard via the realtime API's
    # input_audio_transcription model). Indexed by conversation-item id so we
    # can join against the speech ranges that share the same item_id.
    input_transcripts_by_item: dict[str, str] = {}

    # Agent side: emit ONE row per `response.content_part.done` (audio with
    # transcript). This is the most reliable signal — there's exactly one per
    # spoken utterance, and it carries the full transcript text. We also
    # collect reasoning_tokens from response.done by response_id so we can
    # attach them.
    agent_ranges: list[dict] = []
    # Audio actually becomes audible at `output_audio_buffer.started`, not at
    # `content_part.done` (which fires when the model finishes generating —
    # before all bytes have been pushed out to the wire). MULTIPLE
    # buffer.started events can fire before any content_part.done (the
    # model streams audio for several utterances ahead of finishing
    # token generation). Keep them in a FIFO queue and pair each
    # content_part.done with the OLDEST unconsumed buffer.started so
    # every transcribed part lands on the wall-clock time its audio
    # actually started playing — not on the later token-completion time.
    pending_buffer_started_q: list[float] = []
    # Rows attached to the currently-open buffer span. When .stopped fires
    # we'll stamp the real t_end onto each of them — content_part.done is
    # only emitted when the model FINISHES GENERATING tokens; the wire-out
    # audio plays for several more seconds after that, so without this
    # patch downstream "is the agent done speaking?" logic underestimates.
    rows_in_open_span: list[dict] = []
    reasoning_by_resp: dict[str, int] = {}

    # First pass: collect reasoning_tokens by response_id (response.done
    # arrives AFTER content_part.done in the same response, so we need a
    # two-pass approach — or buffer rows and attach after).
    pending_rows_by_resp: dict[str, list[dict]] = {}

    for ts, body in iter_log_lines(path):
        if ts is None:
            continue

        # Customer side
        if "input_audio_buffer.speech_started" in body:
            customer_started = ts
            customer_started_item_id = None
            mm = re.search(r'"item_id"\s*:\s*"([^"]+)"', body)
            if mm:
                customer_started_item_id = mm.group(1)
            continue
        if "input_audio_buffer.speech_stopped" in body and customer_started is not None:
            mm = re.search(r'"item_id"\s*:\s*"([^"]+)"', body)
            stop_item_id = mm.group(1) if mm else customer_started_item_id
            customer_ranges.append({
                "t": round(customer_started - anchor, 3),
                "t_end": round(ts - anchor, 3),
                "kind": "customer_speech",
                "b_item_id": stop_item_id,
            })
            customer_started = None
            customer_started_item_id = None
            continue

        # B's own transcription of A's audio for each user input item.
        # Fires after the input audio has been transcribed via the
        # input_audio_transcription model (enabled in the eval script).
        if "conversation.item.input_audio_transcription.completed" in body:
            m = re.search(r"\{.*\}\s*$", body)
            if m:
                try:
                    evt = json.loads(m.group(0))
                    iid = evt.get("item_id")
                    txt = (evt.get("transcript") or "").strip()
                    if iid and txt:
                        input_transcripts_by_item[iid] = txt
                except Exception:
                    pass
            continue

        # Track when agent audio actually starts playing on the wire.
        # Multiple .started events can queue up before any .stopped fires
        # when the model emits successive utterances back-to-back.
        if "output_audio_buffer.started" in body:
            pending_buffer_started_q.append(ts)
            continue

        # Audio finished playing on the wire — stamp every row that lives
        # inside this span with the real end time.
        if "output_audio_buffer.stopped" in body:
            t_end = round(ts - anchor, 3)
            for r in rows_in_open_span:
                r["t_end"] = t_end
            rows_in_open_span = []
            # Clear any leftover started events that never got paired
            # (defensive — shouldn't happen in normal flow).
            pending_buffer_started_q = []
            continue

        # One row per audio content_part.done with a real transcript
        if "response.content_part.done" in body:
            m = re.search(r"\{.*\}\s*$", body)
            if not m:
                continue
            try:
                evt = json.loads(m.group(0))
            except Exception:
                continue
            part = evt.get("part") or {}
            if part.get("type") != "audio":
                continue
            text = (part.get("transcript") or "").strip()
            if not text:
                continue
            rid = evt.get("response_id")
            # Pair with the OLDEST unconsumed buffer.started in the queue.
            # When the model streams audio for several utterances ahead of
            # the token stream finishing, multiple .started events queue
            # up before the corresponding content_part.done events arrive.
            # Without FIFO pairing, every part collapses to the timestamp
            # of the LATEST .started and the earlier utterances lose
            # their real audio-onset wall time — they end up identical to
            # the next row's timestamp even though they played seconds
            # earlier.
            if pending_buffer_started_q:
                audio_ts = pending_buffer_started_q.pop(0)
            else:
                audio_ts = ts
            row = {
                "t": round(audio_ts - anchor, 3),
                "t_end": round(audio_ts - anchor, 3),
                "kind": "agent_speech",
                "text": text,
            }
            agent_ranges.append(row)
            rows_in_open_span.append(row)
            if rid:
                pending_rows_by_resp.setdefault(rid, []).append(row)
            continue

        # Reasoning tokens from response.done (apply to all rows tied to rid)
        if '"type":"response.done"' in body:
            m = re.search(r"\{.*\}\s*$", body)
            if not m:
                continue
            try:
                evt = json.loads(m.group(0))
            except Exception:
                continue
            resp = (evt.get("response") or {})
            rid = resp.get("id")
            usage = resp.get("usage") or {}
            otd = usage.get("output_token_details") or {}
            r_tok = otd.get("reasoning_tokens")
            if rid and r_tok and rid in pending_rows_by_resp:
                for r in pending_rows_by_resp[rid]:
                    r["reasoning_tokens"] = r_tok

    # Attach B's input transcription to customer_speech rows by item_id.
    for r in customer_ranges:
        iid = r.get("b_item_id")
        if iid and iid in input_transcripts_by_item:
            r["b_heard_text"] = input_transcripts_by_item[iid]

    return customer_ranges + agent_ranges


def parse_tool_calls(tool_log_path: Path, anchor: float) -> list[dict]:
    if not tool_log_path.exists():
        return []
    out = []
    for entry in json.loads(tool_log_path.read_text()):
        t_epoch = (entry.get("t") or 0) / 1000.0
        out.append({
            "t": round(t_epoch - anchor, 3),
            "kind": "tool_call",
            "tool": entry.get("tool"),
            "args": entry.get("args") or {},
            "result_ok": (entry.get("result") or {}).get("ok"),
            "result_error": (entry.get("result") or {}).get("error"),
        })
    return out


def attach_clip_text(events: list[dict], decision_tree_path: Path):
    """Customer-clip-injected events only carry node_id. Look up text + the
    ground-truth item info from the decision tree. Synthetic fallback nodes
    (suffix `__fb`) inherit their parent node's text, tagged as fallback."""
    if not decision_tree_path.exists():
        return
    tree = json.loads(decision_tree_path.read_text())
    nodes = tree.get("nodes") or {}
    # Node IDs synthesized at play time by the adapter's compound-fallback
    # swap: `<base_id>__fb_i<item_index>` (`g1_s1_i0_main__fb_i0`) —
    # matches the sub-clip cut from the compound main's item-N span.
    # Legacy shape `<base_id>__fb` (single fallback_clip / post-revision
    # sub-slice) still supported for pre-sub_clips runs.
    _sub_re = re.compile(r"__fb_i(\d+)$")
    for ev in events:
        if ev.get("kind") != "customer_clip_injected":
            continue
        node_id = ev["node_id"]
        sub_m = _sub_re.search(node_id)
        is_fallback = node_id.endswith("__fb") or sub_m is not None
        if sub_m:
            lookup_id = node_id[: sub_m.start()]
            sub_idx = int(sub_m.group(1))
        elif node_id.endswith("__fb"):
            lookup_id = node_id[:-4]
            sub_idx = None
        else:
            lookup_id = node_id
            sub_idx = None
        node = nodes.get(lookup_id, {})
        sub = None
        if sub_idx is not None:
            for fc in (node.get("fallback_clips") or []):
                if fc.get("item_index") == sub_idx:
                    sub = fc
                    break
        if sub:
            # Per-item sub-clip: use its own transcript + audio filename.
            ev["text"] = sub.get("text") or node.get("text", "")
            ev["clip_filename"] = sub.get("clip") or node.get("clip")
        elif is_fallback and node.get("fallback_text"):
            # Legacy fallback with a narrower post-revision transcript.
            ev["text"] = node["fallback_text"]
            ev["clip_filename"] = node.get("fallback_clip") or node.get("clip")
        elif is_fallback:
            ev["text"] = node.get("text", "")
            ev["clip_filename"] = node.get("fallback_clip") or node.get("clip")
        else:
            ev["text"] = node.get("text", "")
            ev["clip_filename"] = node.get("clip")
        ev["stage"] = node.get("stage")
        ev["is_fallback"] = is_fallback


def audio_duration_s(wav_path: Path) -> float:
    if not wav_path.exists():
        return 0.0
    with wave.open(str(wav_path)) as w:
        return w.getnframes() / w.getframerate()


def load_clip_source_windows(task_id: str) -> dict[str, tuple[float, float]]:
    """Read the per-task clip manifest and build node_id → (source_start, source_end)
    in absolute session-audio time.

    Maps:
      - Direct clips (main/opt/etc.) by clip_id
      - Option sub-clips (`<clip>_opt`) from each parent's option_spans[0]
      - Fallback nodes (`<clip>__fb`) share their parent's source window
        (the fallback only replays from `post_revision_offset_t` onward — we
        use that narrower span when present)
    """
    manifest = ARTIFACTS_PATH / "customer_clips" / f"{task_id}.json"
    if not manifest.exists():
        return {}
    try:
        m = json.loads(manifest.read_text())
    except Exception:
        return {}
    out: dict[str, tuple[float, float]] = {}
    for c in m.get("clips", []):
        cid = c.get("clip_id")
        if not cid:
            continue
        if c.get("audio_start") is not None and c.get("audio_end") is not None:
            t0 = float(c["audio_start"])
            t1 = float(c["audio_end"])
            out[cid] = (t0, t1)
            # Fallback for this node plays from post_revision_offset_t onward
            post_rev = c.get("post_revision_offset_t")
            if post_rev is not None:
                out[f"{cid}__fb"] = (float(post_rev), t1)
            else:
                out[f"{cid}__fb"] = (t0, t1)
        # Option sub-clip is keyed off the parent clip by suffix `_opt`
        opt_spans = c.get("option_spans") or []
        if opt_spans:
            sp = opt_spans[0]
            if sp.get("t_start") is not None and sp.get("t_end") is not None:
                out[f"{cid}_opt"] = (float(sp["t_start"]), float(sp["t_end"]))
    return out


REFERENCE_TRANSCRIPT_PAD_SPANS = 3


def load_reference_transcript(
    task_id: str,
    session_id: str,
    game_index: int,
    source_offset_s: float,
) -> list[dict]:
    """Load the per-game span timeline used for the reference-audio transcript.

    Filters session-level `timeline.json` by `game_index`, sorts by t_start,
    and for each customer-side span, attaches the list of clip_ids that
    cover any part of it (so the UI can show which decision-tree nodes
    a span feeds into and surface cross-clip duplication).

    When `task_id` is a per-stage tid (`…g<gi>.s<si>`), the result is
    further narrowed to the active stage's spans plus a small number
    of padding spans (`REFERENCE_TRANSCRIPT_PAD_SPANS` on each side)
    from the neighbouring stages — enough conversational context for
    cross-stage debugging without spilling the whole game's transcript
    into the active-stage view. Each span carries a `_stage_context`
    field (``"active" | "prev_pad" | "next_pad"``) so the frontend
    can render padding rows distinctly.
    """
    # Per-stage tids carry an `.s<si>` suffix — strip + parse to derive
    # the active stage index. Legacy game-level tids leave this None and
    # the function falls back to the original whole-game behaviour.
    try:
        from clip_eval.pipeline.artifact_store import parse_tid as _parse_tid
        _parsed = _parse_tid(task_id) if task_id else None
    except Exception:
        _parsed = None
    active_stage = _parsed[2] if _parsed else None
    align_path = ARTIFACTS_PATH / "sessions" / "order_up" / session_id / "timeline.json"
    spans: list[dict] = []
    if align_path.exists():
        try:
            d = json.loads(align_path.read_text())
            spans = d.get("deduped_spans") or []
        except Exception:
            spans = []
    game_spans = [s for s in spans if s.get("game_index") == game_index]
    # Decide which stages we need to back-fill from classified_spans.
    # Cases:
    #   1. game_spans is empty (legacy fallback)            → all stages
    #   2. active_stage is set AND timeline.json is missing
    #      one of {active_stage-1, active_stage, active_stage+1}
    #      (common when timeline.json is stale post per-stage
    #      classify)                                        → just the missing
    #      stages that fall in the active±1 window
    stages_in_timeline = {
        s.get("stage_index") for s in game_spans
        if s.get("stage_index") is not None
    }
    need_stages: set[int] | None = None
    if not game_spans:
        need_stages = None  # all stages (legacy whole-game backfill)
    elif active_stage is not None:
        candidate = {active_stage - 1, active_stage, active_stage + 1}
        missing = {s for s in candidate if s >= 0 and s not in stages_in_timeline}
        if missing:
            need_stages = missing
    if need_stages is not None or not game_spans:
        # Fall back to the per-stage source of truth:
        # classified_spans/g{game}_s*.json, which is always written by the
        # per-stage build. We synthesize the timeline payload locally so
        # the reference-transcript pane still renders.
        cls_dir = ARTIFACTS_PATH / "sessions" / "order_up" / session_id / "classified_spans"
        if cls_dir.exists():
            prefix = f"g{game_index:02d}_s"
            for cls_path in sorted(cls_dir.glob(f"{prefix}*.json")):
                try:
                    cls = json.loads(cls_path.read_text()) or {}
                except Exception:
                    continue
                stage_meta = cls.get("_stage") or {}
                si = stage_meta.get("stage_index")
                if si is None:
                    try:
                        si = int(cls_path.stem.split("_s")[-1])
                    except Exception:
                        continue
                # When need_stages is None we're doing a full-game backfill
                # (empty timeline); otherwise restrict to just the missing
                # stages so we don't re-add spans timeline.json already
                # has (the dedupe step would catch them anyway, but skipping
                # the per-file parse is cheaper and avoids edge-case
                # mismatches between timeline.deduped_spans and the raw
                # classified_spans they were derived from).
                if need_stages is not None and si not in need_stages:
                    continue
                for sp in cls.get("spans") or []:
                    s = dict(sp)
                    s["game_index"] = game_index
                    s["stage_index"] = si
                    if "stage_type" not in s:
                        s["stage_type"] = stage_meta.get("stage_type")
                    if "restaurant" not in s:
                        s["restaurant"] = stage_meta.get("restaurant")
                    # Only this-stage spans belong in the reference transcript;
                    # mirrors the filter dedupe_and_rollup applies.
                    if s.get("stage_attribution") in (None, "this"):
                        game_spans.append(s)
    # Annotate each span with the index it occupies inside its own
    # `classified_spans/g<g>_s<s>.json` file. Used by the raw-span edit
    # endpoints — the frontend sends back the original index so the server
    # can patch the right entry. Matching is on (t_start, t_end, label, text).
    classified_dir = (
        ARTIFACTS_PATH / "sessions" / "order_up" / session_id / "classified_spans"
    )
    raw_by_stage: dict[int, list[dict]] = {}
    for s in game_spans:
        si = s.get("stage_index")
        if si is None or si in raw_by_stage:
            continue
        path = classified_dir / f"g{game_index:02d}_s{si:02d}.json"
        if not path.exists():
            continue
        try:
            raw_by_stage[si] = (json.loads(path.read_text()) or {}).get("spans") or []
        except Exception:
            raw_by_stage[si] = []
    def _norm(t: str) -> str:
        # Whitespace + case + leading/trailing punctuation collapse —
        # spans can pick up trailing periods or doubled spaces along the
        # way that shouldn't break the editor.
        s = " ".join((t or "").lower().split())
        return s.strip(" .,!?;:\"'")

    for s in game_spans:
        si = s.get("stage_index")
        raw_list = raw_by_stage.get(si) or []
        s_t0 = round(float(s.get("t_start") or 0), 3)
        s_t1 = round(float(s.get("t_end") or 0), 3)
        s_label = s.get("label") or ""
        s_text = (s.get("text") or "").strip()
        s_text_norm = _norm(s_text)
        # Strict 4-tuple first (cheap; covers most cases).
        idx = -1
        for i, rs in enumerate(raw_list):
            if (
                round(float(rs.get("t_start") or 0), 3) == s_t0
                and round(float(rs.get("t_end") or 0), 3) == s_t1
                and (rs.get("label") or "") == s_label
                and (rs.get("text") or "").strip() == s_text
            ):
                idx = i
                break
        # Relaxed fallback: same label + normalized-text equal + any
        # time overlap. Handles drift from a partial re-classify (e.g.
        # `_clamp_padding_to_neighbors` shifting t_start by a couple
        # hundred ms while timeline.json still carries the old value).
        if idx < 0:
            best_overlap = 0.0
            for i, rs in enumerate(raw_list):
                if (rs.get("label") or "") != s_label:
                    continue
                if _norm((rs.get("text") or "").strip()) != s_text_norm:
                    continue
                r_t0 = round(float(rs.get("t_start") or 0), 3)
                r_t1 = round(float(rs.get("t_end") or 0), 3)
                overlap = min(s_t1, r_t1) - max(s_t0, r_t0)
                if overlap > best_overlap:
                    best_overlap = overlap
                    idx = i
        s["__raw_span_index"] = idx
        if idx >= 0:
            rs = raw_list[idx]
            s["__edited_by"] = rs.get("edited_by")
            s["__edited_at"] = rs.get("edited_at")
    game_spans.sort(key=lambda s: (s.get("t_start") or 0, s.get("stage_index") or 0))
    # The session-level dedupe in dedupe_and_rollup.py only filters by
    # stage_attribution=="this", so spans that sit between two stages can
    # be claimed by BOTH adjacent stage classifications and end up twice
    # in deduped_spans (e.g. "What?" and "OK, you're good." between
    # stage 2's last action and stage 3's first). Collapse those here by
    # (t_start, t_end, label, text) — keep the earliest stage_index.
    seen: set[tuple] = set()
    deduped: list[dict] = []
    for s in game_spans:
        key = (
            round(float(s.get("t_start") or 0), 3),
            round(float(s.get("t_end") or 0), 3),
            s.get("label") or "",
            (s.get("text") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    game_spans = deduped

    # Per-stage filter + padding. Game-level tids (active_stage is None)
    # keep the whole-game view. Per-stage tids narrow to the active stage
    # plus REFERENCE_TRANSCRIPT_PAD_SPANS rows from each neighbour stage
    # — the LAST N rows of the previous stage (the conversational tail
    # the operator usually wants for context, e.g. the agent's closing
    # ack) and the FIRST N rows of the next stage (the customer's
    # opening line of the next item). game_spans is already sorted by
    # (t_start, stage_index) at this point.
    if active_stage is not None:
        active_rows: list[dict] = []
        prev_rows: list[dict] = []
        next_rows: list[dict] = []
        for s in game_spans:
            si = s.get("stage_index")
            if si == active_stage:
                s["_stage_context"] = "active"
                active_rows.append(s)
            elif si == active_stage - 1:
                prev_rows.append(s)
            elif si == active_stage + 1:
                next_rows.append(s)
            # Stages farther than ±1 away are dropped.
        # Keep the LAST N spans of the previous stage (closest to the
        # active stage's boundary) and the FIRST N of the next.
        prev_pad = prev_rows[-REFERENCE_TRANSCRIPT_PAD_SPANS:] if prev_rows else []
        next_pad = next_rows[:REFERENCE_TRANSCRIPT_PAD_SPANS] if next_rows else []
        for s in prev_pad:
            s["_stage_context"] = "prev_pad"
        for s in next_pad:
            s["_stage_context"] = "next_pad"
        game_spans = prev_pad + active_rows + next_pad

    manifest_path = (
        ARTIFACTS_PATH / "customer_clips" / f"{task_id}.json"
    )
    clips: list[dict] = []
    if manifest_path.exists():
        try:
            mm = json.loads(manifest_path.read_text())
            clips = mm.get("clips") or []
        except Exception:
            clips = []

    def _pre_merge_clip_id_for_span(sp: dict, clip_kind: str | None) -> str | None:
        """Chip label for a classified span in the reference-transcript
        view. The ref-row already shows the span's label badge
        (customer_order / customer_repair / customer_response) next to
        the timestamp, so the chip needs to carry DIFFERENT info — the
        manifest clip identity the span landed in. Uses the
        per-span (stage, item) plus the MANIFEST clip's kind so:
          - Cross-item natural-breath merges still show the span's
            ORIGINAL item identity (e.g. a panna-cotta span absorbed
            into pad-thai's main shows as g{gi}_s{si}_i1_main, not
            the post-merge i0_main).
          - Corr clips show their full kind (opt1_corr, not just
            corr) so it's clear which prior clip is being corrected.
        Falls back to None when the span has no enumeration info."""
        if not clip_kind:
            return None
        si = sp.get("stage_index")
        items = sp.get("item_indices") or []
        if si is None or not items:
            return None
        return f"g{game_index}_s{si}_i{items[0]}_{clip_kind}"

    out: list[dict] = []
    for sp in game_spans:
        t0 = float(sp.get("t_start") or 0)
        t1 = float(sp.get("t_end") or t0)
        label = sp.get("label") or ""
        clip_refs: list[dict] = []
        if label.startswith("customer_"):
            # Match a clip by overlap on [audio_start, audio_end] AND
            # presence of an entry in customer_spans whose label matches
            # and whose [t_start, t_end] overlaps the global span. The
            # customer_spans entries are post-split / per-item, so a single
            # global span may map to multiple clips (e.g. an order that
            # mentions two items gets split into one clip per item).
            span_stage = sp.get("stage_index")
            for c in clips:
                cas = c.get("audio_start")
                cae = c.get("audio_end")
                if cas is None or cae is None:
                    continue
                if cae <= t0 or cas >= t1:
                    continue
                # Stage gate: never associate a clip from a different
                # stage with this span. Adjacent stages can have clips
                # whose audio_end / audio_start fall within a few tens
                # of milliseconds of a neighboring stage's customer
                # span (boundary jitter from the diarizer / merger),
                # which used to produce spurious clip_refs — e.g. a
                # stage-0 "Chicken parmesan." restatement clip showing
                # up as a clip_ref on a stage-1 "Pad Thai." span just
                # because the chicken-parm clip's end at 26.54 s
                # overlapped Pad Thai's start at 26.48 s by 60 ms.
                if c.get("stage_index") is not None and span_stage is not None:
                    if c.get("stage_index") != span_stage:
                        continue
                covered_span: dict | None = None
                for csp in c.get("customer_spans") or []:
                    if csp.get("label") != label:
                        continue
                    cs0 = csp.get("t_start")
                    cs1 = csp.get("t_end")
                    if cs0 is None or cs1 is None:
                        continue
                    if cs1 <= t0 or cs0 >= t1:
                        continue
                    # Require the per-clip span to overlap the global
                    # span by at least 50% of the shorter window —
                    # avoids spurious chips when two adjacent clips'
                    # boundaries touch by a few hundred ms (e.g. item 1's
                    # restatement2 ending 240ms after item 0's opt2
                    # starts, leaving the item 0 span chipped with item 1's
                    # restatement2 clip id).
                    ov = max(0.0, min(cs1, t1) - max(cs0, t0))
                    short = min(cs1 - cs0, t1 - t0)
                    if short > 0 and (ov / short) < 0.5:
                        continue
                    covered_span = csp
                    break
                if covered_span is not None:
                    # Per-clip portion of the global span — used by the UI to
                    # highlight the text fragment that maps into THIS clip
                    # when the user hovers its chip. Clamp to the global
                    # span window so a tiny pad doesn't poke outside.
                    pcs = max(t0, float(covered_span.get("t_start") or t0))
                    pce = min(t1, float(covered_span.get("t_end") or t1))
                    # Prefer the clip's `transcript` (which after the
                    # prompt-biased re-transcription pass reflects what was
                    # actually said in the per-clip audio window) over the
                    # classifier's customer_span text (which echoes a shorter,
                    # error-prone whisper segment — e.g. "Thai and some"
                    # instead of "Italian sub"). The clip's full transcript
                    # is what the word-LCS matcher in the UI hover-highlight
                    # uses to find the matching fragment inside the global
                    # span's text.
                    pclip_text = (c.get("transcript") or "").strip()
                    if not pclip_text:
                        pclip_text = (covered_span.get("text") or "").strip()
                    clip_refs.append({
                        "clip_id": c.get("clip_id"),
                        # Pre-merge synthetic identity — what this span
                        # WOULD have been called before the cross-item
                        # main→main merge collapsed it into a sibling's
                        # clip_id. Used by the UI as the chip label so
                        # reviewers see the original per-item identity
                        # rather than the merge-survivor's. `clip_id`
                        # above is still the manifest's real clip_id and
                        # is used for pin/edit operations.
                        "display_clip_id": _pre_merge_clip_id_for_span(sp, c.get("clip_kind")) or c.get("clip_id"),
                        "stage_index": c.get("stage_index"),
                        "item_index": c.get("item_index"),
                        "clip_kind": c.get("clip_kind"),
                        "span_t_start": round(pcs, 3),
                        "span_t_end": round(pce, 3),
                        "span_text": pclip_text,
                        # Full clip audio window — the UI uses this when a
                        # reference-transcript chip is clicked to pin the
                        # edit view onto this clip (so the user can adjust
                        # boundaries / preview / save without first having
                        # to play the run audio to that clip).
                        "clip_audio_start": round(float(cas), 3),
                        "clip_audio_end": round(float(cae), 3),
                    })
        out.append({
            "t_start": round(t0, 3),
            "t_end": round(t1, 3),
            "slice_t_start": round(t0 - source_offset_s, 3),
            "slice_t_end": round(t1 - source_offset_s, 3),
            "label": label,
            "text": sp.get("text") or "",
            "stage_index": sp.get("stage_index"),
            "stage_attribution": sp.get("stage_attribution"),
            "item_indices": sp.get("item_indices") or [],
            "clip_role": sp.get("clip_role"),
            "replaces_clip_kind": sp.get("replaces_clip_kind"),
            "clip_refs": clip_refs,
            "raw_span_index": sp.get("__raw_span_index", -1),
            "edited_by": sp.get("__edited_by"),
            "edited_at": sp.get("__edited_at"),
            # "active" | "prev_pad" | "next_pad" — set only when the
            # tid is per-stage (`…g<gi>.s<si>`). Game-level tids leave
            # this unset and the UI treats every row as active.
            "stage_context": sp.get("_stage_context"),
        })
    return out


def load_game_audio_window(session_id: str, game_index: int) -> tuple[float, float] | None:
    """Pull `(audio_start, audio_end)` for the game from stage_alignment.json.
    The dashboard's /source_audio endpoint slices the session audio to
    `[max(0, audio_start - 0.5), audio_end + 0.5]`, so the slice's start
    offset in session time is `max(0, audio_start - 0.5)`.
    """
    align_path = ARTIFACTS_PATH / "sessions" / "order_up" / session_id / "stage_alignment.json"
    if not align_path.exists():
        return None
    try:
        align = json.loads(align_path.read_text())
        g = align["games"][game_index]
        return float(g["audio_start"]), float(g["audio_end"])
    except Exception:
        return None


def _load_audit_sidecars(record_dir: Path) -> dict:
    """Read the three post-eval audit sidecars (when present) and return
    a compact summary suitable for the run-view UI's audit panel. Each
    key is None when the corresponding audit didn't run for this record
    (the eval driver fires each audit conditionally based on which cart
    tag is set).

    - off_policy: judged + count of off-policy asks + the per-ask list
      (item_id, group_name, agent_quote, turn_t, rationale).
    - preemptive: judged + boolean verdict + timing evidence.
    - missing_item_root_cause: judged + any_not_detected + per-item rows.
    """
    out: dict = {
        "off_policy": None,
        "preemptive": None,
        "missing_item_root_cause": None,
    }
    op_path = record_dir / "off_policy_audit.json"
    if op_path.exists():
        try:
            d = json.loads(op_path.read_text())
            out["off_policy"] = {
                "judged": bool(d.get("judged")),
                "asks": d.get("asks") or [],
                "off_policy": d.get("off_policy") or [],
            }
        except Exception:
            pass
    pt_path = record_dir / "preemptive_tool_audit.json"
    if pt_path.exists():
        try:
            d = json.loads(pt_path.read_text())
            out["preemptive"] = {
                "judged": bool(d.get("judged")),
                "preemptive": bool(d.get("preemptive")),
                "vad_before_clip": d.get("vad_before_clip"),
                "tool_before_clip_end": d.get("tool_before_clip_end"),
                "first_b_vad_t": d.get("first_b_vad_t"),
                "first_clip_t": d.get("first_clip_t"),
                "first_clip_t_end": d.get("first_clip_t_end"),
                "first_tool_call_t": d.get("first_tool_call_t"),
                # LLM intent classifier output (one entry per extra item).
                "items": d.get("items") or [],
                "model": d.get("model"),
            }
        except Exception:
            pass
    mr_path = record_dir / "missing_item_root_cause.json"
    if mr_path.exists():
        try:
            d = json.loads(mr_path.read_text())
            out["missing_item_root_cause"] = {
                "judged": bool(d.get("judged")),
                "any_not_detected": bool(d.get("any_not_detected")),
                "items": d.get("items") or [],
            }
        except Exception:
            pass
    return out


def build(record_dir: Path) -> dict:
    score_path = record_dir / "score.json"
    score = json.loads(score_path.read_text()) if score_path.exists() else {}
    final_cart = (
        json.loads((record_dir / "final_cart.json").read_text())
        if (record_dir / "final_cart.json").exists() else {}
    )
    task_id = score.get("task_id") or final_cart.get("task_id")

    # ── ANCHOR (wav-timeline t=0 ↔ wall-clock) ──────────────────────────
    # Pick the wall-clock instant that corresponds to wav-file position
    # zero. This is what every event's `t` is measured relative to:
    #   t = event_wall_clock - anchor
    # so an event at t=12.3 means "at wav-file position 12.3s" which the
    # dashboard's audio player will be at after 12.3 seconds of playback.
    #
    # Primary source: the orchestrator's `[Orch] sink(A)#1 bytesIn=...`
    # log line. That line fires inside `pc.ontrack` for role A's first
    # incoming audio chunk — i.e., the orchestrator just received its
    # first audio frame from A's WebRTC track. The wav writer for A.wav
    # is already open by then, and that first byte gets written to A.wav
    # position 0. So this ISO timestamp is the closest we have to "the
    # wall clock when A.wav position 0 was recorded."
    #
    # WHY NOT audio-content scanning? Tempting to find the first
    # non-silent sample in A.wav and call that "position 0 reference."
    # We don't: A.wav contains background noise/comfort audio from the
    # orchestrator's silence padding, so "first non-silent" is ambiguous;
    # cross-correlation against the first clip is more reliable but
    # introduces its own biases (clip ramp-up, jitter). The log-derived
    # anchor is deterministic and machine-readable.
    anchor = recording_start_iso(record_dir / "orchestrator.log")
    if anchor is None:
        # Last-ditch fallback for old runs whose orchestrator log lacks
        # ISO timestamps. Use earliest adapter-log ISO.
        a_anchor = first_iso_in_log(record_dir / "adapter_A.log")
        b_anchor = first_iso_in_log(record_dir / "adapter_B.log")
        candidates = [t for t in (a_anchor, b_anchor) if t is not None]
        if not candidates:
            raise SystemExit(f"No ISO timestamps in adapter logs of {record_dir}")
        anchor = min(candidates)

    # ── A-SIDE ANCHOR CALIBRATION ──────────────────────────────────────
    # Empirically: A.wav's audio content is ~0.8s "behind" where the
    # sink(A)#1 anchor predicts. That is, when adapter_A.log says
    # `inject_start g8_s0_i0_main` at wall T, the audio for that clip
    # actually appears in A.wav at position (T - sink(A)#1_wall) + 0.8s.
    # Root cause: the gap between when sink(A)#1 fires (= first audio
    # frame received from A's WebRTC) and when the wav writer's first
    # emitted byte actually lands at wav position 0. The exact split
    # between writer-stream buffering and WebRTC negotiation startup is
    # not pinned down, but it's STABLE per run.
    #
    # Calibration: cross-correlate the FIRST injected clip's source wav
    # against A.wav near where the log claims it should be. Measure
    # `delta = expected_pos - actual_pos`. Apply `anchor += delta` so
    # `t = wall - anchor` puts every event at its true wav position.
    #
    # This delta is uniform across ALL events that derive their `t` from
    # wall - anchor: A's clip injects, B's data-channel events, tool
    # calls, etc. So shifting the anchor globally is correct — the
    # underlying wav-vs-wall offset is a property of the
    # WebRTC+writer pipeline, not of any particular role.
    #
    # CAVEAT: older runs can have B-side drift superimposed on
    # this constant. B's data-channel events were still wall-anchored
    # correctly, but B's AUDIO drifted progressively LATER in B.wav due
    # to OpenAI vs local clock skew accumulating in B's downlinkBuf.
    # That meant after calibration, A's events landed perfectly but B's
    # events appeared progressively before B's audio. With the
    # the constant offset. Current transports resample and pace provider
    # output so B-side residual error remains bounded.
    inject_offset_s = 0.0
    first_inject_iso_val = first_inject_iso(record_dir / "adapter_A.log")
    if first_inject_iso_val is not None and task_id:
        tree_path = ARTIFACTS_PATH / "decision_trees" / f"{task_id}.json"
        if tree_path.exists():
            try:
                tree = json.loads(tree_path.read_text())
                nodes = tree.get("nodes") or {}
                clips_base = tree.get("clips_base_dir")
                initial_nid = tree.get("initial_node")
                node = nodes.get(initial_nid, {})
                clip_basename = node.get("clip")
                if clip_basename and clips_base:
                    clip_path = REPO_ROOT / clips_base / clip_basename
                    expected_t = first_inject_iso_val - anchor
                    # search_pad_s=12 covers the observed worst-case ~5.6s
                    # pacer-warmup offset (orchestrator buffers up to ~5s
                    # of audio before sink(A)#1 fires) with headroom; cap
                    # is similarly widened. Cross-correlation quality
                    # score (≥0.3) inside calibrate_inject_offset_s is
                    # the real guard against bad matches.
                    delta = calibrate_inject_offset_s(
                        record_dir / "A.wav", clip_path, expected_t,
                        search_pad_s=12.0,
                    )
                    # Bound accepts small negatives (sink(A)#1 stdio
                    # flush can lag the actual first frame by ~100ms)
                    # AND larger negatives (pacer queue holds bytes for
                    # 10-15s before sink(A) fires its first frame; the
                    # inject log timestamp then predates the anchor).
                    if delta is not None and -20.0 < delta < 15.0:
                        inject_offset_s = delta
                        anchor += delta  # shifts ALL event times back
            except Exception:
                pass

    events: list[dict] = []
    events += parse_clip_player_events(record_dir / "adapter_A.log", anchor)
    events += parse_b_log_events(record_dir / "adapter_B.log", anchor)
    events += parse_tool_calls(record_dir / "tool_log.json", anchor)

    # Attach clip text via decision tree
    if task_id:
        tree_path = ARTIFACTS_PATH / "decision_trees" / f"{task_id}.json"
        attach_clip_text(events, tree_path)

    # Attach source-audio windows (the timestamps in the original session
    # recording where this clip's audio was extracted from). Used by the
    # dashboard to scrub the source waveform in sync with playback.
    source_windows = load_clip_source_windows(task_id) if task_id else {}
    for ev in events:
        if ev.get("kind") != "customer_clip_injected":
            continue
        win = source_windows.get(ev.get("node_id") or "")
        if win:
            ev["source_start"], ev["source_end"] = win

    # Merge each customer_clip_injected (from A) into the nearest
    # customer_speech range (from B): the clip's text is what was PLAYED to
    # the agent, so it doubles as the spoken transcript for B's VAD-detected
    # speech segment.
    #
    # Customer audio is purely pre-recorded clips, so every customer_speech
    # range SHOULD correspond to exactly one clip-inject event. In practice
    # B's VAD can fragment one clip into multiple speech ranges when the
    # clip has internal pauses (e.g. "Soy sauce. ... No, no, no, let me
    # get ponzu sauce"). We claim the first range per clip and drop the
    # rest as noise within that clip's playback window.
    cust_speech = sorted(
        [e for e in events if e["kind"] == "customer_speech"],
        key=lambda r: r["t"],
    )
    drop = set()
    inject_events = [e for e in events if e["kind"] == "customer_clip_injected"]
    inject_events.sort(key=lambda e: e["t"])

    # For each clip-inject, claim ALL B-side VAD ranges that fall within its
    # playback window (from inject.t up to the next inject's t). We merge
    # them into one row but preserve their original {t,t_end} pairs in
    # `vad_segments` so the dashboard can still surface the fine-grained
    # timing if needed.
    #
    # The merged row's `t` is the INJECT time (A's wall clock), not the VAD
    # speech_started time. The clip's audio lands in A.wav at the moment
    # clipPlayer pushes bytes to the WebRTC source; B's VAD detects that
    # speech 1-2s later (Opus jitter buffer + OpenAI server-side debouncing).
    # For dashboard ↔ combined.wav alignment we want the inject time.
    claimed: set[int] = set()
    for i, ev in enumerate(inject_events):
        next_t = inject_events[i + 1]["t"] if i + 1 < len(inject_events) else float("inf")
        matched = [
            r for r in cust_speech
            if id(r) not in claimed
            and r["t"] >= ev["t"] - 4.0
            and r["t"] < next_t
        ]
        if matched:
            primary = matched[0]
            primary.setdefault("text", ev.get("text") or "")
            primary["node_id"] = ev["node_id"]
            primary["clip_filename"] = ev.get("clip_filename")
            if ev.get("is_fallback"):
                primary["is_fallback"] = True
            if ev.get("source_start") is not None:
                primary["source_start"] = ev["source_start"]
                primary["source_end"] = ev["source_end"]
            # Capture VAD timing BEFORE mutating primary. We preserve the
            # original B-clock segments in `vad_segments` so the user can see
            # when each VAD span fired in the recording, while shifting the
            # row's own t to the inject time.
            vad_segments = [
                {"t": round(r["t"], 3), "t_end": round(r["t_end"], 3)}
                for r in matched
            ]
            primary["t"] = ev["t"]
            # Block duration priority:
            #   1. played_dur_s — bytes/sr from clipPlayer's `Injected`
            #      log line. This is what was ACTUALLY pushed at eval
            #      time, immune to post-eval refiner narrowing of the
            #      source-side manifest. The source wav on disk today
            #      can be shorter than what was played (refiner shrank
            #      audio_end after the eval ran), and bar should reflect
            #      what was audible to the agent, not the current file.
            #   2. source_start..source_end — manifest's refined window
            #      (used for legacy runs whose `Injected` log lines
            #      lacked the bytes/sr suffix).
            #   3. vad_span — used only for synthetic yes/no clips not
            #      in the source manifest.
            if ev.get("played_dur_s") is not None:
                clip_dur = float(ev["played_dur_s"])
            elif ev.get("source_start") is not None and ev.get("source_end") is not None:
                clip_dur = float(ev["source_end"]) - float(ev["source_start"])
            else:
                clip_dur = vad_segments[-1]["t_end"] - vad_segments[0]["t"]
            primary["t_end"] = round(ev["t"] + max(0.05, clip_dur), 3)
            primary["vad_segments"] = vad_segments
            for r in matched:
                claimed.add(id(r))
            drop.add(id(ev))
            # Mark non-primary VAD ranges for removal — their timing lives
            # on inside primary.vad_segments.
            for r in matched[1:]:
                drop.add(id(r))
        else:
            # No B-side VAD range matches this inject — typically because
            # B's VAD missed the clip's audio (short / quiet / soft onset).
            # Re-cast as a customer_speech entry, AND seed t_end from the
            # actual played duration so the timeline block reflects the
            # real played clip (not the 1s default fallback in getSpeechEnd
            # nor a now-narrower refined manifest window).
            ev["kind"] = "customer_speech"
            if ev.get("played_dur_s") is not None:
                ev["t_end"] = round(ev["t"] + max(0.05, float(ev["played_dur_s"])), 3)
            elif ev.get("source_start") is not None and ev.get("source_end") is not None:
                clip_dur = float(ev["source_end"]) - float(ev["source_start"])
                ev["t_end"] = round(ev["t"] + max(0.05, clip_dur), 3)

    # Drop ANY remaining un-claimed customer_speech rows — these are stray
    # VAD events that didn't fall inside any clip's playback window (rare;
    # usually echoes/late-firing VAD).
    for r in cust_speech:
        if id(r) not in claimed:
            drop.add(id(r))

    events = [e for e in events if id(e) not in drop]

    # Attach the routing oracle's reasoning to each customer_speech row.
    # When the model picks a node via select_clip(node_id) or select_fallback,
    # the reasoning explains WHY this clip was chosen. We pair each
    # customer_speech with the most recent routing event that resolved to
    # the same node_id (or a fallback on that node).
    routing_events = sorted(
        [e for e in events if e["kind"] == "routing"],
        key=lambda e: e["t"],
    )
    # Since the short-id rollout, `play("...")` router picks arrive in the
    # log as SHORT ids (e.g. `i99_thats_all`), while customer_speech rows
    # still key on the LONG in-tree id (`g8_s2_i99_thats_all`). Match after
    # stripping the `g<g>_s<s>_` prefix from both sides so the reasoning
    # + tool_response attach for short-id picks too.
    # Strip both legacy `__fb` and new per-item `__fb_i<N>` (compound
    # main sub-clip swap) so a customer_speech row for a sub-clip pairs
    # with the router's routing event on the base compound id.
    _fb_sub_re = re.compile(r"__fb_i\d+$")
    def _tail(nid: str | None) -> str | None:
        if not nid:
            return nid
        if nid.endswith("__fb"):
            base = nid[:-4]
        else:
            sub_m = _fb_sub_re.search(nid)
            base = nid[: sub_m.start()] if sub_m else nid
        m = re.match(r"^g\d+_s\d+_(.+)$", base)
        return m.group(1) if m else base
    # Pairing is ONE-TO-ONE and order-preserving: the Nth play of a node
    # binds to the Nth routing decision that resolved to that node, and
    # each routing event is consumed exactly once. A many-to-one join
    # (take the latest matching decision, no consumption) misattributes
    # whenever the same node is played twice — the clipPlayer replays an
    # unacked clip with NO new routing decision, so the earlier decision
    # would leak forward onto a replay it did not cause.
    #
    # ROUTING_BIND_TOLERANCE_S: the routing line is emitted around the
    # inject_start it triggers, and which lands first is a race (observed
    # both +9ms and -1ms in the same run). A decision may therefore sit
    # slightly AFTER the clip it caused, so allow a small forward window
    # instead of requiring `routing.t <= speech.t` — that strict form
    # orphans a decision from its own clip, which then leaks onto the
    # next play of the same node.
    ROUTING_BIND_TOLERANCE_S = 0.25
    consumed: set[int] = set()
    for r in sorted(
        (e for e in events if e["kind"] == "customer_speech"),
        key=lambda e: e["t"],
    ):
        nid = r.get("node_id")
        if not nid:
            continue
        r_tail = _tail(nid)
        chosen = next(
            (
                rev for rev in routing_events
                if id(rev) not in consumed
                and rev["t"] <= r["t"] + ROUTING_BIND_TOLERANCE_S
                and _tail(rev.get("node_id")) == r_tail
            ),
            None,
        )
        # No unconsumed decision for this node: this play is an
        # auto-replay (or otherwise router-less). Leave it unannotated
        # rather than inheriting a stale decision.
        if chosen is None:
            continue
        consumed.add(id(chosen))
        r["chosen_reasoning"] = chosen.get("reasoning")
        r["chosen_decision"] = chosen.get("decision")
        if chosen.get("tool_response") is not None:
            r["chosen_tool_response"] = chosen["tool_response"]

    # Merge A's transcription of B's audio onto agent_speech rows.
    # `agent_a_heard` ranges come from A's whisper-transcribed input audio
    # (A's perception of what the agent said). For each agent_speech row,
    # find the A-heard range whose time interval most overlaps it and
    # attach the transcript as `a_heard_text`. A-side has ~1.7s delivery
    # latency so we widen the overlap window.
    a_heard_ranges = sorted(
        [e for e in events if e["kind"] == "agent_a_heard"],
        key=lambda e: e["t"],
    )
    drop_a_heard = set()
    for r in events:
        if r["kind"] != "agent_speech":
            continue
        r_start = r["t"]
        r_end = r.get("t_end") or (r_start + 1.0)
        best = None
        best_score = -1
        for ah in a_heard_ranges:
            if id(ah) in drop_a_heard:
                continue
            ah_start = ah["t"]
            ah_end = ah.get("t_end") or (ah_start + 1.0)
            # Allow A-side window to be shifted up to 2.5s later than the
            # agent_speech (event-delivery latency we measured).
            if ah_end < r_start - 0.5 or ah_start > r_end + 2.5:
                continue
            # Overlap score: width of intersection, biased toward A-side
            # starting after B-side (typical due to latency).
            o0 = max(r_start, ah_start)
            o1 = min(r_end + 2.5, ah_end)
            overlap = max(0.0, o1 - o0)
            if overlap > best_score:
                best_score = overlap
                best = ah
        if best and best_score > 0:
            r["a_heard_text"] = best.get("text", "")
            drop_a_heard.add(id(best))
    # Relabel A-heard ranges as `vad_a_of_b` — A's session VAD-detected
    # B speaking. The merge loop above already pulled the transcript
    # onto matching agent_speech rows; we keep the ranges as standalone
    # events so the mini-timeline can render them as gap-band markers
    # (under the agent track).
    for e in events:
        if e["kind"] == "agent_a_heard":
            e["kind"] = "vad_a_of_b"
    # Mirror B's VAD-detected-A ranges as `vad_b_of_a` markers for the
    # lower gap band. CRITICAL: customer_speech.t/t_end was overwritten
    # earlier (~L1272/L1286) with the INJECT wall-clock time so the
    # rendered bar aligns with combined.wav. The actual B-side VAD
    # timestamps are preserved on each row's `vad_segments[]`. Use those
    # directly so the gap-band shows the real VAD detection windows
    # (typically 0.5-2s offset from inject, reflecting Opus jitter +
    # OpenAI server-side debouncing latency).
    for e in [ev for ev in events if ev.get("kind") == "customer_speech"]:
        for seg in e.get("vad_segments") or []:
            events.append({
                "t": seg["t"],
                "t_end": seg.get("t_end"),
                "kind": "vad_b_of_a",
            })

    events.sort(key=lambda e: (e["t"], 0 if e["kind"] == "customer_clip_injected" else 1))

    # Pull task-level info
    expected_cart = score.get("expected_cart") or []
    actual_cart = (final_cart.get("cart") or [])
    submitted = bool(final_cart.get("submitted"))

    # Variant signal — runs scored with the complex menu have score.json's
    # `variant` field set to "complex" by run_order_up_eval.sh. Default to
    # "simple" for historical runs predating the variant stamp.
    variant = score.get("variant") or "simple"

    # Pick the task file matching the run's variant. Complex runs read
    # the menu_sample (and prompt) from `<tid>.complex.task.json`; simple
    # runs read `<tid>.task.json` as before. We fall back to the simple
    # file if the complex sibling is missing, which only happens when an
    # older run was scored against a since-removed complex config.
    task_meta = {}
    source_audio_offset = None
    source_audio_duration = None
    reference_transcript: list[dict] = []
    stages_meta: dict[str, dict] = {}
    task_path = None
    if task_id:
        cand = ARTIFACTS_PATH / "order_up_tasks" / f"{task_id}.complex.task.json"
        if cand.exists():
            task_path = cand
    if task_path and task_path.exists():
        t = json.loads(task_path.read_text())
        task_meta = {
            "restaurant_name": t.get("restaurant_name"),
            "menu_sample": t.get("menu_sample"),
        }
        sid = t.get("session_id")
        gi = t.get("game_index")
        if sid and gi is not None:
            win = load_game_audio_window(sid, int(gi))
            if win is not None:
                # Mirror the slicing logic from viz_dashboard._send_source_clip:
                # the slice is [max(0, audio_start − 0.5), audio_end + 0.5].
                source_audio_offset = max(0.0, win[0] - 0.5)
                source_audio_duration = (win[1] + 0.5) - source_audio_offset
            if task_id:
                reference_transcript = load_reference_transcript(
                    task_id, sid, int(gi),
                    source_audio_offset if source_audio_offset is not None else 0.0,
                )
            # Per-stage expected_items (drives the item_indices multi-select
            # in the span-edit UI).
            align_path = (
                ARTIFACTS_PATH / "sessions" / "order_up" / sid / "stage_alignment.json"
            )
            if align_path.exists():
                try:
                    align = json.loads(align_path.read_text())
                    g = align["games"][int(gi)]
                    for stage in g.get("stages") or []:
                        si = stage.get("stage_index")
                        if si is None:
                            continue
                        stages_meta[str(si)] = {
                            "stage_index": si,
                            "stage_type": stage.get("stage_type"),
                            "expected_items": stage.get("expected_items") or [],
                            "expected_item_ids": stage.get("expected_item_ids") or [],
                        }
                except Exception:
                    pass

    # Per-run prompt snapshot (preferred). Falls back to live task config so
    # older runs that lacked the snapshot still display something useful.
    prompt_path = record_dir / "prompt.json"
    if prompt_path.exists():
        try:
            prompt_data = json.loads(prompt_path.read_text())
        except Exception:
            prompt_data = {}
    else:
        prompt_data = {}
        if task_path and task_path.exists():
            tt = json.loads(task_path.read_text())
            prompt_data = {
                "examinee_system_prompt": tt.get("examinee_system_prompt"),
                "tools": tt.get("tools"),
                "snapshot_missing": True,
            }

    # Audit sidecar summaries. Each is None if the audit didn't run for
    # this record (older runs, or the trigger tag wasn't set). The UI
    # reads these alongside score.tags to render a per-run audit panel
    # explaining WHY each cart-diff tag fired.
    audits = _load_audit_sidecars(record_dir)
    diff_d = score.get("diff") or {}
    diff_summary = {
        "missing": diff_d.get("missing") or [],
        "extra": diff_d.get("extra") or [],
        "wrong_option_or_variant": diff_d.get("wrong_option_or_variant") or [],
        "matched": diff_d.get("matched") or [],
        "totals": diff_d.get("totals") or {},
    }

    return {
        "task_id": task_id,
        "run_ts": record_dir.name,
        "variant": variant,
        "score": score.get("Task-specific score"),
        "submitted": submitted,
        "tags": list(score.get("tags") or []),
        "finalize_reason": score.get("finalize_reason"),
        "diff_summary": diff_summary,
        "audits": audits,
        "expected_cart": expected_cart,
        "final_cart": actual_cart,
        "anchor_epoch_s": anchor,
        "audio_duration_s": round(audio_duration_s(record_dir / "combined.wav"), 2),
        "source_audio_offset_s": (
            round(source_audio_offset, 3) if source_audio_offset is not None else None
        ),
        "source_audio_duration_s": (
            round(source_audio_duration, 3) if source_audio_duration is not None else None
        ),
        "events": events,
        "reference_transcript": reference_transcript,
        "stages": stages_meta,
        "prompt": prompt_data,
        **task_meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("record_dir", type=Path)
    args = ap.parse_args()
    data = build(args.record_dir)
    out = args.record_dir / "timeline.json"
    out.write_text(json.dumps(data, indent=2))
    print(
        f"wrote {out} ({len(data['events'])} events, "
        f"audio={data['audio_duration_s']}s, score={data['score']})"
    )


if __name__ == "__main__":
    main()
