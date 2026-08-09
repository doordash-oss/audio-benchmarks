"""Per-run behavioral metrics for one Order Up eval.

Reads an eval's record_dir (the directory containing adapter_A.log,
orchestrator.log, B.wav, score.json, prompt.json) and writes
`<record_dir>/eval_metrics.json` with per-clip turn metrics + a
per-run aggregate including bucketed breakdowns.

Per-clip metrics:
  - agent_response_latency_s    silence-to-speech transition relative to clip end
  - agent_response_duration_s   summed agent speech across the response window
  - responded                   any meaningful agent speech in the window
  - interrupted                 the inline RMS-VAD logged `interrupt_flag`
  - clip_props                  duration, muted_intervals presence,
                                max intra-customer pause, etc.

Aggregate buckets:
  - by_clip_duration_lt_1_5s   short clips vs ≥1.5s
  - by_has_muted_agent_block   clips that originally contained an
                               agent acknowledgement / backchannel
                               (natural mid-clip yield point)
  - by_long_intra_clip_pause   clips with ≥0.5s gap between customer
                               sub-spans (natural mid-clip yield)
  - by_clip_role               main / resp / corr / opt

Run:
    python -m clip_eval.scoring.eval_metrics <record_dir>
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import wave
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

# ---------------------------------------------------------------------------
# RMS hysteresis VAD with (start, end) interval output.
# ---------------------------------------------------------------------------
WIRE_SR = 48000
FRAME_S = 0.01          # 10 ms buckets
SPEECH_DB = -45.0       # threshold to ENTER speech state
SILENCE_DB = -50.0      # threshold to EXIT speech state (hysteresis)
ENTRY_BUCKETS = 3       # consecutive above-threshold → onset
EXIT_BUCKETS = 20       # consecutive below-threshold → offset (200 ms)
MIN_GAP_S = 0.30        # merge adjacent intervals separated by <300 ms

# Per-clip thresholds.
LONG_PAUSE_THRESHOLD_S = 0.5   # gap between customer sub-spans counts as "long"
SHORT_CLIP_THRESHOLD_S = 1.5   # bucket boundary for "short clip"
RESPONSE_MIN_DURATION_S = 0.10  # filter transient noise from the responded flag

ISO_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\]")
INJECT_RE = re.compile(r"\[ClipPlayer\] inject_start (\S+)")
INTERRUPT_RE = re.compile(
    r"\[ClipPlayer\] interrupt_flag clip_id=(\S+) delta_ms=(-?\d+)"
)
MISSING_ANSWER_RE = re.compile(
    r"\[ClipPlayer\] missing_answer stage_index=(\d+) item_index=(\d+)"
)
SINK_B_RE = re.compile(r"sink\(B\)#1\b")


def _parse_iso(s: str) -> float:
    return (
        datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )


def detect_wav_intervals(wav_path: Path) -> tuple[list[tuple[float, float]], float]:
    """Run RMS-hysteresis VAD on a mono PCM16 wav file and return
    (intervals, duration_s). Each interval is `(speech_start, speech_end)`
    in seconds relative to the wav's t=0. Adjacent intervals separated
    by <MIN_GAP_S are merged so a single long utterance with a brief
    intra-word dip stays as one interval."""
    with wave.open(str(wav_path), "rb") as w:
        n = w.getnframes()
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw != 2:
        raise RuntimeError(f"{wav_path.name}: expected PCM16, got sampwidth={sw}")
    if sr != WIRE_SR:
        raise RuntimeError(f"{wav_path.name}: expected {WIRE_SR}Hz, got {sr}Hz")
    # Defensive against partial/streaming WAV headers: trust the bytes
    # actually returned by readframes(), not the header's frame count.
    # WAVs written by interrupted runs (sweep kill mid-stream) can have
    # a `data` chunk size of 0xFFFFFFFF (streaming sentinel) that makes
    # getnframes() return a multi-GB value, which then crashes
    # struct.unpack. Clamp n to what we actually have on hand.
    actual_n = len(raw) // (ch * sw)
    if actual_n != n:
        print(
            f"  [eval_metrics] WARN: {wav_path.name} header reports "
            f"{n} frames but only {actual_n} are present — clamping "
            f"(file may have been truncated by sweep kill)"
        )
        n = actual_n
        raw = raw[: n * ch * sw]
    samples = struct.unpack(f"<{n * ch}h", raw)
    if ch > 1:
        samples = samples[::ch]
    duration_s = len(samples) / sr

    frame = int(sr * FRAME_S)
    nb = len(samples) // frame
    # Per-frame RMS in dBFS
    db = [0.0] * nb
    for i in range(nb):
        s = samples[i * frame : (i + 1) * frame]
        sq = sum(x * x for x in s) / len(s)
        rms = math.sqrt(sq) / 32768.0
        db[i] = 20.0 * math.log10(max(rms, 1e-9))

    intervals: list[tuple[float, float]] = []
    in_speech = False
    above_run = 0
    below_run = 0
    onset_idx: int | None = None
    last_above_idx: int | None = None
    for i, x in enumerate(db):
        if not in_speech:
            if x > SPEECH_DB:
                above_run += 1
                if above_run >= ENTRY_BUCKETS:
                    onset_idx = i - ENTRY_BUCKETS + 1
                    in_speech = True
                    last_above_idx = i
                    below_run = 0
            else:
                above_run = 0
        else:
            if x < SILENCE_DB:
                below_run += 1
                if below_run >= EXIT_BUCKETS:
                    # offset = last_above_idx + 1 frame (end of frame)
                    end_idx = (last_above_idx if last_above_idx is not None else i) + 1
                    intervals.append((onset_idx * FRAME_S, end_idx * FRAME_S))
                    in_speech = False
                    above_run = 0
                    onset_idx = None
            else:
                below_run = 0
                last_above_idx = i
    if in_speech and onset_idx is not None:
        end_idx = (last_above_idx if last_above_idx is not None else nb - 1) + 1
        intervals.append((onset_idx * FRAME_S, end_idx * FRAME_S))

    # Merge near-adjacent intervals so a short intra-utterance dip
    # doesn't fragment one agent turn into two.
    merged: list[tuple[float, float]] = []
    for s, e in intervals:
        if merged and s - merged[-1][1] < MIN_GAP_S:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged, duration_s


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------
def get_b_wav_anchor(orch_log: Path) -> float:
    """Return the wall-clock epoch seconds of the orchestrator's first
    `sink(B)#1` line (when it began receiving / writing B's audio)."""
    with orch_log.open() as f:
        for line in f:
            if not SINK_B_RE.search(line):
                continue
            m = ISO_RE.match(line)
            if m:
                return _parse_iso(m.group(1))
    raise RuntimeError(f"no sink(B)#1 marker in {orch_log}")


def parse_adapter_a_log(
    adapter_a_log: Path,
) -> tuple[list[dict], dict[str, int], list[dict]]:
    """Parse adapter_A.log for inject_start events, interrupt_flag log
    lines, and missing_answer log lines. Returns
    (injects, interrupts, missing_answers) where:
      - injects: chronological list of {"wall": float, "clip_id": str}
      - interrupts: {clip_id: delta_ms} — one per flagged inject (the
        adapter logs one flag per inject due to its lastInjectFlagged
        one-shot guard).
      - missing_answers: chronological list of
        {"wall": float, "stage_index": int, "item_index": int}.
        The most-recent inject_start prior to each missing_answer is
        the clip whose response window contained the unanswerable
        agent question."""
    injects: list[dict] = []
    interrupts: dict[str, int] = {}
    missing_answers: list[dict] = []
    with adapter_a_log.open() as f:
        for line in f:
            ts_m = ISO_RE.match(line)
            if not ts_m:
                continue
            ts = _parse_iso(ts_m.group(1))
            m_inj = INJECT_RE.search(line)
            if m_inj:
                injects.append({"wall": ts, "clip_id": m_inj.group(1)})
                continue
            m_int = INTERRUPT_RE.search(line)
            if m_int:
                interrupts[m_int.group(1)] = int(m_int.group(2))
                continue
            m_ma = MISSING_ANSWER_RE.search(line)
            if m_ma:
                missing_answers.append({
                    "wall": ts,
                    "stage_index": int(m_ma.group(1)),
                    "item_index": int(m_ma.group(2)),
                })
    return injects, interrupts, missing_answers


def _is_synthetic_clip(clip_id: str) -> bool:
    """Synthetic injects (yes / no / fallback) don't carry the
    customer-line metadata stored in customer_clips/<task>.json.
    Identified by adapter_A.log naming conventions in clipPlayer."""
    suffixes = ("_main_yes", "_main_no", "__fb")
    return any(clip_id.endswith(s) for s in suffixes)


# ---------------------------------------------------------------------------
# Manifest lookup
# ---------------------------------------------------------------------------
def load_manifest_for_run(record_dir: Path) -> tuple[str, dict[str, dict]]:
    """Load `artifacts/customer_clips/<task_id>.json` and index its clips
    by clip_id. Returns (task_id, clips_by_id). The manifest path is
    derived from prompt.json's task_id."""
    prompt = json.loads((record_dir / "prompt.json").read_text())
    task_id = prompt["task_id"]
    v2_root = record_dir.parents[3]   # outputs/OrderUp/<task_id>/<run_ts>/
    manifest_path = v2_root / "customer_clips" / f"{task_id}.json"
    if not manifest_path.exists():
        # Alternate layout: artifacts/customer_clips/...
        manifest_path = (
            v2_root.parent / "customer_clips" / f"{task_id}.json"
        )
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"customer_clips manifest not found for {task_id} "
            f"(tried {manifest_path})"
        )
    manifest = json.loads(manifest_path.read_text())
    by_id: dict[str, dict] = {}
    for c in manifest.get("clips") or []:
        cid = c.get("clip_id")
        if cid:
            by_id[cid] = c
    return task_id, by_id


def _clip_props(clip: dict | None) -> dict:
    """Derive bucketing/analysis properties from a customer_clips entry."""
    if clip is None:
        return {
            "clip_role": None,
            "clip_kind": None,
            "trigger_kind": None,
            "has_muted_intervals": False,
            "muted_total_s": 0.0,
            "n_customer_spans": 0,
            "max_inter_customer_gap_s": 0.0,
            "had_agent_block_in_source": False,
        }
    muted = clip.get("muted_intervals") or []
    spans = clip.get("customer_spans") or []
    # Largest gap between consecutive customer sub-spans within this clip.
    max_gap = 0.0
    for i in range(len(spans) - 1):
        gap = float(spans[i + 1].get("t_start", 0)) - float(spans[i].get("t_end", 0))
        if gap > max_gap:
            max_gap = gap
    return {
        "clip_role": clip.get("clip_role"),
        "clip_kind": clip.get("clip_kind"),
        "trigger_kind": clip.get("trigger_kind"),
        "has_muted_intervals": len(muted) > 0,
        "muted_total_s": round(sum(float(b) - float(a) for a, b in muted), 3),
        "n_customer_spans": len(spans),
        "max_inter_customer_gap_s": round(max_gap, 3),
        "had_agent_block_in_source": len(muted) > 0,
    }


# ---------------------------------------------------------------------------
# Per-clip computation
# ---------------------------------------------------------------------------
def _agent_segments_in_window(
    intervals: list[tuple[float, float]], t0: float, t1: float
) -> list[tuple[float, float]]:
    """Return the (clipped) agent-speech intervals that overlap [t0, t1]."""
    out = []
    for s, e in intervals:
        if e <= t0 or s >= t1:
            continue
        out.append((max(s, t0), min(e, t1)))
    return out


def _missing_answer_index_for_inject(
    inject_wall: float,
    next_inject_wall: float | None,
    missing_answers: list[dict],
) -> dict | None:
    """Return the missing_answer event whose wall time falls in
    (inject_wall, next_inject_wall] — i.e. the agent's unanswerable
    question happened during THIS clip's response window. Returns
    None if no missing_answer applies."""
    for ma in missing_answers:
        if ma["wall"] <= inject_wall:
            continue
        if next_inject_wall is not None and ma["wall"] > next_inject_wall:
            continue
        return ma
    return None


def compute_per_clip(
    injects: list[dict],
    interrupts: dict[str, int],
    missing_answers: list[dict],
    clips_by_id: dict[str, dict],
    agent_intervals: list[tuple[float, float]],
    anchor_wall: float,
    recording_duration_s: float,
) -> list[dict]:
    rows: list[dict] = []
    for i, inj in enumerate(injects):
        clip_id = inj["clip_id"]
        inject_t = round(inj["wall"] - anchor_wall, 3)
        synthetic = _is_synthetic_clip(clip_id)
        clip = clips_by_id.get(clip_id)
        if clip is not None:
            clip_dur = round(
                float(clip.get("audio_end", 0)) - float(clip.get("audio_start", 0)),
                3,
            )
        else:
            clip_dur = None  # unknown duration for synthetic clips
        clip_ends_t = (
            round(inject_t + clip_dur, 3) if clip_dur is not None else None
        )

        if i + 1 < len(injects):
            next_inject_t = round(injects[i + 1]["wall"] - anchor_wall, 3)
        else:
            next_inject_t = round(recording_duration_s, 3)

        # Window for measuring the agent's response to THIS clip.
        # Default lower bound is clip_ends_t; for synthetic / unknown
        # duration clips we fall back to inject_t.
        #
        # NOTE: `clip_ends_t = inject_t + clip_duration_s` is computed
        # against the `inject_start` log timestamp, NOT against the
        # clip's actual end on B.wav. The clip player logs `inject_start`
        # BEFORE WAV decode + pacing + WebRTC transit, so the audio
        # arrives at B's session ~150-250ms later than `inject_t`.
        # Consequently `clip_ends_t` is ~200ms EARLIER than the true
        # B-side clip end, biasing `agent_response_latency_s` upward
        # by that same ~200ms on every clip. The bias is approximately
        # constant across clips and runs, so relative comparisons (by
        # bucket, by task, across runs) are unaffected; absolute
        # latency numbers carry a known ~+200ms offset relative to
        # "agent reaction time after the audio actually finished."
        win_start = clip_ends_t if clip_ends_t is not None else inject_t
        win_end = next_inject_t

        latency_s: float | None = None
        duration_s = 0.0
        n_segments = 0
        if win_end > win_start:
            segs = _agent_segments_in_window(agent_intervals, win_start, win_end)
            n_segments = len(segs)
            if segs:
                latency_s = round(segs[0][0] - win_start, 3)
            duration_s = round(sum(e - s for s, e in segs), 3)

        # Did the routing model log a `missing_answer` somewhere in
        # THIS clip's response window? If so, the agent asked an
        # on-topic question the customer's clip set didn't cover —
        # the next inject is an advance via next_item logic, not a
        # real reply. Tag this turn for downstream metric handling.
        next_inj_wall = injects[i + 1]["wall"] if i + 1 < len(injects) else None
        ma = _missing_answer_index_for_inject(inj["wall"], next_inj_wall, missing_answers)
        row = {
            "clip_id": clip_id,
            "synthetic": synthetic,
            "inject_t_s": inject_t,
            "clip_duration_s": clip_dur,
            "clip_ends_t_s": clip_ends_t,
            "next_inject_t_s": next_inject_t,
            "response_window_s": [round(win_start, 3), round(win_end, 3)],
            "agent_response_latency_s": latency_s,
            "agent_response_duration_s": duration_s,
            "agent_response_segments": n_segments,
            "responded": duration_s >= RESPONSE_MIN_DURATION_S,
            "interrupted": clip_id in interrupts,
            "interrupt_delta_ms": interrupts.get(clip_id),
            "agent_asked_unanswerable": ma is not None,
            "missing_answer_target": (
                {"stage_index": ma["stage_index"], "item_index": ma["item_index"]}
                if ma else None
            ),
            "clip_props": _clip_props(clip),
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Aggregation + buckets
# ---------------------------------------------------------------------------
def _bucket_stats(rows: list[dict]) -> dict:
    """Compute n, response_rate, interrupt_rate, median latency + duration
    over a list of per-clip rows."""
    if not rows:
        return {
            "n": 0,
            "response_rate": None,
            "interrupt_rate": None,
            "median_latency_s": None,
            "median_duration_s": None,
        }
    n = len(rows)
    responded = sum(1 for r in rows if r["responded"])
    interrupted = sum(1 for r in rows if r["interrupted"])
    latencies = [r["agent_response_latency_s"] for r in rows if r["agent_response_latency_s"] is not None]
    durations = [r["agent_response_duration_s"] for r in rows if r["responded"]]
    return {
        "n": n,
        "response_rate": round(responded / n, 4),
        "interrupt_rate": round(interrupted / n, 4),
        "median_latency_s": round(median(latencies), 3) if latencies else None,
        "median_duration_s": round(median(durations), 3) if durations else None,
    }


def compute_aggregates(per_clip: list[dict]) -> dict:
    """Run-level aggregates over the non-synthetic clips."""
    real = [r for r in per_clip if not r["synthetic"]]
    overall = _bucket_stats(real)
    overall["n_unanswerable_agent_questions"] = sum(
        1 for r in real if r.get("agent_asked_unanswerable")
    )

    buckets: dict[str, dict] = {
        "by_clip_duration_lt_1_5s": {
            "true": _bucket_stats(
                [r for r in real if (r["clip_duration_s"] or 0) < SHORT_CLIP_THRESHOLD_S]
            ),
            "false": _bucket_stats(
                [r for r in real if (r["clip_duration_s"] or 0) >= SHORT_CLIP_THRESHOLD_S]
            ),
        },
        "by_has_muted_agent_block": {
            "true": _bucket_stats(
                [r for r in real if r["clip_props"]["has_muted_intervals"]]
            ),
            "false": _bucket_stats(
                [r for r in real if not r["clip_props"]["has_muted_intervals"]]
            ),
        },
        "by_long_intra_clip_pause": {
            "true": _bucket_stats(
                [r for r in real if r["clip_props"]["max_inter_customer_gap_s"] >= LONG_PAUSE_THRESHOLD_S]
            ),
            "false": _bucket_stats(
                [r for r in real if r["clip_props"]["max_inter_customer_gap_s"] < LONG_PAUSE_THRESHOLD_S]
            ),
        },
        # Clips whose response window contained a `missing_answer`
        # event get singled out — the agent asked something on-topic
        # that the clip set couldn't answer, so the response-rate /
        # interrupt-rate signal here is about clip coverage, not
        # about the agent's behavior.
        "by_agent_asked_unanswerable": {
            "true": _bucket_stats(
                [r for r in real if r.get("agent_asked_unanswerable")]
            ),
            "false": _bucket_stats(
                [r for r in real if not r.get("agent_asked_unanswerable")]
            ),
        },
        "by_clip_role": {},
    }
    for role in ("main", "resp", "corr", "opt"):
        buckets["by_clip_role"][role] = _bucket_stats(
            [r for r in real if r["clip_props"]["clip_role"] == role]
        )

    overall["buckets"] = buckets
    return overall


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def analyze_run(record_dir: Path) -> dict:
    adapter_a = record_dir / "adapter_A.log"
    orch_log = record_dir / "orchestrator.log"
    b_wav = record_dir / "B.wav"
    for p in (adapter_a, orch_log, b_wav):
        if not p.exists():
            raise FileNotFoundError(f"missing eval artifact: {p}")

    task_id, clips_by_id = load_manifest_for_run(record_dir)
    anchor_wall = get_b_wav_anchor(orch_log)
    injects, interrupts, missing_answers = parse_adapter_a_log(adapter_a)
    agent_intervals, recording_duration_s = detect_wav_intervals(b_wav)
    per_clip = compute_per_clip(
        injects, interrupts, missing_answers, clips_by_id, agent_intervals,
        anchor_wall, recording_duration_s,
    )
    aggregate = compute_aggregates(per_clip)
    return {
        "task_id": task_id,
        "run_ts": record_dir.name,
        "anchor_wall_s": anchor_wall,
        "recording_duration_s": round(recording_duration_s, 3),
        "n_clips": len(per_clip),
        "per_clip": per_clip,
        "aggregate": aggregate,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("record_dir", type=Path,
                    help="One eval's record directory (contains B.wav, adapter_A.log, …).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path (default: <record_dir>/eval_metrics.json).")
    ap.add_argument("--stdout", action="store_true",
                    help="Print result to stdout instead of writing to disk.")
    args = ap.parse_args()
    if not args.record_dir.exists():
        print(f"record_dir does not exist: {args.record_dir}", file=sys.stderr)
        return 2
    result = analyze_run(args.record_dir)
    text = json.dumps(result, indent=2)
    if args.stdout:
        print(text)
    else:
        out = args.out or (args.record_dir / "eval_metrics.json")
        out.write_text(text)
        agg = result["aggregate"]
        print(
            f"[eval_metrics] {args.record_dir.name}: "
            f"n_clips={result['n_clips']} "
            f"response_rate={agg.get('response_rate')} "
            f"interrupt_rate={agg.get('interrupt_rate')} "
            f"median_latency_s={agg.get('median_latency_s')} "
            f"-> {out}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
