"""Full-Duplex-Bench-style turn-taking analysis for one Order-Up run.

DETERMINISTIC. No LLM. All labels are arithmetic on VAD timestamps, agent
word counts, and (later) the cart diff. Implements the locked spec:

  Timeline: agent speech intervals (RMS-VAD on B.wav) and clip inject_start
  times share one clock (orchestrator anchor). So we can place every agent
  onset relative to the clip that was playing.

  Turn boundary  = customer's last voiced moment IN the clip (RMS-VAD on the
                   clip's customer wav), expressed on the run clock as
                   inject_t + last_speech_in_clip.
  Backchannel    = agent utterance < 1.0 s AND < 2 words  (paper rule).
  Takeover (TO)  = agent produced real (non-silent, non-backchannel) speech
                   in the clip's response window.
  Interruption   = a FRESH agent onset (silence -> speech) that starts during
   (barge-in)      the clip, BEFORE the turn boundary. A real utterance.
  Carryover      = agent was ALREADY speaking when the clip started (an agent
   (NOT counted)   interval spans inject_t). Excluded — it's pacing, not a
                   barge-in.
  Where          = at the barge-in onset, was the customer in active speech
                   or a natural pause (0.4-1.0 s gap)?  [active | pause]

Per-clip label: no_response | backchannel | interruption | good_take

Writes <record_dir>/turn_taking.json.

Run:  python -m clip_eval.scoring.turn_taking_analysis <record_dir>
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from clip_eval.scoring.eval_metrics import (
    detect_wav_intervals,
    get_b_wav_anchor,
    parse_adapter_a_log,
    _parse_iso,
    ISO_RE,
    _is_synthetic_clip,
)
from clip_eval.pipeline.paths import REPO_ROOT, ARTIFACTS_PATH

# Paper thresholds.
BACKCHANNEL_MAX_DURATION_S = 1.0
BACKCHANNEL_MAX_WORDS = 2          # "< 2 words" -> 0 or 1 words
NATURAL_PAUSE_MIN_S = 0.4
NATURAL_PAUSE_MAX_S = 1.0
BARGEIN_TOLERANCE_S = 0.25         # jitter + 200-250ms human transition window
CLEAR_MARGIN_S = 1.0              # onset must be >=1s before customer's last word to
                                  # count as a CLEAR interruption (else "marginal/early")
# A genuine barge-in INTO this clip means the agent let the customer talk for a
# beat and THEN cut in. If the agent's onset is in the first MIN_BARGEIN_ONSET_S
# of the clip, it isn't cutting into THIS clip's content — it's a spurious
# VAD trigger on background noise / a spill-over from the previous turn. Human
# review (Jun 2026) of all 30 clear candidates: every confirmed-real one starts
# >=2.44s in; the false positives all fire <1.5s in. Gating the CLEAR tier on
# this makes the headline interruption rate deterministic (no hand-verify needed).
MIN_BARGEIN_ONSET_S = 2.0
# #1 artifact filter: a flagged interruption whose onset is within the first
# EARLY_ONSET_S of the clip AND whose PREVIOUS clip got no response is not a
# real barge-in — it's a delayed response to the previous clip (or a timeout
# collision) that coincided with this clip's start. Reclassified as non-response.
EARLY_ONSET_S = 1.5

# Agent transcript event shapes (OpenAI realtime vs Gemini/pipecat).
_OAI_DONE_RE = re.compile(r'"type":"response\.output_audio_transcript\.done"')
_TRANSCRIPT_FIELD_RE = re.compile(r'"transcript":"((?:[^"\\]|\\.)*)"')
_GEMINI_RESP_RE = re.compile(r'"type":\s*"transcript",\s*"source":\s*"response"')
_TEXT_FIELD_RE = re.compile(r'"text":\s*"((?:[^"\\]|\\.)*)"')
# pipecat (gemini_live / cascaded) emits the full agent utterance here, with an
# ISO timestamp: `[<iso>] [roleB-viz] agent_transcript {"text": "..."}`. OpenAI
# does NOT emit this (it uses response.output_audio_transcript.done), so there is
# no double-counting. This is the per-clip word source for gemini + cascade.
_VIZ_TRANSCRIPT_RE = re.compile(r'\[roleB-viz\]\s+agent_transcript\s')
# clipPlayer logs this when its 8s watchdog fires (agent never responded).
_NO_SPEECH_RE = re.compile(r"No agent speech within \d+ms")


def _parse_timeouts(adapter_b_log: Path, anchor: float) -> list[float]:
    """Return clipPlayer-timeout times (rel to anchor) — moments the agent
    failed to respond within the 8s watchdog. Parsed from adapter_A.log."""
    out: list[float] = []
    if not adapter_b_log.exists():
        return out
    for line in adapter_b_log.read_text(errors="ignore").splitlines():
        m = ISO_RE.match(line)
        if m and _NO_SPEECH_RE.search(line):
            out.append(_parse_iso(m.group(1)) - anchor)
    return out


def _word_count(text: str) -> int:
    cleaned = re.sub(r"[^\w\s']", " ", text or "").strip()
    return len(cleaned.split()) if cleaned else 0


def _parse_agent_transcripts(adapter_b_log: Path, anchor: float) -> list[dict]:
    out: list[dict] = []
    if not adapter_b_log.exists():
        return out
    for line in adapter_b_log.read_text(errors="ignore").splitlines():
        m_iso = ISO_RE.match(line)
        if not m_iso:
            continue
        t_rel = _parse_iso(m_iso.group(1)) - anchor
        text = None
        if _OAI_DONE_RE.search(line):
            mt = _TRANSCRIPT_FIELD_RE.search(line)
            if mt: text = mt.group(1)
        elif _GEMINI_RESP_RE.search(line) or _VIZ_TRANSCRIPT_RE.search(line):
            mt = _TEXT_FIELD_RE.search(line)
            if mt: text = mt.group(1)
        if text is None:
            continue
        try: text = bytes(text, "utf-8").decode("unicode_escape")
        except Exception: pass
        out.append({"t_rel": round(t_rel, 3), "text": text, "words": _word_count(text)})
    return out


def _clip_customer_vad(clip_id: str, manifest: dict):
    """Return (speech_intervals, duration) from RMS-VAD on the clip's
    customer wav, or (None, None) if unavailable. Intervals are seconds
    from clip start."""
    clip = next((c for c in manifest.get("clips", []) if c.get("clip_id") == clip_id), None)
    if not clip or not clip.get("audio_path"):
        return None, None
    wav = REPO_ROOT / clip["audio_path"]
    if not wav.exists():
        return None, None
    try:
        return detect_wav_intervals(wav)
    except Exception:
        return None, None


def _agent_state_at(intervals, t):
    """Is the agent mid-speech at time t? Returns the interval covering t, or None."""
    for s, e in intervals:
        if s <= t <= e:
            return (s, e)
    return None


def _first_fresh_onset_in(intervals, t0, t1):
    """Earliest agent speech onset strictly within (t0, t1] that is a FRESH
    onset (the interval starts in the window — i.e. silence preceded it).
    Returns onset time or None."""
    best = None
    for s, e in intervals:
        if t0 < s <= t1:
            if best is None or s < best:
                best = s
    return best


def _words_in_window(transcripts, t0, t1):
    return sum(tr["words"] for tr in transcripts
              if tr["t_rel"] >= t0 and (t1 is None or tr["t_rel"] <= t1))


def _customer_state_at(cust_intervals, t):
    """active | pause | outside — what was the customer doing at clip-relative
    time t. 'pause' only if the gap the onset lands in is 0.4-1.0s (paper)."""
    if not cust_intervals:
        return "unknown"
    # inside a speech interval?
    for s, e in cust_intervals:
        if s <= t <= e:
            return "active"
    # in a gap between two customer intervals?
    for i in range(len(cust_intervals) - 1):
        gap_s = cust_intervals[i][1]
        gap_e = cust_intervals[i + 1][0]
        if gap_s < t < gap_e:
            gap = gap_e - gap_s
            return "pause" if NATURAL_PAUSE_MIN_S <= gap <= NATURAL_PAUSE_MAX_S else "active"
    return "active"  # before first / after last handled by boundary logic


def analyze(record_dir: Path) -> dict:
    adapter_a = record_dir / "adapter_A.log"
    orch = record_dir / "orchestrator.log"
    b_wav = record_dir / "B.wav"
    if not (adapter_a.exists() and orch.exists() and b_wav.exists()):
        return {"skip_reason": "missing adapter_A.log / orchestrator.log / B.wav"}

    anchor = get_b_wav_anchor(orch)
    agent_intervals, _ = detect_wav_intervals(b_wav)
    injects, _interrupts, _ma = parse_adapter_a_log(adapter_a)
    transcripts = _parse_agent_transcripts(record_dir / "adapter_B.log", anchor)
    timeouts = _parse_timeouts(adapter_a, anchor)   # clipPlayer 8s-watchdog fires

    # task manifest for per-clip customer audio + durations
    task_id = record_dir.parent.name
    mp = ARTIFACTS_PATH / "customer_clips" / f"{task_id}.json"
    manifest = json.loads(mp.read_text()) if mp.exists() else {"clips": []}
    clip_dur = {}
    for c in manifest.get("clips", []):
        a, b = c.get("audio_start"), c.get("audio_end")
        if a is not None and b is not None:
            clip_dur[c["clip_id"]] = round(b - a, 3)

    rows = []
    seen_clips: Counter = Counter()   # how many times each clip_id injected so far
    for i, inj in enumerate(injects):
        cid = inj["clip_id"]
        if _is_synthetic_clip(cid):
            continue
        is_replay = seen_clips[cid] >= 1   # 2nd+ playing => recovery replay
        seen_clips[cid] += 1
        inj_t = inj["wall"] - anchor
        next_t = (injects[i + 1]["wall"] - anchor) if i + 1 < len(injects) else None
        dur = clip_dur.get(cid)
        cust_intervals, _ = _clip_customer_vad(cid, manifest)
        # turn boundary on the run clock
        last_speech = cust_intervals[-1][1] if cust_intervals else dur
        boundary_t = (inj_t + last_speech) if last_speech is not None else (inj_t + (dur or 0))

        # response window for backchannel/word-count + takeover
        win0 = inj_t + (dur or 0)              # clip end
        win1 = next_t
        segs = [(s, e) for s, e in agent_intervals if (win1 is None or s < win1) and e > win0]
        resp_dur = sum(min(e, win1 if win1 else e) - max(s, win0) for s, e in segs) if segs else 0.0
        responded = resp_dur >= 0.10
        words = _words_in_window(transcripts, win0, win1)
        is_backchannel = responded and resp_dur < BACKCHANNEL_MAX_DURATION_S and words < BACKCHANNEL_MAX_WORDS

        # carryover: agent already speaking AT inject?
        carryover = _agent_state_at(agent_intervals, inj_t) is not None
        # fresh barge-in onset during the clip, before boundary (minus transition tolerance)
        fresh_onset = _first_fresh_onset_in(agent_intervals, inj_t, boundary_t - BARGEIN_TOLERANCE_S)
        # greeting exclusion: with speakFirst, the agent's FIRST speech is the
        # greeting ("Hi, what can I get you?"), not a barge-in. If the onset
        # falls in the agent's earliest speech interval, it's the greeting.
        is_greeting = False
        if fresh_onset is not None and agent_intervals:
            ga, gb = agent_intervals[0]
            if ga <= fresh_onset <= gb:
                is_greeting = True
        is_bargein = (fresh_onset is not None) and (not carryover) and (not is_greeting)

        # margin = how far BEFORE the customer's last word the agent started.
        # >= CLEAR_MARGIN_S  -> clear interruption (agent cut the customer mid-order)
        # <  CLEAR_MARGIN_S  -> marginal/early (customer was trailing off; within the
        #                       natural turn-transition zone — not a clear barge-in)
        margin = round(boundary_t - fresh_onset, 3) if fresh_onset is not None else None
        intr_tier = intr_kind = None

        # classify
        if is_bargein:
            label = "interruption"
            takeover = 1
            # CLEAR requires BOTH: cut in >=1s before the customer's last word
            # (margin) AND started >=2s into the clip (onset) — i.e. let them
            # talk, then barged in. Early-onset ones are demoted to "marginal"
            # (kept for inspection, excluded from the headline rate) because
            # they're typically noise-triggered, not a barge-in of THIS clip.
            onset_into_clip = fresh_onset - inj_t
            is_clear = margin >= CLEAR_MARGIN_S and onset_into_clip >= MIN_BARGEIN_ONSET_S
            intr_tier = "clear" if is_clear else "marginal"
            intr_kind = "recovery" if is_replay else "live"
        elif not responded:
            label = "no_response"
            takeover = 0
        elif is_backchannel:
            label = "backchannel"
            takeover = 0
        else:
            label = "good_take"
            takeover = 1

        # where (only meaningful for interruptions)
        where = None
        if is_bargein and cust_intervals is not None:
            where = _customer_state_at(cust_intervals, fresh_onset - inj_t)

        # turn-taking latency for clean takes (boundary -> onset of response)
        take_latency = None
        if label == "good_take" and segs:
            take_latency = round(min(s for s, e in segs) - boundary_t, 3)

        # did the clipPlayer 8s-watchdog fire during THIS clip's window?
        win_end = next_t if next_t is not None else (inj_t + (dur or 0) + 15)
        timed_out = any(inj_t <= t <= win_end for t in timeouts)

        rows.append({
            "clip_id": cid,
            "label": label,
            "intr_tier": intr_tier,        # clear | marginal (interruptions only)
            "intr_kind": intr_kind,        # live | recovery (interruptions only)
            "interruption_artifact": False,   # set in pass 2 (#1 filter)
            "artifact_reason": None,
            "margin_before_boundary_s": margin,
            "takeover": takeover,
            "responded": responded,
            "timed_out": timed_out,
            "carryover_at_start": carryover,
            "greeting_overlap": is_greeting,
            "bargein_onset_rel_clip_s": round(fresh_onset - inj_t, 3) if fresh_onset is not None else None,
            "customer_boundary_rel_clip_s": round(last_speech, 3) if last_speech is not None else None,
            "where": where,
            "agent_resp_duration_s": round(resp_dur, 3),
            "agent_words": words,
            "take_latency_s": take_latency,
        })

    # ---- PASS 2 (#1 filter): split TRUE interruptions from timing artifacts ----
    # Per-run VAD offset = typical clip-end -> response lag on clean takes
    # (consistent within a run, varies across runs — per Dan).
    _lats = [r["take_latency_s"] for r in rows if r["label"] == "good_take" and r["take_latency_s"] is not None]
    run_offset = round(statistics.median(_lats), 3) if _lats else None
    for i, r in enumerate(rows):
        if r["label"] != "interruption":
            continue
        onset_rel = r["bargein_onset_rel_clip_s"]   # onset relative to clip start
        prev = rows[i - 1] if i > 0 else None
        prev_no_resp = prev is not None and (not prev["responded"] or prev.get("timed_out"))
        # artifact: fired in the very early start of THIS clip AND the previous
        # clip got no response -> this is the delayed response to the previous
        # clip / a timeout collision, not a barge-in of the current clip.
        if onset_rel is not None and onset_rel < EARLY_ONSET_S and prev_no_resp:
            r["label"] = "no_response"       # per Dan: fold into no-response
            # (provenance kept in interruption_artifact=True + artifact_reason)
            r["interruption_artifact"] = True
            r["artifact_reason"] = ("timeout_collision" if (prev and prev.get("timed_out"))
                                    else "responding_to_previous")
            r["takeover"] = 0
            r["intr_tier"] = None
            r["intr_kind"] = None

    n = len(rows)
    def frac(lbl): return round(sum(1 for r in rows if r["label"] == lbl) / n, 4) if n else None
    intr = [r for r in rows if r["label"] == "interruption"]   # TRUE interruptions (post-filter)
    artifacts = [r for r in rows if r["interruption_artifact"]]
    n_clear = sum(1 for r in intr if r["intr_tier"] == "clear" and r["intr_kind"] == "live")
    n_marginal = sum(1 for r in intr if r["intr_tier"] == "marginal")
    n_recovery = sum(1 for r in intr if r["intr_kind"] == "recovery")
    lats = [r["take_latency_s"] for r in rows if r["label"] == "good_take" and r["take_latency_s"] is not None]
    agg = {
        "n_clips": n,
        "run_vad_offset_s": run_offset,        # per-run clip->response lag (#1 calibration)
        # headline: clear, live, TRUE interruptions
        "interruption_rate": round(n_clear / n, 4) if n else None,
        "n_interruptions_clear": n_clear,
        "n_interruptions_marginal": n_marginal,
        "n_interruptions_recovery": n_recovery,
        "n_interruptions_true_total": len(intr),
        # #1 artifacts: reclassified as non-response (timing coincidences)
        "n_artifacts": len(artifacts),
        "artifacts_by_reason": {
            "timeout_collision": sum(1 for r in artifacts if r["artifact_reason"] == "timeout_collision"),
            "responding_to_previous": sum(1 for r in artifacts if r["artifact_reason"] == "responding_to_previous"),
        },
        "good_take_rate": frac("good_take"),
        "backchannel_rate": frac("backchannel"),
        "no_response_rate": frac("no_response"),   # now includes reclassified artifacts
        "takeover_rate": round(sum(r["takeover"] for r in rows) / n, 4) if n else None,
        "median_take_latency_s": round(statistics.median(lats), 3) if lats else None,
        "clear_interruptions_where": {
            "active": sum(1 for r in intr if r["intr_tier"] == "clear" and r["intr_kind"] == "live" and r["where"] == "active"),
            "pause": sum(1 for r in intr if r["intr_tier"] == "clear" and r["intr_kind"] == "live" and r["where"] == "pause"),
        },
    }
    return {"task_id": task_id, "run_ts": record_dir.name, "aggregate": agg, "per_clip": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("record_dir", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    r = analyze(args.record_dir)
    (args.out or (args.record_dir / "turn_taking.json")).write_text(json.dumps(r, indent=2))
    a = r.get("aggregate")
    if a:
        print(f"[turn_taking] {args.record_dir.name}: true_intr={a['n_interruptions_true_total']} "
              f"(clear={a['n_interruptions_clear']} marg={a['n_interruptions_marginal']}) "
              f"ARTIFACTS={a['n_artifacts']} {a['artifacts_by_reason']} "
              f"offset={a['run_vad_offset_s']}s no_resp={a['no_response_rate']}")
    else:
        print(f"[turn_taking] {args.record_dir.name}: {r.get('skip_reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
