#!/usr/bin/env python3
"""Cross-run turn-taking breakdowns + dashboard summary.

Reads the per-run turn_taking.json for the latest complex run per (task, model)
and rolls up:

  #2  no-response rate BUCKETED BY CLIP LENGTH (clip audio duration). Folds the
      timeout-collision / responding-to-previous artifacts into no-response (they
      are already relabeled "no_response" by the analyzer's pass-2 filter, with
      provenance kept in interruption_artifact / artifact_reason).

  + the full per-model summary the /compare dashboard reads (interruption rate,
    no-response, backchannel, takeover, latency).

All deterministic; no LLM. Run: python -m clip_eval.scoring.turn_taking_breakdowns
"""
from __future__ import annotations
import json, sys
from pathlib import Path

from clip_eval.scoring.turn_taking_analysis import ARTIFACTS_PATH

# clip-length buckets (seconds of clip audio)
LEN_BUCKETS = [("<0.5s", 0.0, 0.5), ("0.5-1.5s", 0.5, 1.5),
               ("1.5-3s", 1.5, 3.0), ("3s+", 3.0, float("inf"))]


def _clip_durations(task_id: str) -> dict:
    mp = ARTIFACTS_PATH / "customer_clips" / f"{task_id}.json"
    if not mp.exists():
        return {}
    out = {}
    for c in json.loads(mp.read_text()).get("clips", []):
        a, b = c.get("audio_start"), c.get("audio_end")
        if a is not None and b is not None:
            out[c["clip_id"]] = round(b - a, 3)
    return out


def _bucket(dur):
    if dur is None:
        return None
    for name, lo, hi in LEN_BUCKETS:
        if lo <= dur < hi:
            return name
    return None


def build(final_runs: dict) -> dict:
    """#2: no-response rate by clip length, per model."""
    nr = {p: {b[0]: [0, 0] for b in LEN_BUCKETS} for p in ("openai", "gemini", "cascade", "grok")}
    for key, rd in final_runs.items():
        tid, prov = key.split("|")
        tt = Path(rd) / "turn_taking.json"
        if not tt.exists():
            continue
        data = json.loads(tt.read_text())
        durs = _clip_durations(tid)
        for r in data.get("per_clip", []):
            b = _bucket(durs.get(r["clip_id"]))
            if b is None:
                continue
            nr[prov][b][0] += 1
            if r["label"] == "no_response":        # includes folded artifacts
                nr[prov][b][1] += 1
    no_response_by_len = {}
    for prov, buckets in nr.items():
        no_response_by_len[prov] = {
            name: {"clips": c, "no_response": n,
                   "rate_pct": round(100 * n / c, 1) if c else None}
            for name, (c, n) in buckets.items()
        }
    return {
        "no_response_by_clip_len": no_response_by_len,
        "len_buckets": [b[0] for b in LEN_BUCKETS],
    }


def build_summary(final_runs: dict) -> dict:
    """Full per-model turn-taking summary for the dashboard (/api/turn_taking).
    Rebuilt from current deterministic turn_taking.json — no human verification.
    Interruptions = CLEAR tier (margin>=1.0 AND onset>=2.0). We do NOT split
    interruptions by run-success: the metric stands alone, uncorrelated with
    score."""
    import statistics
    provs = ("openai", "gemini", "cascade", "grok")
    acc = {p: {"clips": 0, "takeover": 0, "no_response": 0, "backchannel": 0,
               "good_take": 0, "lats": [], "clear": 0}
           for p in provs}
    for key, rd in final_runs.items():
        tid, prov = key.split("|")
        a = acc[prov]
        d = json.loads((Path(rd) / "turn_taking.json").read_text())
        for r in d.get("per_clip", []):
            a["clips"] += 1
            a["takeover"] += r.get("takeover", 0)
            lbl = r["label"]
            if lbl in ("no_response", "backchannel", "good_take"):
                a[lbl] += 1
            if lbl == "good_take" and r.get("take_latency_s") is not None:
                a["lats"].append(r["take_latency_s"])
            if lbl == "interruption" and r.get("intr_tier") == "clear" and r.get("intr_kind") == "live":
                a["clear"] += 1
    by_model = {}
    for p in provs:
        a = acc[p]; n = a["clips"] or 1
        by_model[p] = {
            "clips": a["clips"],
            "takeover_pct": round(100 * a["takeover"] / n, 1),
            "no_response_pct": round(100 * a["no_response"] / n, 1),
            "backchannel_pct": round(100 * a["backchannel"] / n, 1),
            "good_take_pct": round(100 * a["good_take"] / n, 1),
            "median_latency_s": round(statistics.median(a["lats"]), 2) if a["lats"] else None,
            "interruptions_clear": a["clear"],
            "interruption_pct": round(100 * a["clear"] / n, 1),
        }
    bd = build(final_runs)
    return {
        "models": list(provs),
        "note": "fully deterministic (VAD + logs); no human verification",
        "by_model": by_model,
        "no_response_by_clip_len": bd["no_response_by_clip_len"],
        "len_buckets": bd["len_buckets"],
        "methodology": {
            "deterministic": "RMS-VAD on agent B.wav + customer clip audio, agent transcript, clip-player logs. No LLM, no human review.",
            "interruption": "agent starts a FRESH utterance before the customer's last word; CLEAR tier requires margin>=1.0s before that word AND onset>=2.0s into the clip (agent let the customer talk, then cut in). Excludes greeting / carryover / re-inject / <0.25s transition; early-onset overlaps after a no-response are reclassified as delayed-response artifacts.",
            "validation": "onset>=2.0 gate calibrated on 30 hand-listened candidates: 100% recall of real interruptions, ~78% precision (4 residuals have timing identical to real ones).",
            "thresholds": "FDB arXiv:2503.04721 — transition 0.25s, natural pause 0.4-1.0s, backchannel <1s & <2 words",
        },
    }


def _print(res):
    print("\n=== #2  No-response rate by clip length (folds timeout artifacts) ===")
    print("model    " + "".join(f"{b:>14}" for b in res["len_buckets"]))
    for prov, row in res["no_response_by_clip_len"].items():
        cells = ""
        for b in res["len_buckets"]:
            d = row[b]
            cells += f"{(str(d['rate_pct'])+'% ('+str(d['no_response'])+'/'+str(d['clips'])+')'):>14}"
        print(f"{prov:8} {cells}")


if __name__ == "__main__":
    final = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/final_runs.json"))
    res = build(final)
    (ARTIFACTS_PATH / "turn_taking_breakdowns.json").write_text(json.dumps(res, indent=2))
    _print(res)
    summ = build_summary(final)
    (ARTIFACTS_PATH / "turn_taking_summary.json").write_text(json.dumps(summ, indent=2))
    print("\n=== #1 interruption summary (rebuilt, deterministic) ===")
    for m, b in summ["by_model"].items():
        print(f"  {m:8} clips={b['clips']:>3} interrupt={b['interruptions_clear']:>2} ({b['interruption_pct']}%) "
              f"no_resp={b['no_response_pct']}% takeover={b['takeover_pct']}% lat={b['median_latency_s']}s")
    print(f"\nwrote {ARTIFACTS_PATH/'turn_taking_summary.json'} and turn_taking_breakdowns.json")
