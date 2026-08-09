"""Post-eval check: how often did the clipPlayer have to force-kick
the agent because it stopped responding?

The clipPlayer (role A) arms an 8-second watchdog after every clip
inject (``clipPlayer_adapter.js:673``). If the agent hasn't started
speaking within that window, A logs

    [ClipPlayer] No agent speech within 8000ms after clip — firing routing.

and force-creates a response on B's session.

This isn't a correctness signal — a run can be 1.0 score with
multiple timeouts ("slow but correct"). It is a latency-stress signal
that tells us how confidently the agent kept the conversation moving.

Each detected timeout is anchored to the most-recent
``[ClipPlayer] inject_start <clip_id>`` line so the timeout is tied
to a specific clip. Writes ``clipplayer_timeout_audit.json`` next to
``score.json``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

LOG_TS_RE = re.compile(
    r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z)\] "
)
TIMEOUT_RE = re.compile(r"\[ClipPlayer\] No agent speech within (\d+)ms after clip")
INJECT_START_RE = re.compile(r"\[ClipPlayer\] inject_start (\S+)")


def _parse_iso(s: str) -> float:
    """ISO-Z → epoch seconds."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def run_audit(record_dir: Path) -> dict:
    """Walk adapter_A.log, return one entry per timeout line. Anchor
    each timeout to the most recent ``inject_start`` clip_id seen. The
    log-anchored timestamps get normalized to ``t_rel_s`` against the
    FIRST log line so timings line up with the rest of the eval
    artifacts (timeline.json's events use the same convention)."""
    out: dict = {
        "record_dir": str(record_dir),
        "n_timeouts": 0,
        "timeouts": [],
        "skip_reason": None,
    }
    log = record_dir / "adapter_A.log"
    if not log.exists():
        out["skip_reason"] = "no adapter_A.log"
        return out

    first_ts: float | None = None
    last_inject_clip_id: str | None = None
    last_inject_t: float | None = None
    timeouts: list[dict] = []

    for line in log.read_text(errors="replace").splitlines():
        m_ts = LOG_TS_RE.match(line)
        if not m_ts:
            continue
        t = _parse_iso(m_ts.group(1))
        if first_ts is None:
            first_ts = t

        m_inj = INJECT_START_RE.search(line)
        if m_inj:
            last_inject_clip_id = m_inj.group(1)
            last_inject_t = t - first_ts
            continue

        m_to = TIMEOUT_RE.search(line)
        if m_to:
            t_rel = t - first_ts
            timeouts.append({
                "t_rel_s": round(t_rel, 3),
                "timeout_ms": int(m_to.group(1)),
                "anchor_clip_id": last_inject_clip_id,
                "anchor_clip_t_rel_s": (
                    round(last_inject_t, 3) if last_inject_t is not None else None
                ),
                "wait_since_clip_s": (
                    round(t_rel - last_inject_t, 3)
                    if last_inject_t is not None else None
                ),
            })

    out["n_timeouts"] = len(timeouts)
    out["timeouts"] = timeouts
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("record_dir", type=Path)
    ap.add_argument("--out", type=Path,
                    help="output path (default: "
                         "<record_dir>/clipplayer_timeout_audit.json)")
    args = ap.parse_args()
    out = run_audit(args.record_dir)
    out_path = args.out or (args.record_dir / "clipplayer_timeout_audit.json")
    out_path.write_text(json.dumps(out, indent=2))
    if out.get("skip_reason"):
        print(f"[clipplayer_timeout] {args.record_dir.name}: skipped "
              f"({out['skip_reason']}) → {out_path}")
    else:
        n = out["n_timeouts"]
        print(f"[clipplayer_timeout] {args.record_dir.name}: "
              f"{n} timeout(s) → {out_path}")
        for t in (out.get("timeouts") or [])[:3]:
            print(f"    t_rel={t['t_rel_s']}s  after clip "
                  f"{t['anchor_clip_id']} (waited {t['wait_since_clip_s']}s)")
        if n > 3:
            print(f"    ... +{n - 3} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
