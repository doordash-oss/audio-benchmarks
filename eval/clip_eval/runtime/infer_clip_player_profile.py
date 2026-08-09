"""Infer the clipPlayer profile from a task's customer_clips manifest.

Rule: if any `clip_role == "main"` clip has customer_spans whose union of
`item_indices` covers 2+ distinct items, this is a compound-main stage
and needs the "restate-on-partial-ack" profile — it teaches the
clipPlayer to route to a restatement when the agent partial-acks a
multi-item utterance. Anything else uses "default".

Called from `run_order_up_eval.sh` when the caller hasn't passed
`--clip-player-profile`, so the profile is auto-selected per-tid from
the manifest built by the pipeline. The CLI arg still wins if given.
"""
from __future__ import annotations

import json
import sys

from clip_eval.pipeline.paths import CUSTOMER_CLIPS_DIR


def infer_profile(task_id: str) -> str:
    p = CUSTOMER_CLIPS_DIR / f"{task_id}.json"
    if not p.exists():
        return "default"
    try:
        m = json.loads(p.read_text())
    except Exception:
        return "default"
    for c in m.get("clips") or []:
        if c.get("clip_role") != "main":
            continue
        items: set[int] = set()
        for sp in c.get("customer_spans") or []:
            for i in sp.get("item_indices") or []:
                if i is not None:
                    items.add(int(i))
        if len(items) >= 2:
            return "restate-on-partial-ack"
    return "default"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m clip_eval.runtime.infer_clip_player_profile <task_id>", file=sys.stderr)
        sys.exit(2)
    print(infer_profile(sys.argv[1]))
