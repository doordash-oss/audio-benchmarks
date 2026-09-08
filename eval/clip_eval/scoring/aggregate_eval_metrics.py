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

"""Cross-run aggregator for eval_metrics.json.

Walks every per-run eval_metrics.json (default: every Order Up run
under artifacts/outputs/) and produces a combined report: overall,
by-task, and the same bucket dimensions the per-run file carries.

CLI:
    python -m clip_eval.scoring.aggregate_eval_metrics
    python -m clip_eval.scoring.aggregate_eval_metrics --out /tmp/aggregate.json
    python -m clip_eval.scoring.aggregate_eval_metrics --glob 'artifacts/outputs/OrderUp/order_up.The_Rustic_Table.*/*/eval_metrics.json'
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

from clip_eval.scoring.eval_metrics import (
    LONG_PAUSE_THRESHOLD_S,
    SHORT_CLIP_THRESHOLD_S,
)


def _stats(rows: list[dict]) -> dict:
    """Compute n + rates + medians over a list of per-clip rows."""
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
    latencies = [
        r["agent_response_latency_s"] for r in rows
        if r["agent_response_latency_s"] is not None
    ]
    durations = [r["agent_response_duration_s"] for r in rows if r["responded"]]
    return {
        "n": n,
        "response_rate": round(responded / n, 4),
        "interrupt_rate": round(interrupted / n, 4),
        "median_latency_s": round(median(latencies), 3) if latencies else None,
        "median_duration_s": round(median(durations), 3) if durations else None,
    }


def _bucketize(rows: list[dict]) -> dict:
    """Produce the same bucket dimensions as the per-run file —
    but accumulated across every clip in every run."""
    out: dict[str, dict] = {
        "by_clip_duration_lt_1_5s": {
            "true": _stats(
                [r for r in rows if (r["clip_duration_s"] or 0) < SHORT_CLIP_THRESHOLD_S]
            ),
            "false": _stats(
                [r for r in rows if (r["clip_duration_s"] or 0) >= SHORT_CLIP_THRESHOLD_S]
            ),
        },
        "by_has_muted_agent_block": {
            "true": _stats(
                [r for r in rows if r["clip_props"]["has_muted_intervals"]]
            ),
            "false": _stats(
                [r for r in rows if not r["clip_props"]["has_muted_intervals"]]
            ),
        },
        "by_long_intra_clip_pause": {
            "true": _stats(
                [r for r in rows if r["clip_props"]["max_inter_customer_gap_s"]
                                    >= LONG_PAUSE_THRESHOLD_S]
            ),
            "false": _stats(
                [r for r in rows if r["clip_props"]["max_inter_customer_gap_s"]
                                    < LONG_PAUSE_THRESHOLD_S]
            ),
        },
        "by_agent_asked_unanswerable": {
            "true": _stats(
                [r for r in rows if r.get("agent_asked_unanswerable")]
            ),
            "false": _stats(
                [r for r in rows if not r.get("agent_asked_unanswerable")]
            ),
        },
        "by_clip_role": {},
    }
    for role in ("main", "corr", "opt"):
        out["by_clip_role"][role] = _stats(
            [r for r in rows if r["clip_props"]["clip_role"] == role]
        )
    return out


def _fmt_stats(s: dict) -> str:
    """Format a stats row as a single Markdown table cell value."""
    if s["n"] == 0:
        return "n=0"
    return (
        f"n={s['n']} resp={s['response_rate']} "
        f"int={s['interrupt_rate']} "
        f"med_lat={s['median_latency_s']} med_dur={s['median_duration_s']}"
    )


def _print_md_table(title: str, rows: list[tuple[str, dict]]):
    print(f"\n### {title}\n")
    print("| group | n | response_rate | interrupt_rate | median_latency_s | median_duration_s |")
    print("|---|---|---|---|---|---|")
    for label, s in rows:
        print(
            f"| {label} | {s['n']} | {s['response_rate']} | "
            f"{s['interrupt_rate']} | {s['median_latency_s']} | "
            f"{s['median_duration_s']} |"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob",
        default="artifacts/outputs/OrderUp/*/*/eval_metrics.json",
        help="Glob pattern (relative to v2/) for per-run metric files.",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="Write the aggregate JSON here too (default: stdout-only).",
    )
    args = ap.parse_args()

    files = sorted(Path(".").glob(args.glob))
    if not files:
        print(f"no files matched: {args.glob}", file=sys.stderr)
        return 1

    all_clips: list[dict] = []
    by_task: dict[str, list[dict]] = defaultdict(list)
    n_runs = 0
    for p in files:
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            print(f"  skip {p}: {e}", file=sys.stderr)
            continue
        n_runs += 1
        task_id = d.get("task_id")
        clips = [c for c in (d.get("per_clip") or []) if not c.get("synthetic")]
        all_clips.extend(clips)
        by_task[task_id].extend(clips)

    overall = _stats(all_clips)
    buckets = _bucketize(all_clips)
    task_stats = {tid: _stats(rows) for tid, rows in sorted(by_task.items())}

    # ── Stdout report ─────────────────────────────────────────────────
    print(f"# Eval metrics aggregate ({n_runs} runs, {len(all_clips)} real clips)\n")
    _print_md_table("Overall", [("overall", overall)])
    _print_md_table(
        "By task",
        [(t, s) for t, s in task_stats.items()],
    )
    for bucket_name, groups in buckets.items():
        _print_md_table(
            bucket_name,
            [(g, s) for g, s in groups.items()],
        )

    if args.out:
        args.out.write_text(json.dumps({
            "n_runs": n_runs,
            "n_clips": len(all_clips),
            "overall": overall,
            "by_task": task_stats,
            "buckets": buckets,
        }, indent=2))
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
