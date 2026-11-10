#!/usr/bin/env python3
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

"""Postprocess step for run_order_up_eval.sh.

Runs eval_metrics, every cart-error audit, and turn_taking_analysis
in-process (rather than as a bash loop over separate `-m` subprocess
calls), writing each step's usual sidecar json next to score.json, and
finally prints one consolidated report covering the cart score and
every audit's outcome.

Usage: order_up_eval_report.py <record_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clip_eval.scoring import eval_metrics, turn_taking_analysis
from clip_eval.scoring.cart_error_audit import (
    missing_item_intent_audit,
    missing_item_root_cause,
    off_policy_audit,
    preemptive_tool_audit,
    wrong_option_audit,
)

# Order matters:
#   1. preemptive_tool_audit promotes extra-item → wrong-item via the
#      silence check; later audits read the updated tag set.
#   2. wrong_option_audit reads the wrong-option-or-variant pairs the
#      scorer surfaced (no dependency on #1 but cheap to keep close).
#   3. missing_item_root_cause runs clipplayer_timeout_audit via
#      auto_timeout and adds the `missing-item-not-detected` sub-tag; it
#      MUST land before #4 so the intent audit can read that tag when
#      classifying.
#   4. missing_item_intent_audit classifies the remaining missing items
#      (premature_submit / compound_line / clarify_abandoned / other)
#      for dashboard breakdown rollup.
AUDITS = [
    ("preemptive_tool_audit", preemptive_tool_audit, {}),
    ("wrong_option_audit", wrong_option_audit, {}),
    ("missing_item_root_cause", missing_item_root_cause, {"auto_timeout": True}),
    ("missing_item_intent_audit", missing_item_intent_audit, {}),
    ("off_policy_audit", off_policy_audit, {}),
]


def run_step(name: str, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        print(f"[report] {name} failed (non-fatal): {e}", file=sys.stderr)
        return None, str(e)


def print_report(record_dir: Path, score: dict, metrics: dict | None,
                  audit_results: dict, turn_taking: dict | None) -> None:
    diff = score.get("diff") or {}
    print(f"\n[report] {record_dir}")
    print(f"  task_id       = {score.get('task_id')}")
    print(f"  task_score    = {score.get('Task-specific score')}")
    print(f"  matched       = {diff.get('display_matched')}")
    print(f"  submitted     = {score.get('submitted')}  finalize_reason={score.get('finalize_reason')}")
    tags = score.get("tags") or []
    print(f"  tags          = {tags if tags else '(none)'}")

    if metrics is not None:
        agg = metrics.get("aggregate") or {}
        print(f"  eval_metrics  = response_rate={agg.get('response_rate')} n_clips={metrics.get('n_clips')}")
    else:
        print("  eval_metrics  = FAILED")

    if turn_taking is not None:
        agg = turn_taking.get("aggregate")
        if agg:
            print(f"  turn_taking   = true_intr={agg.get('n_interruptions_true_total')} "
                  f"no_resp={agg.get('no_response_rate')}")
        else:
            print(f"  turn_taking   = {turn_taking.get('skip_reason')}")
    else:
        print("  turn_taking   = FAILED")

    print("  audits:")
    for name, (result, err) in audit_results.items():
        if err is not None:
            print(f"    {name:28s} FAILED: {err}")
        elif result.get("skip_reason"):
            print(f"    {name:28s} skipped ({result['skip_reason']})")
        elif result.get("judged"):
            n = len(result.get("items") or result.get("pairs") or result.get("asks") or [])
            print(f"    {name:28s} judged ({n} item(s))")
        else:
            print(f"    {name:28s} ran (no judged flag)")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("record_dir", type=Path)
    args = ap.parse_args()
    record_dir = args.record_dir

    score_path = record_dir / "score.json"
    if not score_path.exists():
        print(f"[report] no score.json in {record_dir}", file=sys.stderr)
        return 1
    score = json.loads(score_path.read_text())

    metrics, _ = run_step("eval_metrics", eval_metrics.analyze_run, record_dir)
    if metrics is not None:
        (record_dir / "eval_metrics.json").write_text(json.dumps(metrics, indent=2))

    audit_results: dict[str, tuple[dict | None, str | None]] = {}
    for name, module, kwargs in AUDITS:
        result, err = run_step(name, module.run_audit, record_dir, **kwargs)
        if result is not None:
            (record_dir / f"{name}.json").write_text(json.dumps(result, indent=2))
        audit_results[name] = (result, err)

    turn_taking, _ = run_step("turn_taking_analysis", turn_taking_analysis.analyze, record_dir)
    if turn_taking is not None:
        (record_dir / "turn_taking.json").write_text(json.dumps(turn_taking, indent=2))

    print_report(record_dir, score, metrics, audit_results, turn_taking)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
