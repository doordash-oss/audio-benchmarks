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

"""Post-eval root-cause for `missing-item` runs.

Joins the clipPlayer timeout log against the missing-item set to
identify a specific sub-class: ``missing-item-not-detected``. This
fires when a missing item's MAIN CLIP was played and the next thing
to happen was a clipPlayer timeout — meaning the agent never
responded to the line that introduced the missing item, and the
clipPlayer had to force-kick the session forward.

That pattern attributes the miss to the agent's silence, not to a
mis-routed clip or a hallucinated cart action.

Pipeline:
  1. Triggers only on runs tagged ``missing-item``.
  2. Requires ``clipplayer_timeout_audit.json`` next to score.json
     (run that audit first, or pass --auto to run it inline).
  3. For each missing expected_cart item, finds its main clip_id by
     position within its source stage's expected items
     (``g{gi}_s{si}_i{idx}_main``).
  4. Marks the item ``not_detected`` when any timeout entry's
     ``anchor_clip_id`` matches that main clip_id.
  5. Emits a sidecar at ``<record_dir>/missing_item_root_cause.json``
     and patches ``score.json`` to add the
     ``missing-item-not-detected`` tag when any item triggered.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from clip_eval.pipeline.paths import ARTIFACTS_PATH
from clip_eval.scoring.clipplayer_timeout_audit import (
    run_audit as run_timeout_audit,
)

TRIGGER_TAG = "missing-item"
NEW_TAG = "missing-item-not-detected"


def _stage_items_in_order(expected_cart: list[dict], stage_index: int) -> list[dict]:
    """Items in the run's expected_cart whose source_stage_index matches,
    preserving the cart's listed order. The clip-builder writes
    ``g{gi}_s{si}_i{idx}_*`` using THIS ordering, so the index here is
    the same index that appears in clip_ids."""
    return [
        e for e in expected_cart
        if e.get("source_stage_index") == stage_index
    ]


def _main_clip_id(game_index: int, stage_index: int, item_index: int) -> str:
    return f"g{game_index}_s{stage_index}_i{item_index}_main"


def _item_clip_prefix(game_index: int, stage_index: int, item_index: int) -> str:
    """Prefix shared by every clip that introduces or specifies one
    expected item: ``g{gi}_s{si}_i{idx}_`` — covers ``_main``, ``_opt*``,
    ``_restatement*``, ``_main_corr_*``, etc. Used to widen the timeout
    match so a clipPlayer timeout at ANY of the item's clips (not just
    its main) counts as "agent silent on this item."
    """
    return f"g{game_index}_s{stage_index}_i{item_index}_"


def _parse_tid(task_id: str) -> tuple[int, int] | None:
    """Per-stage tid → (game_index, stage_index). Returns None for
    per-game tids (which the audit can't reason about without further
    context)."""
    parts = task_id.split(".")
    if len(parts) < 5:
        return None
    g_part, s_part = parts[-2], parts[-1]
    if not (g_part.startswith("g") and s_part.startswith("s")):
        return None
    try:
        return int(g_part[1:]), int(s_part[1:])
    except ValueError:
        return None


def run_audit(record_dir: Path, *, auto_timeout: bool = False,
              patch_tags: bool = True) -> dict:
    out: dict = {
        "record_dir": str(record_dir),
        "trigger_tag": TRIGGER_TAG,
        "new_tag": NEW_TAG,
        "judged": False,
        "skip_reason": None,
        "any_not_detected": False,
        "items": [],
    }
    sp = record_dir / "score.json"
    if not sp.exists():
        out["skip_reason"] = "no score.json"
        return out
    score = json.loads(sp.read_text())
    tags = list(score.get("tags") or [])
    out["tags_before"] = list(tags)  # snapshot — patching below mutates `tags`
    if TRIGGER_TAG not in tags:
        out["skip_reason"] = f"tag {TRIGGER_TAG!r} not on this run"
        return out
    task_id = score.get("task_id") or record_dir.parent.name
    gs = _parse_tid(task_id)
    if gs is None:
        out["skip_reason"] = f"task_id {task_id!r} is not per-stage shape"
        return out
    game_index, _stage_default = gs

    # Load timeout audit (run it now if requested + missing).
    ta_path = record_dir / "clipplayer_timeout_audit.json"
    if not ta_path.exists():
        if auto_timeout:
            ta = run_timeout_audit(record_dir)
            ta_path.write_text(json.dumps(ta, indent=2))
        else:
            out["skip_reason"] = (
                "no clipplayer_timeout_audit.json — run "
                "clipplayer_timeout_audit first or pass --auto"
            )
            return out
    else:
        ta = json.loads(ta_path.read_text())
    timeout_anchors = {
        t.get("anchor_clip_id")
        for t in (ta.get("timeouts") or [])
        if t.get("anchor_clip_id")
    }
    out["n_timeouts"] = ta.get("n_timeouts") or 0
    out["timeout_anchors"] = sorted(timeout_anchors)

    expected_cart = list(score.get("expected_cart") or [])
    missing_set: list[dict] = (
        (score.get("diff") or {}).get("missing") or []
    )
    if not missing_set:
        # The tag claims a missing item but the diff is empty — should
        # not happen for well-formed score.json, but bail safely.
        out["skip_reason"] = "no missing entries in diff despite tag"
        return out

    # Index expected_cart per stage so we can look up item_index by item_id.
    # source_stage_index is per-game; for per-stage tids it matches the tid's
    # stage segment (i.e. the only stage in the cart).
    per_item: list[dict] = []
    any_hit = False
    for miss in missing_set:
        item_id = miss.get("item_id")
        if not item_id:
            continue
        # Find the matching expected entry to read source_stage_index.
        matched_expected = next(
            (e for e in expected_cart if e.get("item_id") == item_id),
            None,
        )
        if matched_expected is None:
            per_item.append({
                "item_id": item_id,
                "skip_reason": "item not found in expected_cart",
            })
            continue
        si = matched_expected.get("source_stage_index")
        if si is None:
            per_item.append({
                "item_id": item_id,
                "skip_reason": "expected entry has no source_stage_index",
            })
            continue
        stage_items = _stage_items_in_order(expected_cart, si)
        try:
            item_index = next(
                i for i, e in enumerate(stage_items)
                if e.get("item_id") == item_id
            )
        except StopIteration:
            per_item.append({
                "item_id": item_id,
                "skip_reason": "item not found in stage_items",
            })
            continue
        main_clip_id = _main_clip_id(game_index, si, item_index)
        item_prefix = _item_clip_prefix(game_index, si, item_index)
        # Match ANY clip belonging to this item: main, opt*, restatement*,
        # main_corr_*. Same agent-silent failure mode regardless of which
        # clip kind the timeout anchored at.
        item_timeout_anchors = sorted(
            a for a in timeout_anchors if a and a.startswith(item_prefix)
        )
        not_detected = bool(item_timeout_anchors)
        main_timed_out = main_clip_id in timeout_anchors
        if not_detected:
            any_hit = True
        per_item.append({
            "item_id": item_id,
            "stage_index": si,
            "item_index": item_index,
            "main_clip_id": main_clip_id,
            "main_clip_timed_out": bool(main_timed_out),
            "timed_out_clip_ids": item_timeout_anchors,
            "not_detected": bool(not_detected),
        })

    out["judged"] = True
    out["items"] = per_item
    out["any_not_detected"] = bool(any_hit)

    # Patch score.json's tags list. Add the new tag, never remove. This
    # mirrors how `missing-answer-missing-item` is attached by the
    # cart scorer — once a tag is on a run, downstream tools (the
    # dashboard's compare block, the off-policy audit) read tags from
    # score.json as the source of truth.
    if patch_tags and any_hit and NEW_TAG not in tags:
        tags.append(NEW_TAG)
        score["tags"] = tags
        sp.write_text(json.dumps(score, indent=2))
        out["tags_patched"] = True
    else:
        out["tags_patched"] = False
    out["tags_after"] = tags
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("record_dir", type=Path)
    ap.add_argument("--auto", action="store_true",
                    help="run clipplayer_timeout_audit inline if its "
                         "sidecar is missing (default: skip)")
    ap.add_argument("--no-patch", action="store_true",
                    help="don't add the new tag to score.json — just "
                         "write the sidecar")
    ap.add_argument("--out", type=Path,
                    help="output path (default: "
                         "<record_dir>/missing_item_root_cause.json)")
    args = ap.parse_args()

    out = run_audit(
        args.record_dir,
        auto_timeout=args.auto,
        patch_tags=not args.no_patch,
    )
    out_path = args.out or (args.record_dir / "missing_item_root_cause.json")
    out_path.write_text(json.dumps(out, indent=2))
    if not out.get("judged"):
        print(f"[missing_item_root_cause] {args.record_dir.name}: "
              f"skipped ({out.get('skip_reason')})")
    else:
        n_hits = sum(1 for i in out["items"] if i.get("not_detected"))
        n_items = len(out["items"])
        verdict = (
            "NOT_DETECTED" if out["any_not_detected"]
            else "no-not-detected"
        )
        patched = " (tag added to score.json)" if out.get("tags_patched") else ""
        print(f"[missing_item_root_cause] {args.record_dir.name}: "
              f"{verdict}  ({n_hits}/{n_items} missing items had a main-clip "
              f"timeout){patched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
