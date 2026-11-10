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

"""Resolve the V1 benchmark working-set task_ids for run_order_up_dataset.sh.

Usage: resolve_v1_task_ids.py <bench_md> <artifacts_path> [--offline] [--prefilter]

Every row in the clean manifest is a working-set tid — no status flags to
strip, no profile column to parse. Ordering follows file order. Prints one
task_id per line.

ID SPACE. The manifest names tasks by their original (legacy) ids; the
published dataset renames them, and `task_id_map()` is the bridge. When a
map is available every row is translated to its published id and rows with
no published counterpart are reported on stderr and skipped. When no map
exists (a local checkout with no published dataset pinned) the legacy
ids are emitted unchanged.

PREFILTER is OFF by default, deliberately. It checks that each tid has a
usable decision tree and a complex task config, which is useful with a warm cache
but fatal for a fresh clone: nothing is on disk yet, hydration happens
per-task inside the runner, so filtering on artifact presence drops every
task and the run dies with "No tasks to run". Pass --prefilter for
runs where the artifact cache is already populated.

--offline restricts artifact lookups to the local cache.
"""
import os
import re
import sys
from pathlib import Path

ARGS = sys.argv[3:]
if "--offline" in ARGS:
    os.environ["CLIP_EVAL_OFFLINE"] = "1"
PREFILTER = "--prefilter" in ARGS

from clip_eval.pipeline.artifact_store import (  # noqa: E402
    exists,
    read_json,
    stage_decision_tree_ref,
    stage_task_config_ref,
)


def unevaluable(tid: str) -> bool:
    """Structural unevaluable check — broken decision tree. The clean
    manifest should never contain unevaluable tids in the first place;
    this is a defense-in-depth guard for edit accidents."""
    ref = stage_decision_tree_ref(tid)
    try:
        t = read_json(ref)
    except Exception:
        return True
    init = t.get("initial_node")
    if not init:
        return True
    if init not in (t.get("nodes") or {}):
        return True
    if not init.endswith("_main") and "_main" not in init:
        return True
    return False


def _published_by_legacy() -> dict:
    """legacy_task_id -> published task_id, or {} when no map is available.

    A missing map is not an error: fall back to
    emitting the manifest's own ids.
    """
    try:
        from clip_eval.pipeline.artifact_store import task_id_map
    except ImportError:
        return {}
    try:
        return {v["legacy_task_id"]: k for k, v in task_id_map().items()}
    except Exception as e:  # noqa: BLE001 - absent/unreachable map is fine
        print(
            f"[resolve_v1_task_ids] no task id map ({e}); "
            "emitting manifest ids unchanged",
            file=sys.stderr,
        )
        return {}


def main() -> None:
    bench_md = Path(sys.argv[1])
    # sys.argv[2] (artifacts_path) is unused: lookups go through the
    # artifact store, which reads ARTIFACTS_PATH itself. Kept as a
    # positional for call-site compatibility.

    legacy_ids: list[str] = []
    seen: set[str] = set()
    for line in bench_md.read_text().splitlines():
        m = re.match(r"^\| `(order_up\.[^`]+)` \|", line)
        if not m:
            continue
        tid = m.group(1)
        if tid in seen:
            continue
        seen.add(tid)
        if not re.search(r"\.s\d+$", tid):  # per-stage rows only
            continue
        legacy_ids.append(tid)

    published_by_legacy = _published_by_legacy()
    missing: list[str] = []
    out: list[str] = []
    for legacy in legacy_ids:
        if published_by_legacy:
            tid = published_by_legacy.get(legacy)
            if tid is None:
                missing.append(legacy)
                continue
        else:
            tid = legacy
        out.append(tid)

    if PREFILTER:
        kept = []
        for tid in out:
            if unevaluable(tid):
                continue
            if not exists(stage_task_config_ref(tid, complex=True)):
                continue
            kept.append(tid)
        dropped = len(out) - len(kept)
        if dropped:
            print(
                f"[resolve_v1_task_ids] --prefilter dropped {dropped} tid(s) "
                "with no usable tree or complex task config",
                file=sys.stderr,
            )
        out = kept

    for tid in out:
        print(tid)

    if missing:
        shown = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        print(
            f"[resolve_v1_task_ids] WARNING: {len(missing)} manifest row(s) have "
            f"no published task and were skipped: {shown}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
