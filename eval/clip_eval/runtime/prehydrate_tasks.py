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

"""Hydrate every task in a sweep up front, in one pass over the dataset.

Without this, each worker hydrates its own task inside
``order_up_eval_setup.py``, which calls ``cache_pull_stage(tid)``. That
resolves a single id by streaming the dataset and linear-scanning until it
matches — measured at ~75s per task on a cold cache, so a 248-task sweep
spends hours re-downloading the same shards. ``cache_pull_stages`` reads the
parquet shards once instead (measured ~12s for 5 tasks cold, including the
full shard download), and the per-worker calls then find a warm cache and
return immediately.

Lives in its own file rather than a heredoc in the shell wrapper, matching
``resolve_v1_task_ids.py`` — see commit a44facc, which removed a multiline
Python call embedded in a shell command substitution.

Usage: prehydrate_tasks.py <task_ids_file>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    tids = [
        line.strip()
        for line in Path(sys.argv[1]).read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not tids:
        return 0

    from clip_eval.pipeline.artifact_store import (
        cache_is_current, cache_pull_stages, hf_repo_id, hf_revision,
    )

    stale = [t for t in tids if not cache_is_current(t)]
    if not stale:
        print(f"[prehydrate] {len(tids)} task(s) already cached at "
              f"{hf_revision()[:12]}…")
        return 0

    print(f"[prehydrate] {len(stale)} of {len(tids)} task(s) need hydrating "
          f"from {hf_repo_id()} @ {hf_revision()[:12]}…", flush=True)
    t0 = time.time()
    n = cache_pull_stages(tids)
    print(f"[prehydrate] {n} task(s) ready in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
