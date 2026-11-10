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

"""Guard: a shared clip the router can play must be registered for hydration.

A missing shared clip is an invisible failure: `availableClipBasename` returns
null for a missing file and the choice is dropped from `valid_choices`, so the
model is simply never offered the response and nothing errors.

The mechanism is that the router names a basename and nothing forces that name
into `SHARED_CLIP_NAMES`. So this reads the basenames straight out of the
router and requires each one to be registered.

Run: python -m clip_eval.tests.test_shared_clip_pool
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DECISION_TREE_JS = REPO_ROOT / "clip_eval" / "role_a" / "decision_tree.js"

# Any single-quoted `*.wav` literal in the router is a basename it can hand to
# resolveClipPath. Catches both `availableClipBasename('x.wav')` and the
# `clip: 'thats_all.wav'` node synthesized by _ensureSharedThatsAllNode, which
# the narrower call-site pattern misses.
WAV_LITERAL = re.compile(r"'([A-Za-z0-9_.-]+\.wav)'")


def _shared_clip_names() -> tuple[str, ...]:
    sys.path.insert(0, str(REPO_ROOT))
    from clip_eval.pipeline.artifact_store import SHARED_CLIP_NAMES

    return SHARED_CLIP_NAMES


def test_router_basenames_are_registered() -> None:
    names = set(_shared_clip_names())
    found = set(WAV_LITERAL.findall(DECISION_TREE_JS.read_text()))
    unregistered = sorted(found - names)
    assert not unregistered, (
        f"{DECISION_TREE_JS.name} can inject {unregistered}, which is not in "
        f"SHARED_CLIP_NAMES. Add it there (and to the internal build "
        f"pipeline's tuple + TTS builder) or hydration will not fetch it and "
        f"the response will silently vanish from valid_choices."
    )


def test_literal_matches_tuple() -> None:
    """The typing Literal and the runtime tuple must not drift apart."""
    from typing import get_args

    sys.path.insert(0, str(REPO_ROOT))
    from clip_eval.pipeline.artifact_store import SHARED_CLIP_NAMES, SharedClipName

    assert set(get_args(SharedClipName)) == set(SHARED_CLIP_NAMES)


def test_local_pool_is_complete_if_present() -> None:
    """When a hydrated pool exists on disk, it must hold every name.

    Skipped rather than failed when absent: this repo is often checked out
    without artifacts, and hydration itself raises in that case.
    """
    artifacts = os.environ.get("ARTIFACTS_PATH")
    root = Path(artifacts) if artifacts else REPO_ROOT / "artifacts"
    pool = root / "clips" / "order_up_shared"
    if not pool.is_dir():
        print(f"  skip: no pool at {pool}")
        return
    have = {p.name for p in pool.glob("*.wav")}
    names = set(_shared_clip_names())
    assert not (names - have), f"pool {pool} missing {sorted(names - have)}"
    assert not (have - names), (
        f"pool {pool} holds unregistered {sorted(have - names)} — a clip was "
        f"added to the pool without being registered, so nothing downstream "
        f"knows to publish it"
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}\n      {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
