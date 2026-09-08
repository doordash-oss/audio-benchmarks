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

"""Guard: code and docs must agree on which dataset revision is pinned.

Commit 31caf66 fixed exactly this drift — "the doc pointed HF_REVISION at the
snapshot taken before the background noise and task id map were published", so
following the documented setup verbatim produced a run missing inputs the code
expected. The pin lives in one constant but is quoted in prose and in a
copy-pasteable `load_dataset` snippet, and a republish has to move all of them
together.

Run: python -m clip_eval.tests.test_hf_revision_pin
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = REPO_ROOT.parent

SHA40 = re.compile(r"\b[0-9a-f]{40}\b")

# Files that legitimately quote the pin. Anything else mentioning a 40-hex sha
# is checked too, but these are the ones a republish must update.
SEARCH = (
    REPO_ROOT / "clip_eval" / "pipeline" / "artifact_store.py",
    PROJECT_ROOT / "RUNNING_BENCHMARKS.md",
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "index.md",
)


def _pinned() -> str:
    sys.path.insert(0, str(REPO_ROOT))
    from clip_eval.pipeline.artifact_store import DEFAULT_HF_REVISION

    return DEFAULT_HF_REVISION


def test_every_sha_matches_the_pin() -> None:
    pinned = _pinned()
    bad: list[str] = []
    for path in SEARCH:
        if not path.is_file():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            for sha in SHA40.findall(line):
                if sha != pinned:
                    rel = path.relative_to(PROJECT_ROOT)
                    bad.append(f"{rel}:{lineno} pins {sha[:12]}…")
    assert not bad, (
        f"these references disagree with DEFAULT_HF_REVISION "
        f"({pinned[:12]}…):\n  " + "\n  ".join(bad) +
        "\nA republish must move the constant and every doc reference in the "
        "same commit, or a documented setup hydrates a different snapshot "
        "than the code expects."
    )


def test_pin_is_a_full_sha() -> None:
    """A branch name or short sha would let the pinned data move underfoot."""
    pinned = _pinned()
    assert SHA40.fullmatch(pinned), (
        f"DEFAULT_HF_REVISION is {pinned!r}; it must be a full 40-char commit "
        f"sha so benchmark inputs cannot drift"
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
