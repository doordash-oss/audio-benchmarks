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

"""Single source of truth for the eval's directory layout.

All runtime, scoring, and visualization code that needs to locate
artifact dirs should import these constants rather than hard-coding
`REPO_ROOT / "outputs"` or similar — any script that bypasses this
module will silently break next time the layout shifts.

Paths derive from this file's own location (clip_eval/pipeline/paths.py)
so the eval harness runs portable on any checkout: no per-machine setup,
no env-var dance.

`ARTIFACTS_PATH` is the one env-var override (defaults to
`<repo>/artifacts/`); everything else is derived from it.

Companion shell file: `paths.sh` next door in `pipeline/scripts/`; bash
scripts source it for the same constants.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIP_EVAL_ROOT = REPO_ROOT / "clip_eval"
# Artifact dir can be overridden via env var — useful when the artifacts
# ship separately from the code. Defaults to artifacts/ at the project
# root.
ARTIFACTS_PATH = Path(os.environ.get("ARTIFACTS_PATH") or (REPO_ROOT / "artifacts"))

# Build outputs
TASKS_DIR = ARTIFACTS_PATH / "order_up_tasks"
TREES_DIR = ARTIFACTS_PATH / "decision_trees"
CLIPS_DIR = ARTIFACTS_PATH / "clips"
CUSTOMER_CLIPS_DIR = ARTIFACTS_PATH / "customer_clips"
NOISE_DIR_ROOT = ARTIFACTS_PATH / "background_noise"

# Eval-run outputs
RUNS_ROOT = ARTIFACTS_PATH / "outputs"
ORDER_UP_RUNS = RUNS_ROOT / "OrderUp"

# Raw session data (source recordings + alignments)
SESSIONS_ROOT = ARTIFACTS_PATH / "sessions" / "order_up"

# Shared clip pool (yes.wav / no.wav fallbacks)
SHARED_CLIPS_DIR = CLIPS_DIR / "order_up_shared"

# Runtime split index (dataset enumerator the runner reads)
PROMPTS_INDEX = CLIP_EVAL_ROOT / "runtime" / "prompts" / "order_up.json"

# .env file at the project root — eval-runtime modules call load_env() at
# import to pick up OPENAI_API_KEY and friends.
ENV_FILE = REPO_ROOT / ".env"


def load_env() -> None:
    """Load .env from REPO_ROOT.

    Replaces the historical pattern of `load_dotenv("/Users/.../v2/.env")`
    sprinkled across modules — that broke any checkout that wasn't on the
    original author's machine. Safe to call multiple times.
    """
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE)
