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

# Single source of truth for the eval's directory layout.
#
# `source` this file from any shell script that needs to locate
# artifact dirs. Mirrors clip_eval/pipeline/paths.py — keep them in sync.
#
# Paths derive from this file's location
# (clip_eval/pipeline/scripts/paths.sh) so the harness runs portable on
# any checkout.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CLIP_EVAL_ROOT="$REPO_ROOT/clip_eval"
RUNTIME_ROOT="$CLIP_EVAL_ROOT/runtime"
AUDIO_BRIDGE="$RUNTIME_ROOT/audio_bridge.js"
TOKEN_SERVER_SCRIPT="$RUNTIME_ROOT/openai_token_server.js"
# Artifact dir can be overridden via env var — useful when the artifacts
# ship separately from the code. Defaults to artifacts/ next to the
# project root.
ARTIFACTS_PATH="${ARTIFACTS_PATH:-$REPO_ROOT/artifacts}"

# Make npm-installed modules resolvable from any cwd via NODE_PATH so
# orchestrator/adapters/tool-server can require('ws'), require('yargs'),
# etc. regardless of where they're launched from.
export NODE_PATH="${NODE_PATH:+$NODE_PATH:}$REPO_ROOT/node_modules"

# Build outputs
TASKS_DIR="$ARTIFACTS_PATH/order_up_tasks"
TREES_DIR="$ARTIFACTS_PATH/decision_trees"
CLIPS_DIR="$ARTIFACTS_PATH/clips"
CUSTOMER_CLIPS_DIR="$ARTIFACTS_PATH/customer_clips"
NOISE_DIR_ROOT="$ARTIFACTS_PATH/background_noise"

# Eval-run outputs
RUNS_ROOT="$ARTIFACTS_PATH/outputs"
ORDER_UP_RUNS="$RUNS_ROOT/OrderUp"

# Raw session data (source recordings + alignments)
SESSIONS_ROOT="$ARTIFACTS_PATH/sessions/order_up"

# Shared clip pool (yes.wav / no.wav fallbacks)
SHARED_CLIPS_DIR="$CLIPS_DIR/order_up_shared"

