#!/bin/bash
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

# Run a single Order Up clip-player eval end-to-end.
#
# Usage: run_order_up_eval.sh <task_id> [duration_seconds] [--role-b openai|openai-native|openai-fdb|gemini|grok|cascaded] [--offline]
#
# Reads order_up_tasks/<task_id>.complex.task.json (the only supported
# variant). Starts a fresh tool server on a free port, launches
# orchestrator + token server + clipPlayer (Role A) + Role B adapter
# with tools wired to the tool server, waits for the clipPlayer to
# reach the terminal node (or duration timeout), then runs the
# cart-comparison scorer.
#
# Duration: if not passed, auto-computed from the source human playthrough's
# game window length × DURATION_MULT (default 4×, min 180s). Agent turns +
# routing latency typically add 3-5× the original playthrough length.

set -euo pipefail

TASK_ID="${1:?Usage: $0 <task_id> [duration_seconds] [--role-b openai|openai-native|openai-fdb|gemini|ultravox|grok|cascaded]}"
shift || true
# Optional positional duration. Anything starting with "--" is parsed below.
DURATION_ARG=""
ROLE_B_CHOICE="${ROLE_B:-openai}"
OFFLINE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --role-b) ROLE_B_CHOICE="$2"; shift 2 ;;
    --role-b=*) ROLE_B_CHOICE="${1#*=}"; shift ;;
    --offline) OFFLINE=1; shift ;;
    *) if [[ -z "$DURATION_ARG" ]]; then DURATION_ARG="$1"; shift; else echo "unexpected arg: $1" >&2; exit 2; fi ;;
  esac
done
if [[ "$OFFLINE" == "1" ]]; then
  export CLIP_EVAL_OFFLINE=1
fi
if [[ "$ROLE_B_CHOICE" != "openai" && "$ROLE_B_CHOICE" != "openai-native" && "$ROLE_B_CHOICE" != "openai-fdb" && "$ROLE_B_CHOICE" != "gemini" && "$ROLE_B_CHOICE" != "ultravox" && "$ROLE_B_CHOICE" != "grok" && "$ROLE_B_CHOICE" != "cascaded" ]]; then
  echo "--role-b must be 'openai', 'openai-native', 'openai-fdb', 'gemini', 'ultravox', 'grok', or 'cascaded' (got: $ROLE_B_CHOICE)" >&2; exit 2
fi
DURATION_MULT="${DURATION_MULT:-4}"
DURATION_MIN="${DURATION_MIN:-180}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../pipeline/scripts/paths.sh"

# Load eval/.env so provider credentials reach every child process, not just
# the Node ones. The adapters call dotenv themselves, but the Python scorer and
# post-run audits read the environment — so a user who followed the documented
# "put it in eval/.env" path got a `Missing credentials` traceback from the
# first audit.
#
# Values already exported take precedence, matching dotenv's own semantics.
# `set -a; source` would invert that, so a key exported for a single run would
# be silently overridden here while the Node side still honored it.
if [[ -f "$REPO_ROOT/.env" ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    _line="${_line#"${_line%%[![:space:]]*}"}"          # strip leading space
    [[ -z "$_line" || "$_line" == \#* || "$_line" != *=* ]] && continue
    _key="${_line%%=*}"                                  # split on FIRST '='
    _val="${_line#*=}"                                   # keys may contain '='
    _key="${_key//[[:space:]]/}"
    [[ -z "$_key" || -n "${!_key:-}" ]] && continue
    _val="${_val%\"}"; _val="${_val#\"}"
    _val="${_val%\'}"; _val="${_val#\'}"
    export "$_key=$_val"
  done < "$REPO_ROOT/.env"
  unset _line _key _val
fi

TASK_FILE="$TASKS_DIR/${TASK_ID}.complex.task.json"
TREE_FILE="$TREES_DIR/${TASK_ID}.json"
TOOL_SERVER="$SCRIPT_DIR/order_up_tool_server.js"
SCORER="$SCRIPT_DIR/../scoring/score_order_up_cart.py"

# ── Role B (examinee) adapter — selected by --role-b (or $ROLE_B env) ────────
# Supported values:
#   openai    → Pipecat OpenAI Realtime adapter (default)
#   openai-native → direct OpenAI Realtime WebSocket adapter (no Pipecat)
#   openai-fdb → retained Full-Duplex-Bench-derived WebRTC implementation
#   gemini    → pipecat adapter with --provider gemini_live (Gemini Live native S2S)
#   ultravox  → pipecat adapter with --provider ultravox (Ultravox Realtime S2S)
#   grok      → pipecat adapter with --provider grok_realtime (xAI Grok Voice Agent S2S)
#   cascaded  → pipecat adapter with --provider cascaded (Silero VAD + Whisper +
#               gpt-4o + TTS — non-realtime baseline)
# An explicit $ROLE_B_CMD env wins over the dispatch — escape hatch for any
# adapter that follows clip_eval/docs/contracts/30_role_b_protocol.md.
# NOTE: the OpenAI token server is ALWAYS started because Role A (clipPlayer)
# needs it; Role B adapters that don't use it can ignore the passed --tokenServer.
ORCHESTRATOR_SCRIPT="$AUDIO_BRIDGE"
ACTIVE_TOKEN_SERVER_SCRIPT="$TOKEN_SERVER_SCRIPT"
if [[ -z "${ROLE_B_CMD:-}" ]]; then
  case "$ROLE_B_CHOICE" in
    openai)
      ROLE_B_CMD="$REPO_ROOT/.venv/bin/python $CLIP_EVAL_ROOT/role_b/pipecat/pipecat_role_b_adapter.py --provider openai"
      ROLE_B_PROVIDER="openai"
      ROLE_B_MODEL="${ROLE_B_MODEL:-${OPENAI_REALTIME_MODEL:-gpt-realtime-2}}"
      ;;
    openai-native)
      ROLE_B_CMD="node $CLIP_EVAL_ROOT/role_b/openai_native/openai_realtime_adapter.js"
      ROLE_B_PROVIDER="openai_native"
      ROLE_B_MODEL="${ROLE_B_MODEL:-${OPENAI_REALTIME_MODEL:-gpt-realtime-2}}"
      ;;
    openai-fdb)
      ROLE_B_CMD="node $SHARED_ROOT/role_b/reference/gptRealtime_adapter.js"
      ROLE_B_PROVIDER="openai_fdb"
      ROLE_B_MODEL="${ROLE_B_MODEL:-${OPENAI_REALTIME_MODEL:-gpt-realtime-2}}"
      ORCHESTRATOR_SCRIPT="$SHARED_ROOT/orchestrator.js"
      ACTIVE_TOKEN_SERVER_SCRIPT="$SHARED_ROOT/role_b/openai_token_server.js"
      ;;
    gemini)
      # --provider gemini_live tells the pipecat adapter to use Gemini Live's
      # API, not OpenAI Realtime (its default). Without it, --model
      # gemini-3.1-flash-live-preview is sent to OpenAI and rejected as
      # invalid_model.
      ROLE_B_CMD="$REPO_ROOT/.venv/bin/python $CLIP_EVAL_ROOT/role_b/pipecat/pipecat_role_b_adapter.py --provider gemini_live"
      ROLE_B_PROVIDER="gemini_live"
      ROLE_B_MODEL="${ROLE_B_MODEL:-${GEMINI_LIVE_MODEL:-gemini-3.1-flash-live-preview}}"
      ;;
    ultravox)
      # --provider ultravox tells the pipecat adapter to use Ultravox Realtime.
      # The API key is read from ULTRAVOX_API_KEY.
      ROLE_B_CMD="$REPO_ROOT/.venv/bin/python $CLIP_EVAL_ROOT/role_b/pipecat/pipecat_role_b_adapter.py --provider ultravox"
      ROLE_B_PROVIDER="ultravox"
      ROLE_B_MODEL="${ROLE_B_MODEL:-${ULTRAVOX_MODEL:-ultravox-v0.7}}"
      ;;
    grok)
      # --provider grok_realtime tells the pipecat adapter to use xAI's Grok
      # Voice Agent API. The API key is read from XAI_API_KEY (or GROK_API_KEY).
      ROLE_B_CMD="$REPO_ROOT/.venv/bin/python $CLIP_EVAL_ROOT/role_b/pipecat/pipecat_role_b_adapter.py --provider grok_realtime"
      ROLE_B_PROVIDER="grok_realtime"
      ROLE_B_MODEL="${ROLE_B_MODEL:-${GROK_REALTIME_MODEL:-${XAI_REALTIME_MODEL:-grok-voice-think-fast-1.0}}}"
      ;;
    cascaded)
      # Non-realtime baseline: Silero VAD + OpenAI Whisper STT + gpt-4o + OpenAI
      # TTS via the pipecat adapter. Model id is CASCADED_LLM_MODEL (not --model,
      # which the adapter scopes to realtime providers).
      ROLE_B_CMD="$REPO_ROOT/.venv/bin/python $CLIP_EVAL_ROOT/role_b/pipecat/pipecat_role_b_adapter.py --provider cascaded"
      ROLE_B_PROVIDER="cascaded"
      ROLE_B_MODEL="${ROLE_B_MODEL:-${CASCADED_LLM_MODEL:-gpt-4o}}"
      ;;
  esac
else
  ROLE_B_PROVIDER="${ROLE_B_PROVIDER:-custom}"
  ROLE_B_MODEL="${ROLE_B_MODEL:-${OPENAI_REALTIME_MODEL:-gpt-realtime-2}}"
fi
# Exported so the pipecat adapter can read $ROLE_B_PROVIDER as a default when
# --provider isn't explicitly on the command (e.g. via $ROLE_B_CMD overrides).
export ROLE_B_PROVIDER

# ── Setup ─────────────────────────────────────────────────────────────────
# Hydrates the artifact cache (task config, decision tree, customer-clip
# manifest, WAV clips — skipped in --offline mode), validates the task/tree
# files, then computes everything needed before any process starts: the
# skip-on-empty-cart signal, session id, duration, three free ports, the
# examinee prompt and the configure_tools payload. Emits `export K=V` lines.
# See order_up_eval_setup.py — kept out of this script so the logic is
# testable and this file stays a process launcher.
SETUP_FLAGS=()
if [[ "${OFFLINE:-0}" == "1" || "${CLIP_EVAL_OFFLINE:-0}" == "1" ]]; then
  SETUP_FLAGS=(--offline)
fi
SETUP_ENV="$(PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$REPO_ROOT/.venv/bin/python" "$SCRIPT_DIR/order_up_eval_setup.py" \
    "$TASK_ID" "$TASK_FILE" "$TREE_FILE" "$SESSIONS_ROOT" \
    "$DURATION_MULT" "$DURATION_MIN" "$DURATION_ARG" \
    "${SETUP_FLAGS[@]+"${SETUP_FLAGS[@]}"}")" || exit 1
eval "$SETUP_ENV"

# Skip if every expected item is disabled — repair_state.json drained the
# cart to zero items, so there is nothing to order and no meaningful score.
if [[ "${SKIP_EMPTY_CART:-0}" == "1" ]]; then
  echo "[run_order_up] SKIP $TASK_ID: expected_cart is empty (all items disabled by repair pipeline)"
  exit 0
fi

cd "$REPO_ROOT"

if [[ -z "$DURATION_ARG" ]]; then
  echo "[run_order_up] auto-duration: ${DURATION}s (source × ${DURATION_MULT}, min ${DURATION_MIN}s)"
fi
echo "[run_order_up] tool_server port=$TOOL_PORT"

# Locate the pre-extracted background-noise clips for the source session.
NOISE_DIR=""
if [[ -n "$SESSION_ID" && -d "$NOISE_DIR_ROOT/$SESSION_ID" ]]; then
  NOISE_DIR="$NOISE_DIR_ROOT/$SESSION_ID"
  echo "[run_order_up] noise_dir=$NOISE_DIR"
else
  echo "[run_order_up] no noise dir for session=$SESSION_ID; clipPlayer will emit digital silence between clips"
fi

# ── Output dir for this run ───────────────────────────────────────────────
# Timestamp is second-granular. Two parallel evals on the same task can
# still land on the same second, so we need an atomic test-and-create to
# claim a unique record_dir.
#
# `mkdir <dir>` (WITHOUT -p) is atomic — it returns success iff the
# kernel was the one that created the inode, and EEXIST otherwise.
# That's the lock semantic we need: race-loser processes silently retry
# the next suffix. The previous `if [[ -e ]] / mkdir -p` pattern was
# TOCTOU race-prone — two concurrent processes could both see "doesn't
# exist" between the test and the mkdir, then both `mkdir -p`
# (which doesn't fail on existing dirs), end up sharing the same
# record_dir, clobber each other's prompt.json, and race on
# `mkfifo tools.fifo` (first wins, others crash with "File exists").
# That produced mixed provider attribution in the dashboard and lost
# data from the losing processes.
RUN_TS_BASE="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ORDER_UP_RUNS/${TASK_ID}"
RECORD_DIR=""
for SUFFIX in "" _a _b _c _d _e _f _g _h _i _j _k _l _m _n _o _p _q _r _s _t _u _v _w _x _y _z; do
  CANDIDATE_TS="${RUN_TS_BASE}${SUFFIX}"
  CANDIDATE="$ORDER_UP_RUNS/${TASK_ID}/${CANDIDATE_TS}"
  if mkdir "$CANDIDATE" 2>/dev/null; then
    RECORD_DIR="$CANDIDATE"
    RUN_TS="$CANDIDATE_TS"
    break
  fi
done
if [[ -z "$RECORD_DIR" ]]; then
  echo "[run_order_up] ERROR: 27 sibling record_dirs all exist for ${RUN_TS_BASE}" >&2
  exit 1
fi
echo "[run_order_up] record_dir=$RECORD_DIR"

# Snapshot the EXACT prompt + tools the examinee was given on this run,
# next to the recording, so historical runs stay reproducible even if the
# task config changes later. Values go through the environment (not
# shell-interpolated) so a ROLE_B_CMD containing quotes, backslashes or
# newlines cannot break the snapshot. See order_up_eval_snapshot.py.
ROLE_B_CMD="$ROLE_B_CMD" ROLE_B_PROVIDER="$ROLE_B_PROVIDER" ROLE_B_MODEL="$ROLE_B_MODEL" \
TASK_FILE="$TASK_FILE" RECORD_DIR="$RECORD_DIR" TOOL_URL="$TOOL_URL" DURATION="$DURATION" \
PROMPT_B="$PROMPT_B" \
  "$REPO_ROOT/.venv/bin/python" "$SCRIPT_DIR/order_up_eval_snapshot.py"

# ── Start tool server ─────────────────────────────────────────────────────
node "$TOOL_SERVER" \
  --task-config "$TASK_FILE" \
  --record-dir  "$RECORD_DIR" \
  --port        "$TOOL_PORT" \
  > "$RECORD_DIR/tool_server.log" 2>&1 &
TOOL_PID=$!
echo "[run_order_up] tool_server PID=$TOOL_PID"

# Wait for "READY" line (max 5s)
for i in {1..50}; do
  if grep -q "^READY " "$RECORD_DIR/tool_server.log" 2>/dev/null; then break; fi
  sleep 0.1
done
if ! grep -q "^READY " "$RECORD_DIR/tool_server.log"; then
  echo "[run_order_up] tool server failed to start" >&2
  kill "$TOOL_PID" 2>/dev/null || true
  exit 1
fi

# ── Cleanup handler ───────────────────────────────────────────────────────
cleanup() {
  echo "[run_order_up] cleanup"
  pkill -P $$ 2>/dev/null || true
  kill "$TOOL_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── Run single_conversation.sh in the foreground (with our own adapter B
#    launched separately so we can pipe configure_tools to its stdin).
#
# Easiest implementation: build the command line ourselves rather than
# reuse single_conversation.sh, since we need stdin control on adapter B.
# ─────────────────────────────────────────────────────────────────────────
# SIGNAL_PORT / TOKEN_PORT were allocated by order_up_eval_setup.py alongside
# TOOL_PORT, so all three come from one place.

export RECORD_DIR
export SIGNAL_PORT
export OPENAI_TOKEN_SERVER="http://localhost:${TOKEN_PORT}"
export TOKEN_SERVER_PORT="$TOKEN_PORT"

echo "[run_order_up] orchestrator port=$SIGNAL_PORT  token_server port=$TOKEN_PORT"

# Orchestrator
SIGNAL_PORT="$SIGNAL_PORT" node "$ORCHESTRATOR_SCRIPT" \
  > "$RECORD_DIR/orchestrator.log" 2>&1 &
ORCH_PID=$!
sleep 0.8

# Token server (uses TOKEN_SERVER_PORT env)
TOKEN_SERVER_PORT="$TOKEN_PORT" node "$ACTIVE_TOKEN_SERVER_SCRIPT" \
  > "$RECORD_DIR/token_server.log" 2>&1 &
TOK_PID=$!
sleep 0.5

SIGNAL_URL="ws://localhost:${SIGNAL_PORT}/signal"

# Per-task override sidecar (same path the complex builder consumes). When
# present, clipPlayer reads it for `clip_router_hints` — stage-specific
# routing cues that get appended to per-clip descriptions in valid_choices.
# Optional; missing file is a no-op.
TASK_OVERRIDE_PATH="$REPO_ROOT/artifacts/order_up_task_overrides/${TASK_ID}.complex.override.json"
TASK_OVERRIDE_ARGS=()
if [[ -f "$TASK_OVERRIDE_PATH" ]]; then
  TASK_OVERRIDE_ARGS=(--taskOverrides "$TASK_OVERRIDE_PATH")
fi

# Role A: clipPlayer (the customer, plays our extracted clips)
node "$CLIP_EVAL_ROOT/role_a/clipPlayer_adapter.js" \
  --role A --signalUrl "$SIGNAL_URL" \
  --tokenServer "$OPENAI_TOKEN_SERVER" \
  --toolServerUrl "$TOOL_URL" \
  --voice alloy --vadMode slow \
  --decisionTree "$TREE_FILE" \
  --taskConfig "$TASK_FILE" \
  ${NOISE_DIR:+--noiseDir "$NOISE_DIR"} \
  ${TASK_OVERRIDE_ARGS[@]+"${TASK_OVERRIDE_ARGS[@]}"} \
  > "$RECORD_DIR/adapter_A.log" 2>&1 &
A_PID=$!

# Role B: gptRealtime (the agent under test). Pipe configure_tools to its
# stdin via a FIFO. The writer is a separately tracked PID we kill in
# cleanup, so the script doesn't get stuck waiting for a sleep in a piped
# subshell when it exits.
TOOLS_FIFO="$RECORD_DIR/tools.fifo"
mkfifo "$TOOLS_FIFO"
# Writer process: print the configure_tools message, then sleep on the FIFO
# to keep stdin open for the adapter. Tracked as WRITER_PID so cleanup can
# kill it explicitly.
( printf '%s\n' "$TOOLS_JSON"; exec sleep "$((DURATION + 30))" ) > "$TOOLS_FIFO" &
WRITER_PID=$!
# shellcheck disable=SC2086  # $ROLE_B_CMD is an intentional multi-word command
$ROLE_B_CMD \
      --role B --signalUrl "$SIGNAL_URL" \
      --tokenServer "$OPENAI_TOKEN_SERVER" \
      --voice shimmer --vadMode fast \
      --speakFirst \
      --enableTranscription \
      --model "$ROLE_B_MODEL" \
      --systemPrompt "$PROMPT_B" \
      < "$TOOLS_FIFO" \
      > "$RECORD_DIR/adapter_B.log" 2>&1 &
B_PID=$!

echo "[run_order_up] orchestrator=$ORCH_PID token=$TOK_PID A=$A_PID B=$B_PID"

# ── Wait for the conversation to end (clipPlayer terminal) or timeout ─────
# clipPlayer exits with 0 when terminal node hit. We poll for either.
END_AT=$((SECONDS + DURATION))
while (( SECONDS < END_AT )); do
  if ! kill -0 "$A_PID" 2>/dev/null; then
    echo "[run_order_up] clipPlayer ended naturally"
    break
  fi
  sleep 1
done
if (( SECONDS >= END_AT )); then
  echo "[run_order_up] duration timeout (${DURATION}s) — stopping"
fi

# Trigger orchestrator mixdown then kill the rest
kill -SIGINT "$ORCH_PID" 2>/dev/null || true
# WAV headers and combined.wav are finalized asynchronously. Wait for the
# bridge instead of racing scoring/metrics against a fixed sleep.
for _ in {1..80}; do
  if ! kill -0 "$ORCH_PID" 2>/dev/null; then break; fi
  sleep 0.1
done
if kill -0 "$ORCH_PID" 2>/dev/null; then
  echo "[run_order_up] audio bridge did not exit within 8s; forcing shutdown" >&2
  kill -KILL "$ORCH_PID" 2>/dev/null || true
fi
wait "$ORCH_PID" 2>/dev/null || true
kill "$A_PID" "$B_PID" "$TOK_PID" "${WRITER_PID:-}" 2>/dev/null || true
# Also kill any descendants of the writer (the inner sleep).
[ -n "${WRITER_PID:-}" ] && pkill -KILL -P "$WRITER_PID" 2>/dev/null || true
rm -f "${TOOLS_FIFO:-}" 2>/dev/null || true
sleep 0.5

# Tool server: send SIGTERM so it writes final_cart.json on exit
kill "$TOOL_PID" 2>/dev/null || true
wait "$TOOL_PID" 2>/dev/null || true

# ── Score ────────────────────────────────────────────────────────────────
python3 "$SCORER" "$TASK_FILE" "$RECORD_DIR"

# Interrupt detection runs in-process via clipPlayer_adapter.js's RMS
# VAD on the B-audio it receives — see [ClipPlayer] interrupt_flag log
# lines.

# Post-eval: behavioural metrics, every cart-error audit, and turn-taking,
# then one consolidated report of the cart score and each audit's outcome.
# Run in-process by order_up_eval_report.py rather than as a bash loop over
# separate `-m` subprocesses — audit ORDER is load-bearing (preemptive_tool
# promotes extra-item -> wrong-item, and missing_item_root_cause must add
# `missing-item-not-detected` before missing_item_intent classifies), so it
# belongs in one place that can be read and tested.
# Set SKIP_POST_EVAL=1 to skip on batched re-runs that only need the score.
if [[ "${SKIP_POST_EVAL:-0}" == "1" ]]; then
  echo "[run_order_up] SKIP_POST_EVAL=1 → skipping eval_metrics + audits"
else
  (cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$REPO_ROOT/.venv/bin/python" "$SCRIPT_DIR/../scoring/order_up_eval_report.py" \
      "$RECORD_DIR") \
    || echo "[run_order_up] post-eval report failed (non-fatal)"
fi

echo "[run_order_up] done. record_dir=$RECORD_DIR"
