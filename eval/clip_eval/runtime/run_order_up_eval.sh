#!/bin/bash
# Run a single Order Up clip-player eval end-to-end.
#
# Usage: run_order_up_eval.sh <task_id> [duration_seconds] [--role-b openai|gemini|grok|cascaded] [--clip-player-profile <name>] [--offline]
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

TASK_ID="${1:?Usage: $0 <task_id> [duration_seconds] [--role-b openai|gemini|ultravox|grok|cascaded] [--clip-player-profile <name>]}"
shift || true
# Optional positional duration. Anything starting with "--" is parsed below.
DURATION_ARG=""
ROLE_B_CHOICE="${ROLE_B:-openai}"
CLIP_PLAYER_PROFILE="${CLIP_PLAYER_PROFILE:-}"
OFFLINE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --role-b) ROLE_B_CHOICE="$2"; shift 2 ;;
    --role-b=*) ROLE_B_CHOICE="${1#*=}"; shift ;;
    --clip-player-profile) CLIP_PLAYER_PROFILE="$2"; shift 2 ;;
    --clip-player-profile=*) CLIP_PLAYER_PROFILE="${1#*=}"; shift ;;
    --offline) OFFLINE=1; shift ;;
    *) if [[ -z "$DURATION_ARG" ]]; then DURATION_ARG="$1"; shift; else echo "unexpected arg: $1" >&2; exit 2; fi ;;
  esac
done
if [[ "$OFFLINE" == "1" ]]; then
  export CLIP_EVAL_OFFLINE=1
fi
if [[ "$ROLE_B_CHOICE" != "openai" && "$ROLE_B_CHOICE" != "gemini" && "$ROLE_B_CHOICE" != "ultravox" && "$ROLE_B_CHOICE" != "grok" && "$ROLE_B_CHOICE" != "cascaded" ]]; then
  echo "--role-b must be 'openai', 'gemini', 'ultravox', 'grok', or 'cascaded' (got: $ROLE_B_CHOICE)" >&2; exit 2
fi
DURATION_MULT="${DURATION_MULT:-4}"
DURATION_MIN="${DURATION_MIN:-180}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../pipeline/scripts/paths.sh"
TASK_FILE="$TASKS_DIR/${TASK_ID}.complex.task.json"
TREE_FILE="$TREES_DIR/${TASK_ID}.json"
TOOL_SERVER="$SCRIPT_DIR/order_up_tool_server.js"
SCORER="$SCRIPT_DIR/../scoring/score_order_up_cart.py"

# ── Role B (examinee) adapter — selected by --role-b (or $ROLE_B env) ────────
# Supported values:
#   openai    → node $SHARED_ROOT/role_b/reference/gptRealtime_adapter.js  (default)
#   gemini    → pipecat adapter with --provider gemini_live (Gemini Live native S2S)
#   ultravox  → pipecat adapter with --provider ultravox (Ultravox Realtime S2S)
#   grok      → pipecat adapter with --provider grok_realtime (xAI Grok Voice Agent S2S)
#   cascaded  → pipecat adapter with --provider cascaded (Silero VAD + Whisper +
#               gpt-4o + TTS — non-realtime baseline)
# An explicit $ROLE_B_CMD env wins over the dispatch — escape hatch for any
# adapter that follows clip_eval/docs/contracts/30_role_b_protocol.md.
# NOTE: the OpenAI token server is ALWAYS started because Role A (clipPlayer)
# needs it; Role B adapters that don't use it can ignore the passed --tokenServer.
if [[ -z "${ROLE_B_CMD:-}" ]]; then
  case "$ROLE_B_CHOICE" in
    openai)
      ROLE_B_CMD="node $SHARED_ROOT/role_b/reference/gptRealtime_adapter.js"
      ROLE_B_PROVIDER="openai"
      ROLE_B_MODEL="${ROLE_B_MODEL:-${OPENAI_REALTIME_MODEL:-gpt-realtime-2}}"
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

# Pre-pull from GCS: hydrate everything this task needs (manifest, tree,
# task config, per-stage clips, owning session's transcription + audit +
# classified_spans + background_noise, shared TTS) in one batched fetch.
# Skipped in --offline mode (caller's responsibility to ensure cache).
if [[ "${OFFLINE:-0}" != "1" && "${CLIP_EVAL_OFFLINE:-0}" != "1" ]]; then
  TASK_ID="$TASK_ID" PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$REPO_ROOT/.venv/bin/python" - <<'PY' || \
      echo "[run_order_up] warn: cache_pull_stage failed (will rely on local cache)" >&2
import os, sys, time
from clip_eval.pipeline.artifact_store import cache_pull_stage
t0 = time.time()
try:
    cache_pull_stage(os.environ["TASK_ID"])
    print(f"[run_order_up] cache_pull_stage OK in {time.time()-t0:.1f}s", file=sys.stderr)
except Exception as e:
    print(f"[run_order_up] cache_pull_stage error: {e}", file=sys.stderr)
    raise SystemExit(1)
PY
fi

if [[ ! -f "$TASK_FILE" ]]; then
  echo "Missing task file: $TASK_FILE" >&2; exit 1
fi
if [[ ! -f "$TREE_FILE" ]]; then
  echo "Missing decision tree: $TREE_FILE" >&2; exit 1
fi

# Auto-detect clipPlayer profile from the customer_clips manifest when
# the caller didn't pin one via --clip-player-profile. The helper picks
# `restate-on-partial-ack` for stages with any compound-main clip (a
# main clip whose customer_spans cover ≥2 items — the exact case rule 9
# of that profile targets), otherwise `default`. Explicit CLI wins.
if [[ -z "$CLIP_PLAYER_PROFILE" ]]; then
  # Compute the audio-benchmarks eval root from SCRIPT_DIR directly.
  # $REPO_ROOT can get clobbered by callers that pre-set the var (e.g.
  # the FDB wrapper leaks its own REPO_ROOT), so anchor off SCRIPT_DIR
  # which is set from BASH_SOURCE[0] of THIS script.
  _EVAL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
  CLIP_PLAYER_PROFILE="$(TASK_ID="$TASK_ID" \
      ARTIFACTS_PATH="$ARTIFACTS_PATH" \
      PYTHONPATH="$_EVAL_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      "$_EVAL_ROOT/.venv/bin/python" -m clip_eval.runtime.infer_clip_player_profile "$TASK_ID" || echo default)"
  echo "[run_order_up] auto-detected clipPlayer profile: $CLIP_PLAYER_PROFILE"
fi

cd "$REPO_ROOT"

# Skip if every expected item is disabled — repair_state.json drained
# the cart down to zero items, so there's nothing for the agent to
# order and no meaningful score to compute. Exits cleanly with a
# clear message rather than crashing the orchestrator on an empty
# task or running an eval that's guaranteed to score 0/0.
N_EXPECTED="$(python3 -c "import json,sys; print(len((json.load(open('$TASK_FILE')).get('expected_cart') or [])))")"
if [[ "$N_EXPECTED" == "0" ]]; then
  echo "[run_order_up] SKIP $TASK_ID: expected_cart is empty (all items disabled by repair pipeline)"
  exit 0
fi

# Pull the source session id from the task config so we can locate its
# pre-extracted background-noise clip directory.
SESSION_ID="$(python3 -c "import json,sys; print(json.load(open('$TASK_FILE')).get('session_id') or '')")"
NOISE_DIR=""
if [[ -n "$SESSION_ID" && -d "$NOISE_DIR_ROOT/$SESSION_ID" ]]; then
  NOISE_DIR="$NOISE_DIR_ROOT/$SESSION_ID"
  echo "[run_order_up] noise_dir=$NOISE_DIR"
else
  echo "[run_order_up] no noise dir for session=$SESSION_ID; clipPlayer will emit digital silence between clips"
fi

# Auto-compute the duration from the source game window if not provided.
if [[ -z "$DURATION_ARG" ]]; then
  DURATION="$(python3 - <<EOF
import json, math, sys
t = json.load(open("$TASK_FILE"))
sid = t.get("session_id"); gi = t.get("game_index")
src = f"{'$SESSIONS_ROOT'}/{sid}/stage_alignment.json"
try:
    a = json.load(open(src))
    g = a["games"][gi]
    src_dur = g["audio_end"] - g["audio_start"]
except Exception:
    src_dur = 0.0
mult = float("$DURATION_MULT")
mn = float("$DURATION_MIN")
print(int(max(mn, math.ceil(src_dur * mult))))
EOF
)"
  echo "[run_order_up] auto-duration: ${DURATION}s (source × ${DURATION_MULT}, min ${DURATION_MIN}s)"
else
  DURATION="$DURATION_ARG"
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

# ── Pick a free port for the tool server ──────────────────────────────────
TOOL_PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')"
TOOL_URL="http://localhost:${TOOL_PORT}"
echo "[run_order_up] tool_server port=$TOOL_PORT"

# ── Pull tools + examinee prompt from task file ───────────────────────────
PROMPT_B="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['examinee_system_prompt'])" "$TASK_FILE")"
# Optional eval-time examinee-prompt override. When EXAMINEE_PROMPT_OVERRIDE_FILE
# points at a readable file, its contents replace the baked
# examinee_system_prompt for this run — used by the --agent-lang experiment
# (e.g. "blind": strip the baked language directive so the agent is NOT told
# which language the customer speaks). The override feeds BOTH the live Role B
# prompt (below) and the prompt.json snapshot, so recorded runs stay faithful.
if [[ -n "${EXAMINEE_PROMPT_OVERRIDE_FILE:-}" ]]; then
  if [[ -f "$EXAMINEE_PROMPT_OVERRIDE_FILE" ]]; then
    PROMPT_B="$(cat "$EXAMINEE_PROMPT_OVERRIDE_FILE")"
    echo "[run_order_up] examinee prompt OVERRIDDEN from $EXAMINEE_PROMPT_OVERRIDE_FILE (len=${#PROMPT_B})" >&2
  else
    echo "[run_order_up] warn: EXAMINEE_PROMPT_OVERRIDE_FILE not found: $EXAMINEE_PROMPT_OVERRIDE_FILE (using baked prompt)" >&2
  fi
fi
TOOLS_JSON="$(python3 -c "
import json, sys
t=json.load(open(sys.argv[1]))
print(json.dumps({'type':'configure_tools','tools':t['tools'],'serverUrl':sys.argv[2]}))
" "$TASK_FILE" "$TOOL_URL")"

# Snapshot the EXACT prompt + tools the examinee was given on this run.
# Stored next to the recording so historical runs are reproducible even if
# the task config changes later.
# Pass values via the environment (not shell-interpolated into the Python
# source) so a ROLE_B_CMD containing quotes/backslashes/newlines can't break
# the generated snapshot. Quoted heredoc (<<'EOF') = no shell expansion inside.
ROLE_B_CMD="$ROLE_B_CMD" ROLE_B_PROVIDER="$ROLE_B_PROVIDER" ROLE_B_MODEL="$ROLE_B_MODEL" \
TASK_FILE="$TASK_FILE" RECORD_DIR="$RECORD_DIR" TOOL_URL="$TOOL_URL" DURATION="$DURATION" \
PROMPT_B="$PROMPT_B" \
python3 - <<'EOF'
import json, os
t = json.load(open(os.environ["TASK_FILE"]))
snapshot = {
    "task_id": t["task_id"],
    "model": os.environ["ROLE_B_MODEL"],
    "role_b_provider": os.environ["ROLE_B_PROVIDER"],
    "role_b_cmd": os.environ["ROLE_B_CMD"],
    "examinee_system_prompt": os.environ["PROMPT_B"],
    "tools": t["tools"],
    "menu_sample": t.get("menu_sample"),
    "tool_server_url": os.environ["TOOL_URL"],
    "duration_seconds": int(os.environ["DURATION"]),
}
open(os.path.join(os.environ["RECORD_DIR"], "prompt.json"), "w").write(json.dumps(snapshot, indent=2))
EOF

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
SIGNAL_PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')"
TOKEN_PORT="$(python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')"

export RECORD_DIR
export SIGNAL_PORT
export OPENAI_TOKEN_SERVER="http://localhost:${TOKEN_PORT}"
export TOKEN_SERVER_PORT="$TOKEN_PORT"

echo "[run_order_up] orchestrator port=$SIGNAL_PORT  token_server port=$TOKEN_PORT"

# Orchestrator
SIGNAL_PORT="$SIGNAL_PORT" node "$SHARED_ROOT/orchestrator.js" \
  > "$RECORD_DIR/orchestrator.log" 2>&1 &
ORCH_PID=$!
sleep 0.8

# Token server (uses TOKEN_SERVER_PORT env)
TOKEN_SERVER_PORT="$TOKEN_PORT" node "$SHARED_ROOT/role_b/openai_token_server.js" \
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
  --profile "$CLIP_PLAYER_PROFILE" \
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
sleep 1
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

# Post-eval behavioral metrics: per-turn response latency + duration,
# response rate, interrupt-vs-clip-properties analysis. Writes
# `eval_metrics.json` next to score.json. Non-fatal on failure.
# Set SKIP_POST_EVAL=1 to skip on batched re-runs that only need the
# cart score.
if [[ "${SKIP_POST_EVAL:-0}" == "1" ]]; then
  echo "[run_order_up] SKIP_POST_EVAL=1 → skipping eval_metrics + audits"
else
  PYRUN=(uv run python)
  (cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "${PYRUN[@]}" -m clip_eval.scoring.eval_metrics "$RECORD_DIR") \
    || echo "[run_order_up] eval_metrics failed (non-fatal)"

  # Tag-driven audits — each script self-guards on its trigger_tag and
  # no-ops when the relevant cart-diff tag isn't on score.json. Order
  # matters because:
  #   1. preemptive_tool_audit promotes extra-item → wrong-item via the
  #      silence check; later audits read the updated tag set.
  #   2. wrong_option_audit reads the wrong-option-or-variant pairs the
  #      scorer surfaced (no dependency on #1 but cheap to keep close).
  #   3. missing_item_root_cause runs clipplayer_timeout_audit via
  #      --auto and adds the `missing-item-not-detected` sub-tag; it
  #      MUST land before #4 so the intent audit can read that tag
  #      when classifying.
  #   4. missing_item_intent_audit classifies the remaining missing
  #      items (premature_submit / compound_line / clarify_abandoned /
  #      other) for dashboard breakdown rollup.
  for audit in \
    preemptive_tool_audit \
    wrong_option_audit \
    missing_item_root_cause \
    missing_item_intent_audit \
    off_policy_audit; do
    extra_args=()
    [[ "$audit" == "missing_item_root_cause" ]] && extra_args=(--auto)
    (cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      "${PYRUN[@]}" -m "clip_eval.scoring.cart_error_audit.$audit" "$RECORD_DIR" "${extra_args[@]+"${extra_args[@]}"}") \
      || echo "[run_order_up] $audit failed (non-fatal)"
  done

  # Deterministic turn-taking metrics (interruption rate + no-response
  # by clip length). Writes `turn_taking.json` next to score.json. No
  # LLM, no tag-gating — always runs.
  (cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "${PYRUN[@]}" -m clip_eval.scoring.turn_taking_analysis "$RECORD_DIR") \
    || echo "[run_order_up] turn_taking_analysis failed (non-fatal)"
fi

echo "[run_order_up] done. record_dir=$RECORD_DIR"
