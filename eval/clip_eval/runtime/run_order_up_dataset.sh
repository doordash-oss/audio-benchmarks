#!/bin/bash
# Run a set of Order Up tasks against a Role B model, with optional
# concurrency.
#
# Usage:
#   run_order_up_dataset.sh [jobs_file] \
#     [--role-b openai|gemini|ultravox|grok|cascaded] \
#     [--max-concurrency N] \
#     [--clip-player-profile <name>] \
#     [--log-dir <path>] \
#     [--duration <seconds>]
#
# Task list source:
#   - If a positional <jobs_file> is provided, read task_ids from it
#     (one per line; blank lines and `#` comments ignored).
#   - Otherwise default to the V1 benchmark working set, read from
#     $BENCH_MD (defaults to `<repo>/eval/V1_BENCHMARK_SET.md`, adjacent
#     to this eval tree). Every row in that file is already the working
#     set — no status filtering, no profile column. Filters to per-stage
#     tids that are (a) evaluable (have a usable decision tree) and (b)
#     have a complex task config built.
#
# Concurrency:
#   --max-concurrency N (default 1 = sequential). Each eval self-allocates
#   its tool-server / orchestrator / token-server / WebRTC ports, so
#   multiple evals can share a host.
#
# Role B model:
#   --role-b X forwards to run_order_up_eval.sh. If unset, the wrapper
#   falls back to $ROLE_B from env (default openai).
#
# clipPlayer profile:
#   Auto-detected per-tid by run_order_up_eval.sh from the customer_clips
#   manifest (compound-main → restate-on-partial-ack, else default).
#   Override globally with --clip-player-profile <name>. No per-row
#   manifest tag is honored anymore; rebuild artifacts if you need
#   different behavior.
#
# Per-task logs at $LOG_DIR/<tid>.log (default /tmp/order_up_dataset/).
# Status aggregate at $LOG_DIR/STATUS.txt with one OK/FAIL line per task.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../pipeline/scripts/paths.sh"
RUNNER="$SCRIPT_DIR/run_order_up_eval.sh"

JOBS_FILE=""
ROLE_B_ARG=""
MAX_CONCURRENCY=1
CLIP_PLAYER_PROFILE_OVERRIDE=""
LOG_DIR="/tmp/order_up_dataset"
DURATION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role-b) ROLE_B_ARG="$2"; shift 2 ;;
    --role-b=*) ROLE_B_ARG="${1#*=}"; shift ;;
    --max-concurrency|-j) MAX_CONCURRENCY="$2"; shift 2 ;;
    --max-concurrency=*) MAX_CONCURRENCY="${1#*=}"; shift ;;
    --clip-player-profile) CLIP_PLAYER_PROFILE_OVERRIDE="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    -h|--help)
      grep -E '^# ' "$0" | sed 's/^# //'; exit 0 ;;
    --*) echo "unexpected flag: $1" >&2; exit 2 ;;
    *)
      if [[ -z "$JOBS_FILE" ]]; then
        JOBS_FILE="$1"; shift
      else
        echo "unexpected positional arg: $1" >&2; exit 2
      fi
      ;;
  esac
done

# Resolve the task list.
TASK_IDS_FILE="$(mktemp)"
trap 'rm -f "$TASK_IDS_FILE"' EXIT
if [[ -n "$JOBS_FILE" ]]; then
  if [[ ! -f "$JOBS_FILE" ]]; then
    echo "jobs_file not found: $JOBS_FILE" >&2; exit 1
  fi
  # Strip blank lines and `#` comments.
  grep -vE '^\s*(#|$)' "$JOBS_FILE" > "$TASK_IDS_FILE"
  TASK_SOURCE="$JOBS_FILE"
else
  # Default: V1 benchmark working set, filtered by:
  #   per-stage  ∩  evaluable (usable tree)  ∩  has complex task config
  # The manifest lives adjacent to this eval tree (eval/V1_BENCHMARK_SET.md);
  # override with $BENCH_MD.
  BENCH_MD="${BENCH_MD:-$SCRIPT_DIR/../../V1_BENCHMARK_SET.md}"
  if [[ ! -f "$BENCH_MD" ]]; then
    echo "Missing $BENCH_MD — needed for the default benchmark task list" >&2
    exit 1
  fi
  python3 - "$BENCH_MD" "$ARTIFACTS_PATH" > "$TASK_IDS_FILE" <<'PY'
import json, re, sys
from pathlib import Path
bench_md = Path(sys.argv[1])
artifacts = Path(sys.argv[2])

# Every row in the clean manifest is a working-set tid — no status
# flags to strip, no profile column to parse. Ordering follows file
# order.
tids: list[str] = []
seen: set[str] = set()
for line in bench_md.read_text().splitlines():
    m = re.match(r'^\| `(order_up\.[^`]+)` \|', line)
    if not m: continue
    tid = m.group(1)
    if tid in seen: continue
    seen.add(tid)
    tids.append(tid)

def unevaluable(tid: str) -> bool:
    """Structural unevaluable check — broken decision tree. The clean
    manifest should never contain unevaluable tids in the first place;
    this is a defense-in-depth guard for edit accidents."""
    tree_p = artifacts / 'decision_trees' / f'{tid}.json'
    if not tree_p.exists(): return True
    try: t = json.loads(tree_p.read_text())
    except Exception: return True
    init = t.get('initial_node')
    if not init: return True
    if init not in (t.get('nodes') or {}): return True
    if not init.endswith('_main') and '_main' not in init: return True
    return False

for tid in tids:
    if not re.search(r'\.s\d+$', tid): continue                                      # per-stage only
    if unevaluable(tid): continue                                                    # broken tree
    if not (artifacts / 'order_up_tasks' / f'{tid}.complex.task.json').exists():     # no complex config
        continue
    print(tid)
PY
  TASK_SOURCE="$BENCH_MD (V1 working set)"
fi

N_TASKS=$(wc -l < "$TASK_IDS_FILE" | tr -d ' ')
if (( N_TASKS == 0 )); then
  echo "No tasks to run (source: $TASK_SOURCE)" >&2; exit 1
fi

mkdir -p "$LOG_DIR"
STATUS="$LOG_DIR/STATUS.txt"
: > "$STATUS"

echo "[run_order_up_dataset] $N_TASKS tasks (source: $TASK_SOURCE)"
echo "[run_order_up_dataset]   role_b=${ROLE_B_ARG:-${ROLE_B:-openai}} max_concurrency=$MAX_CONCURRENCY"
if [[ -n "$CLIP_PLAYER_PROFILE_OVERRIDE" ]]; then
  echo "[run_order_up_dataset]   profile=${CLIP_PLAYER_PROFILE_OVERRIDE} (global override)"
else
  echo "[run_order_up_dataset]   profile auto-detected per tid from the customer_clips manifest"
fi
if [[ -n "$DURATION" ]]; then
  echo "[run_order_up_dataset]   duration=${DURATION}s (override)"
else
  echo "[run_order_up_dataset]   duration auto (source × 4)"
fi
echo "[run_order_up_dataset]   logs → $LOG_DIR/<tid>.log"
echo "[run_order_up_dataset]   status → $STATUS"
echo

# Export everything the child run_one needs.
export RUNNER LOG_DIR STATUS
export ROLE_B_ARG CLIP_PLAYER_PROFILE_OVERRIDE DURATION

run_one() {
  local tid="$1"
  local profile=""
  # Profile precedence:
  #  1. --clip-player-profile CLI flag (global override) → pin per tid
  #  2. otherwise pass NOTHING and let run_order_up_eval.sh's manifest
  #     auto-detect (compound-main → restate-on-partial-ack, else default).
  if [[ -n "$CLIP_PLAYER_PROFILE_OVERRIDE" ]]; then
    profile="$CLIP_PLAYER_PROFILE_OVERRIDE"
  fi
  local log="$LOG_DIR/$tid.log"
  local args=()
  [[ -n "$ROLE_B_ARG" ]] && args+=("--role-b" "$ROLE_B_ARG")
  [[ -n "$profile" ]] && args+=("--clip-player-profile" "$profile")
  local t0 t1 dur score role_b_disp
  role_b_disp="${ROLE_B_ARG:-${ROLE_B:-openai}}"
  t0=$(date +%s)
  if [[ -n "$DURATION" ]]; then
    if "$RUNNER" "$tid" "$DURATION" "${args[@]}" > "$log" 2>&1; then
      t1=$(date +%s); dur=$((t1 - t0))
      score=$(grep -aoE 'task_score=[0-9.]+' "$log" | tail -1 | cut -d= -f2 || true)
      echo "OK    $(date +%H:%M:%S)  ${dur}s  $role_b_disp/${profile:-auto}  $tid  score=${score:-?}" >> "$STATUS"
    else
      t1=$(date +%s); dur=$((t1 - t0))
      echo "FAIL  $(date +%H:%M:%S)  ${dur}s  $role_b_disp/${profile:-auto}  $tid" >> "$STATUS"
    fi
  else
    if "$RUNNER" "$tid" "${args[@]}" > "$log" 2>&1; then
      t1=$(date +%s); dur=$((t1 - t0))
      score=$(grep -aoE 'task_score=[0-9.]+' "$log" | tail -1 | cut -d= -f2 || true)
      echo "OK    $(date +%H:%M:%S)  ${dur}s  $role_b_disp/${profile:-auto}  $tid  score=${score:-?}" >> "$STATUS"
    else
      t1=$(date +%s); dur=$((t1 - t0))
      echo "FAIL  $(date +%H:%M:%S)  ${dur}s  $role_b_disp/${profile:-auto}  $tid" >> "$STATUS"
    fi
  fi
}
export -f run_one

# xargs -P drives concurrency. With -P 1 it's strictly sequential.
< "$TASK_IDS_FILE" xargs -I{} -P "$MAX_CONCURRENCY" bash -c 'run_one "$1"' _ {}

echo
echo "[run_order_up_dataset] all done — see $STATUS"
n_ok=$(grep -c '^OK' "$STATUS" 2>/dev/null) || n_ok=0
n_fail=$(grep -c '^FAIL' "$STATUS" 2>/dev/null) || n_fail=0
echo "[run_order_up_dataset] OK=$n_ok  FAIL=$n_fail  TOTAL=$((n_ok + n_fail))"
