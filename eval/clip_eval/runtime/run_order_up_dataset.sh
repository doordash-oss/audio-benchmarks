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

# Run a set of Order Up tasks against a Role B model, with optional
# concurrency.
#
# Usage:
#   run_order_up_dataset.sh [jobs_file] \
#     [--role-b openai|openai-native|openai-fdb|gemini|ultravox|grok|cascaded] \
#     [--max-concurrency N] \
#     [--log-dir <path>] \
#     [--dry-run] \
#     [--offline]
#
# --dry-run: resolve and print the task list (with count), then exit
# without running anything.
#
# --offline: restrict the default task-list resolution to the local
# artifact cache (skip GCS existence checks) and forward --offline to
# run_order_up_eval.sh for each task.
#
# Task list source:
#   - If a positional <jobs_file> is provided, read task_ids from it
#     (one per line; blank lines and `#` comments ignored).
#   - Otherwise default to the V1 benchmark working set, read from
#     $BENCH_MD (defaults to `<repo>/eval/V1_BENCHMARK_SET.md`, adjacent
#     to this eval tree). Every row in that file is already the working
#     set — no status filtering, no profile column. Per-stage rows are
#     translated to published task ids via task_id_map() when one is
#     available, and emitted unchanged otherwise.
#
#     Set PREFILTER=1 to additionally require a usable decision tree and a
#     complex task config. That is opt-in on purpose: the checks go through
#     the artifact store (local cache OR GCS), which a fresh checkout
#     cannot answer — hydration is per-task inside the runner — so
#     prefiltering there would drop every task. See resolve_v1_task_ids.py.
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
LOG_DIR="/tmp/order_up_dataset"
DRY_RUN=0
OFFLINE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --role-b) ROLE_B_ARG="$2"; shift 2 ;;
    --role-b=*) ROLE_B_ARG="${1#*=}"; shift ;;
    --max-concurrency|-j) MAX_CONCURRENCY="$2"; shift 2 ;;
    --max-concurrency=*) MAX_CONCURRENCY="${1#*=}"; shift ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --offline) OFFLINE=1; shift ;;
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
  # Default: V1 benchmark working set — every per-stage row, translated to
  # published task ids when a task_id_map is available. Set PREFILTER=1 to
  # additionally require a usable tree + complex task config (internal runs
  # only; see resolve_v1_task_ids.py).
  # The manifest lives adjacent to this eval tree (eval/V1_BENCHMARK_SET.md);
  # override with $BENCH_MD.
  BENCH_MD="${BENCH_MD:-$SCRIPT_DIR/../../V1_BENCHMARK_SET.md}"
  if [[ ! -f "$BENCH_MD" ]]; then
    echo "Missing $BENCH_MD — needed for the default benchmark task list" >&2
    exit 1
  fi
  # Task-list resolution lives in resolve_v1_task_ids.py, not inline here:
  # it translates the manifest's internal ids to published ids via
  # task_id_map() (emitting them unchanged when no map exists), and only
  # prefilters on artifact availability when asked. Prefiltering is opt-in
  # because a fresh checkout has nothing on disk — hydration is per-task
  # inside the runner — so filtering on presence would drop every task and
  # the run would die with "No tasks to run".
  RESOLVE_FLAGS=()
  [[ "$OFFLINE" == "1" ]] && RESOLVE_FLAGS+=(--offline)
  [[ "${PREFILTER:-0}" == "1" ]] && RESOLVE_FLAGS+=(--prefilter)
  PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$REPO_ROOT/.venv/bin/python" "$SCRIPT_DIR/resolve_v1_task_ids.py" \
      "$BENCH_MD" "$ARTIFACTS_PATH" "${RESOLVE_FLAGS[@]+"${RESOLVE_FLAGS[@]}"}" > "$TASK_IDS_FILE"
  TASK_SOURCE="$BENCH_MD (V1 working set)"
fi

N_TASKS=$(wc -l < "$TASK_IDS_FILE" | tr -d ' ')
if (( N_TASKS == 0 )); then
  echo "No tasks to run (source: $TASK_SOURCE)" >&2; exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[run_order_up_dataset] DRY RUN — $N_TASKS tasks (source: $TASK_SOURCE)"
  cat "$TASK_IDS_FILE"
  exit 0
fi

mkdir -p "$LOG_DIR"
STATUS="$LOG_DIR/STATUS.txt"
: > "$STATUS"

echo "[run_order_up_dataset] $N_TASKS tasks (source: $TASK_SOURCE)"
echo "[run_order_up_dataset]   role_b=${ROLE_B_ARG:-${ROLE_B:-openai}} max_concurrency=$MAX_CONCURRENCY"
echo "[run_order_up_dataset]   logs → $LOG_DIR/<tid>.log"
echo "[run_order_up_dataset]   status → $STATUS"
echo

# Hydrate the whole sweep in one pass before fanning out. Each worker would
# otherwise pull its own task inside order_up_eval_setup.py via
# cache_pull_stage(tid), which streams the dataset and linear-scans for that
# single id (~75s/task cold), so a full sweep re-downloads the same shards
# once per task. After this the per-worker calls hit a warm cache and return
# immediately. Non-fatal: if it fails, the workers still hydrate individually.
if [[ "$OFFLINE" != "1" ]]; then
  if PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      "$REPO_ROOT/.venv/bin/python" \
      "$SCRIPT_DIR/prehydrate_tasks.py" "$TASK_IDS_FILE"; then
    echo
  else
    echo "[run_order_up_dataset] warn: pre-hydration failed; each worker will" \
         "hydrate its own task (much slower)" >&2
  fi
fi

# Export everything the child run_one needs.
export RUNNER LOG_DIR STATUS
export ROLE_B_ARG OFFLINE

run_one() {
  local tid="$1"
  local log="$LOG_DIR/$tid.log"
  local args=()
  [[ -n "$ROLE_B_ARG" ]] && args+=("--role-b" "$ROLE_B_ARG")
  [[ "$OFFLINE" == "1" ]] && args+=("--offline")
  local t0 t1 dur role_b_disp
  role_b_disp="${ROLE_B_ARG:-${ROLE_B:-openai}}"
  t0=$(date +%s)
  if "$RUNNER" "$tid" "${args[@]+"${args[@]}"}" > "$log" 2>&1; then
    t1=$(date +%s); dur=$((t1 - t0))
    echo "OK    $(date +%H:%M:%S)  ${dur}s  $role_b_disp  $tid" >> "$STATUS"
  else
    t1=$(date +%s); dur=$((t1 - t0))
    echo "FAIL  $(date +%H:%M:%S)  ${dur}s  $role_b_disp  $tid" >> "$STATUS"
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
