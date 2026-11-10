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

"""Pre-launch setup for run_order_up_eval.sh.

Hydrates the artifact cache, validates the task/tree files exist, then
computes everything the caller needs before it can start processes:
expected-cart emptiness (skip signal), session id, eval duration, three
free ports (tool/signal/token server), the examinee system prompt, and
the configure_tools payload.

Emits the results as `export KEY=VALUE` shell statements on stdout,
meant to be consumed via `eval "$(order_up_eval_setup.py ...)"`.

Usage:
  order_up_eval_setup.py <task_id> <task_file> <tree_file> <sessions_root> \
      <duration_mult> <duration_min> [duration_arg] [--offline]
"""
import argparse
import json
import math
import shlex
import socket
import sys
import time
from pathlib import Path


def free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def emit(pairs: dict) -> None:
    for key, value in pairs.items():
        print(f"export {key}={shlex.quote(str(value))}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task_id")
    p.add_argument("task_file")
    p.add_argument("tree_file")
    p.add_argument("sessions_root")
    p.add_argument("duration_mult", type=float)
    p.add_argument("duration_min", type=float)
    p.add_argument("duration_arg", nargs="?", default="")
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()

    if not args.offline:
        try:
            from clip_eval.pipeline.artifact_store import cache_pull_stage
            t0 = time.time()
            cache_pull_stage(args.task_id)
            print(f"[run_order_up] cache_pull_stage OK in {time.time() - t0:.1f}s", file=sys.stderr)
        except Exception as e:
            print(f"[run_order_up] warn: cache_pull_stage failed: {e} (will rely on local cache)", file=sys.stderr)

    task_file = Path(args.task_file)
    tree_file = Path(args.tree_file)
    if not task_file.is_file():
        print(f"Missing task file: {task_file}", file=sys.stderr)
        sys.exit(1)
    if not tree_file.is_file():
        print(f"Missing decision tree: {tree_file}", file=sys.stderr)
        sys.exit(1)

    task = json.loads(task_file.read_text())

    if len(task.get("expected_cart") or []) == 0:
        emit({"SKIP_EMPTY_CART": "1"})
        return

    session_id = task.get("session_id") or ""

    if args.duration_arg:
        duration = int(args.duration_arg)
    else:
        # Prefer the session alignment file when present locally. The
        # published dataset ships only the derived window in the task
        # config, not stage_alignment.json.
        src_dur = 0.0
        try:
            align_path = Path(args.sessions_root) / session_id / "stage_alignment.json"
            alignment = json.loads(align_path.read_text())
            game = alignment["games"][task["game_index"]]
            src_dur = game["audio_end"] - game["audio_start"]
        except Exception:
            src_dur = float(task.get("source_window_seconds") or 0.0)
        if src_dur <= 0:
            # Warn loudly: with no source window the run falls back to the
            # duration floor and may be cut short.
            print(
                "[run_order_up] WARNING: no source window for this task "
                "(no stage_alignment.json and no source_window_seconds in the "
                "task config); falling back to the duration floor",
                file=sys.stderr,
            )
        duration = int(max(args.duration_min, math.ceil(src_dur * args.duration_mult)))

    tool_port = free_port()
    signal_port = free_port()
    token_port = free_port()
    tool_url = f"http://localhost:{tool_port}"

    tools_json = json.dumps({
        "type": "configure_tools",
        "tools": task["tools"],
        "serverUrl": tool_url,
    })

    emit({
        "SESSION_ID": session_id,
        "DURATION": duration,
        "TOOL_PORT": tool_port,
        "TOOL_URL": tool_url,
        "SIGNAL_PORT": signal_port,
        "TOKEN_PORT": token_port,
        "PROMPT_B": task["examinee_system_prompt"],
        "TOOLS_JSON": tools_json,
    })


if __name__ == "__main__":
    main()
