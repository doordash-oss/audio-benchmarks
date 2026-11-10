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

"""Postprocess step for run_order_up_eval.sh: write prompt.json.

Snapshots the EXACT prompt, tools, and Role B config an examinee was
given on this run, next to the recording, so historical runs stay
reproducible even if the task config changes later.

Reads everything from the environment (not argv) so a ROLE_B_CMD
containing quotes/backslashes/newlines can't break the generated
snapshot.

Required env vars: TASK_FILE, RECORD_DIR, TOOL_URL, DURATION, PROMPT_B,
ROLE_B_CMD, ROLE_B_PROVIDER, ROLE_B_MODEL.
"""
import json
import os


def main() -> None:
    task = json.loads(open(os.environ["TASK_FILE"]).read())
    snapshot = {
        "task_id": task["task_id"],
        "model": os.environ["ROLE_B_MODEL"],
        "role_b_provider": os.environ["ROLE_B_PROVIDER"],
        "role_b_cmd": os.environ["ROLE_B_CMD"],
        "examinee_system_prompt": os.environ["PROMPT_B"],
        "tools": task["tools"],
        "menu_sample": task.get("menu_sample"),
        "tool_server_url": os.environ["TOOL_URL"],
        "duration_seconds": int(os.environ["DURATION"]),
    }
    out_path = os.path.join(os.environ["RECORD_DIR"], "prompt.json")
    with open(out_path, "w") as f:
        f.write(json.dumps(snapshot, indent=2))


if __name__ == "__main__":
    main()
