# runtime/ — Orchestration

Glues Role A + Role B + the tool server together for a live eval run.
Owns process lifecycle, port allocation, WebRTC signalling, and per-run
output dirs.

## Layout

- `orchestrator.js` — WebRTC signalling server. Brokers SDP offer/answer
  between Role A and Role B, mixes their audio, records `A.wav` /
  `B.wav` / `combined.wav` into `RECORD_DIR`.
- `domains/order_up/`
  - `order_up_tool_server.js` — Order Up cart tool handlers (`add_item`,
    `remove_item`, `submit_cart`, `get_cart`).
  - `score_order_up_cart.py` — post-run cart-diff scorer.
  - `run_order_up_eval.sh` — entry point: starts tool server →
    orchestrator → token server → Role A → Role B → waits → scores.
  - `run_order_up_dataset.sh` — runs the full split sequentially.
- `prompts/order_up.json` — split index (one entry per buildable task).

## Running

From `v2/`:

```sh
OPENAI_REALTIME_MODEL=gpt-realtime-2 \
  bash runtime/domains/order_up/run_order_up_eval.sh <task_id> [duration_seconds]
```

Writes everything into `artifacts/outputs/OrderUp/<task_id>/<run_ts>/`.

## Contracts

- Inputs from pipeline: [`docs/contracts/10_pipeline_to_runtime.md`](../docs/contracts/10_pipeline_to_runtime.md)
- Internal A ↔ B audio: [`docs/contracts/20_role_a_role_b_audio.md`](../docs/contracts/20_role_a_role_b_audio.md)
- Role B contract (what the runtime promises the adapter): [`docs/contracts/30_role_b_protocol.md`](../docs/contracts/30_role_b_protocol.md)
- Tool server: [`docs/contracts/40_tool_server.md`](../docs/contracts/40_tool_server.md)
- Outputs to visualization: [`docs/contracts/50_visualization_inputs.md`](../docs/contracts/50_visualization_inputs.md)
