# runtime/ — Orchestration

Glues Role A + Role B + the tool server together for a live eval run.
Owns process lifecycle, port allocation, WebRTC signalling, and per-run
output dirs.

## Layout

- `audio_bridge.js` — WebRTC signalling and audio bridge. Brokers SDP offer/answer
  between Role A and Role B, mixes their audio, records `A.wav` /
  `B.wav` / `combined.wav` into `RECORD_DIR`.
- `openai_token_server.js` — local credential broker used by the Role A
  OpenAI Realtime routing session.
- `order_up_tool_server.js` — Order Up cart tool handlers.
- `run_order_up_eval.sh` — entry point: starts tool server → audio bridge →
  token server → Role A → Role B → waits → scores.
- `run_order_up_dataset.sh` — runs a set of tasks, with optional concurrency.
- `../scoring/score_order_up_cart.py` — post-run cart-diff scorer.

## Running

From `eval/`:

```sh
OPENAI_REALTIME_MODEL=gpt-realtime-2 \
  bash clip_eval/runtime/run_order_up_eval.sh <task_id> [duration_seconds]
```

Writes everything into `artifacts/outputs/OrderUp/<task_id>/<run_ts>/`.

OpenAI Realtime has two Role B implementations:

- `--role-b openai` (default) uses the provider-neutral Pipecat adapter.
- `--role-b openai-native` uses the direct OpenAI Realtime WebSocket
  adapter. It keeps the same local bridge, task inputs, tools, recordings,
  and scorer while removing Pipecat from Role B, making it useful for
  measuring framework overhead.

See [RUNNING_BENCHMARKS.md](../../../RUNNING_BENCHMARKS.md) for setup and the
full list of outputs. The Role B process contract is described in
[`role_b/pipecat/PLAN.md`](../role_b/pipecat/PLAN.md).
