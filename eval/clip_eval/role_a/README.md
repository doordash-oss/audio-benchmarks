# role_a_clip_player/ — Role A: clip-player customer

The customer side of the eval. Plays pre-recorded customer audio clips
into the orchestrator and uses an OpenAI Realtime session as a routing
oracle to decide which clip plays next based on what the agent (Role B)
says.

## Files

- `clipPlayer_adapter.js` — the single adapter. Connects to:
  - orchestrator (`--signalUrl`) for bidirectional audio
  - OpenAI (via `--tokenServer` for ephemeral keys) as the routing model
  - the decision tree (`--decisionTree <path>`) for the clip set + state machine

## Inputs

- `artifacts/decision_trees/<task_id>.json` — routing tree
- `artifacts/clips/<task_id>/*.wav` — customer clips (paths resolved via `tree.clips_base_dir`)
- `artifacts/background_noise/<sid>/noise_*.wav` — optional ambient noise

## Runtime invocation

Invoked by `runtime/run_order_up_eval.sh`; not run
standalone.
