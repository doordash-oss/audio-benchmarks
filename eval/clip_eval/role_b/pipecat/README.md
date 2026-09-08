# Pipecat Role B adapter

This is the benchmark's provider-neutral implementation of Role B, the agent
under test. It connects to the local audio bridge over WebRTC, runs the selected
speech provider through Pipecat, executes cart tools through the local tool
server, and emits normalized transcript/lifecycle events.

Supported provider names are `openai`, `gemini_live`, `ultravox`,
`grok_realtime`, and `cascaded`. OpenAI is the default used by
`run_order_up_eval.sh`.

## Dependencies

Use Python 3.11 or newer and install [requirements.txt](requirements.txt) into
the eval virtual environment:

```bash
cd eval
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r clip_eval/role_b/pipecat/requirements.txt
```

## Run through the benchmark

From `eval/`:

```bash
OPENAI_API_KEY=... \
  bash clip_eval/runtime/run_order_up_eval.sh <task_id> --role-b openai
```

Use `--role-b gemini`, `--role-b ultravox`, `--role-b grok`, or
`--role-b cascaded` for the other runner presets. Each provider requires its
documented API key.

## Transport-only test

The handshake test starts the repository-owned audio bridge and connects two
Python WebRTC peers without contacting a model provider:

```bash
cd eval/clip_eval/role_b/pipecat
../../../.venv/bin/python selftest_handshake.py
```

See [PLAN.md](PLAN.md) for the process, signaling, and verification contracts.
