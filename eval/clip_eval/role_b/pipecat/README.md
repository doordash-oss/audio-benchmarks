# Pipecat Role B adapter (experimental)

A Python, Pipecat-based implementation of the swappable **Role B** (agent-under-test)
contract for the Order Up benchmark — the path to evaluating S2S models beyond the
native OpenAI GPT-realtime adapter. See `PLAN.md` (here) for the full design and the
contract at `clip_eval/docs/contracts/30_role_b_protocol.md`.

Status: **functional, multi-provider** — connects to the orchestrator over WebRTC, runs the
selected model backend via Pipecat, takes the order via cart tools, and emits `agent_stop` +
transcripts. Providers (`ROLE_B_PROVIDER`):

| Provider | Backend | Notes |
|---|---|---|
| `openai` | OpenAI Realtime (native S2S) | default; broadly cart-score parity with the native Node adapter (PLAN.md §M7) |
| `gemini_live` | Gemini Live (native S2S) | M8 |
| `grok_realtime` / `grok` | xAI Grok Voice Agent (native S2S) | uses Pipecat `GrokRealtimeLLMService`; requires `XAI_API_KEY` |
| `cascaded` | Silero VAD + OpenAI Whisper STT + gpt-4o LLM + OpenAI TTS | non-realtime **baseline** for comparison; mirrors FDB-v3's cascaded agent |

Known gaps: `inject` is accepted-but-not-wired (Order Up never sends it); turn-taking parity
validated single-run-per-task (N-run rigor is a TODO).

## Setup

Requires Python ≥3.12 (per the repo `pyproject.toml`; Pipecat 1.x). Dependencies live in the repo's `pyproject.toml`
(single source of truth). The benchmark scripts source `scripts/_env.sh`, which points uv at
the project venv (`v2/.venv`) and `uv run` auto-syncs it — so running through
`scripts/eval.sh` needs no manual setup. To pre-sync the venv yourself:

```bash
UV_PROJECT_ENVIRONMENT=v2/.venv uv sync
```

(The historical per-adapter `shared/role_b/pipecat/.venv` is no longer used — its deps are
folded into the project venv. If the directory still exists locally it's safe to delete.)

## Run via the benchmark runner

The runner (`clip_eval/runtime/run_order_up_eval.sh`) launches Role B from `$ROLE_B_CMD`,
defaulting to the native Node adapter. To use this one:

```bash
export ROLE_B_CMD="uv run python $PWD/shared/role_b/pipecat/pipecat_role_b_adapter.py"
export ROLE_B_PROVIDER=cascaded      # or: openai | gemini_live | grok_realtime
bash scripts/eval.sh order_up.Corner_Bistro.546FF09D.g0
```

Optional overrides:
- `ROLE_B_INTERRUPTIBLE=true|false` — overrides the interruption default (derives from the VAD
  profile; see PLAN.md §M7 for why ON matches native's *behavior* on our path).
- `grok_realtime` only: `XAI_API_KEY` (or `GROK_API_KEY`) is required.
  `GROK_REALTIME_MODEL` / `XAI_REALTIME_MODEL` override the default
  `grok-voice-think-fast-1.0`; `GROK_VOICE` / `XAI_VOICE` can be one of
  `Ara`, `Rex`, `Sal`, `Eve`, or `Leo`. The runner shortcut is:
  `bash scripts/eval.sh <task_id> --role-b grok`.
- `cascaded` only: `CASCADED_STT_MODEL` / `CASCADED_LLM_MODEL` / `CASCADED_TTS_MODEL` /
  `CASCADED_TTS_VOICE` swap individual stages (defaults `whisper-1` / `gpt-4o` / `tts-1`; TTS
  voice falls back to the runner's `--voice` (`shimmer`) and then to `nova` standalone).
  `ROLE_B_MODEL`/`--model` is realtime-oriented and ignored for cascaded — use
  `CASCADED_LLM_MODEL` to change the reasoning model. `ROLE_B_INTERRUPTIBLE=true|false`
  controls whether in-flight tool calls survive a customer interruption.

## Standalone smoke (no orchestrator/model)

```bash
printf '%s\n' '{"type":"configure_tools","tools":[{"name":"add_to_cart"}],"serverUrl":"http://x"}' \
  | uv run python pipecat_role_b_adapter.py --role B --signalUrl ws://localhost:9999/signal --systemPrompt hi
```
Expect: logs the parsed tools, then exits cleanly on stdin EOF.
