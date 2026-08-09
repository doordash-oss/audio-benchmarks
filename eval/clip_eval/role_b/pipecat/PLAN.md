# Plan — Pipecat-based Role B adapter (generalize benchmarking beyond GPT realtime)

**Status:** Draft for approval · **Owner:** aiden.tung · **Date:** 2026-05-29
**Canonical plan.** (Supersedes the earlier roadmap draft at
`clip_eval/docs/PIPECAT_ROLE_B_ADAPTER_PLAN.md`, whose process/sequencing ideas are merged in here.)

---

## 1. Executive overview

Generalize the Full-Duplex-Bench "Order Up" real-time benchmark so it can evaluate many
speech-to-speech (S2S) models, not just OpenAI GPT realtime. Tech-lead direction: use a
**Pipecat-style abstraction** to hook in all S2S models (Gemini Live, xAI/"x", Ultravox, …).

We add a NEW experimental **Pipecat-based Role B adapter** (Python) that satisfies the
*existing* Role B contract. Pipecat is used **only** to implement the Role B agent runtime
(connect to provider, stream audio, register tools) — it does **not** replace the benchmark.

Current control path (unchanged, stays default):
```
run_order_up_eval.sh -> shared/role_b/reference/gptRealtime_adapter.js -> OpenAI Realtime
```
New additive path:
```
run_order_up_eval.sh -> shared/role_b/pipecat/pipecat_role_b_adapter.py -> Pipecat -> provider
```

**Sequencing:** prove the Pipecat path **first with OpenAI Realtime** (we have an API key),
compare against the native Node adapter, THEN add Gemini Live, then xAI/Grok / Ultravox.

## 2. Ownership boundary — what stays ours vs what Pipecat owns

**Stays repo-owned (Pipecat must NOT touch):**
- Task generation from Order Up artifacts (`clip_eval/pipeline/*`)
- Role A clip-player customer simulator (`clip_eval/role_a/clipPlayer_adapter.js`)
- Decision-tree routing for customer clips (`decision_trees/*.json`)
- Live audio bridge (`shared/orchestrator.js`)
- Order Up cart tool server (`clip_eval/runtime/order_up_tool_server.js`)
- Post-run cart scoring + behavior audits (`clip_eval/runtime/score_order_up_cart.py`, etc.)
- Output artifact layout

**Pipecat owns (inside the Role B adapter only):**
- Connecting Role B to the provider/model service
- Streaming audio to/from the model
- Registering tools where supported
- Normalizing provider-specific realtime details

## 3. The Role B contract we must honor (`clip_eval/docs/contracts/30_role_b_protocol.md`)

A Role B adapter is ONE long-running process driven over these surfaces:

1. **CLI args:** `--role B` (req), `--signalUrl ws://host:port/signal` (req),
   `--tokenServer` (optional), `--model`, `--voice`, `--systemPrompt`,
   `--turnMode server_vad|none`, `--vadMode slow|fast`, `--speakFirst` (flag),
   `--enableTranscription` (flag), `--prefill`/`--prefillFile`.
   **New (additive, impl-defined per contract §8):** `--provider openai|gemini_live|ultravox|grok`,
   with `--model` carrying the provider-specific model id.
2. **stdin** (JSON lines; **queue & replay if model channel not ready yet**):
   - `{type:'configure_tools', tools:[{name,description,parameters}], serverUrl}`
   - `{type:'inject', text, trigger_response, trigger_response_delay}`
3. **HTTP tool calls:** for each model tool call, `POST {serverUrl}/api/tool` with
   body `{name, arguments}`; response `{result:<json>}`; feed result back to model.
   Adapter NEVER executes tools.
4. **stdout** (JSON lines): `{type:'event', event:'agent_stop'}` when the agent finishes its
   final utterance; `{type:'transcript_delta',source,delta}` / `{type:'transcript',source,text}`
   when `--enableTranscription`.
5. **WebRTC:** open WS to `--signalUrl`, act as the **offerer**, exchange a single
   bidirectional audio track at 48 kHz mono PCM16 (build to ten 10 ms frames / 100 ms tick).
6. **Lifecycle:** clean exit on SIGTERM/SIGINT and on stdin close; honor `$RECORD_DIR`
   (cwd set by runtime; stderr → `$RECORD_DIR/adapter_B.log`).

### `/signal` protocol (from `shared/orchestrator.js`)
- Adapter is the **offerer**. On WS connect the orchestrator sends `{type:'ready'}`.
- Adapter sends `{role:'B', type:'offer', sdp}` then ICE
  `{role:'B', type:'candidate', candidate:{candidate, sdpMid, sdpMLineIndex}}`.
- Orchestrator waits for BOTH A and B offers, then replies `{role:'B', type:'answer', sdp}`
  and broadcasts ICE candidates tagged with `role`.
- Audio: orchestrator records the track B *sends* → `B.wav`; forwards A's audio to B.
  48 kHz mono PCM16 (`WIRE_SAMPLE_RATE`, default 48000).

## 4. Pipecat / aiortc facts — **VERIFY AT SETUP** (pinned from research spike 2026-05-29)

> These are point-in-time facts; re-verify versions/imports when creating the env.

- **pipecat-ai 1.3.0**, requires **Python ≥3.11** (3.10 unsupported; **3.14 unverified** — the
  repo `.venv` is 3.14, conda `fdb2` is 3.10, so BOTH are unusable). → dedicated venv.
  **Verified locally: `python3.11` IS installed (`/opt/homebrew/bin/python3.11`); `python3.12` is
  NOT.** Use 3.11 (Pipecat-supported, available) or `uv python install 3.12` if we prefer 3.12.
  Install `pipecat-ai[openai,webrtc]` + `aiohttp`. (`uv 0.10.3` verified present.)
- **OpenAI Realtime service:** `from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService`.
  Connects to OpenAI over **WebSocket using the API key directly** → **Role B needs NO token server**.
  Config via `OpenAIRealtimeLLMService.Settings(...)` / `SessionProperties` (instructions, voice
  under `AudioOutput(voice=...)`, `TurnDetection`/`SemanticTurnDetection`, `InputAudioTranscription`).
- **Sample rates:** Realtime is hard-locked to **24000 Hz** and does NOT resample input.
  → resample A's 48k → 24k in the input transport (`audio_in_sample_rate=24000`); output 24k → 48k
  is auto-resampled by `BaseOutputTransport` (`audio_out_sample_rate=48000`). **(48 kHz is only the
  benchmark-boundary rate; do not feed 48 kHz to the Realtime service.)**
- **Tools:** `FunctionSchema`/`ToolsSchema` (`pipecat.adapters.schemas.*`);
  `llm.register_function(name, handler)`; handler gets `FunctionCallParams`, does the HTTP POST,
  calls `await params.result_callback(result_dict)`.
- **agent_stop:** `BotStoppedSpeakingFrame` (`pipecat.frames.frames`), observed via a `FrameProcessor`
  after `transport.output()`. (`TTSStoppedFrame` is earlier = generation done, not playout.)
- **Transcripts:** `TranscriptionFrame`/`InterimTranscriptionFrame` (user), `LLMTextFrame` (bot).
- **Custom transport:** subclass `BaseInputTransport`/`BaseOutputTransport`
  (`pipecat.transports.base_input`/`base_output`), driving our own aiortc `RTCPeerConnection`.
  Copy `RawAudioTrack` + the `read_audio_frame` resample loop from
  `pipecat/transports/smallwebrtc/{transport,connection}.py` (tag v1.3.0). Replace ONLY its
  signaling (it's an answerer; we're an offerer on a custom WS protocol).
- **Pipeline:** `Pipeline([input, llm, AgentStopSignal(), output])` via `PipelineWorker` + `WorkerRunner`.
- **Text injection:** `LLMMessagesAppendFrame` is a stub for Realtime in 1.3.0 — use
  `llm.send_client_event(ConversationItemCreateEvent(...))` + `ResponseCreateEvent()`.
- **aiortc 1.14.0:** outgoing track yields 10 ms `av.AudioFrame`s, `pts` advancing by
  samples-per-frame, `time_base=Fraction(1, sample_rate)`, paced to wall clock, silence when empty.
  Incoming: `await track.recv()` → `av.AudioResampler` → PCM16. May need
  `force_transceivers_to_send_recv()` for one bidi m-line.

## 5. File layout (all NEW; Node GPT adapter + Role A untouched)
```
shared/role_b/pipecat/
  PLAN.md                      # this file (canonical)
  pipecat_role_b_adapter.py    # entry: CLI, signals, stdin/stdout, wiring
  orchestrator_transport.py    # custom Pipecat transport: aiortc + /signal offerer
  raw_audio_track.py           # OutgoingAudioTrack (adapted from SmallWebRTC RawAudioTrack)
  pyproject.toml + uv.lock     # Python 3.12 env, pipecat-ai[openai,webrtc], aiohttp
  README.md                    # how to run + per-model notes
```

## 6. Milestone ladder (incremental; each independently testable)

### M0 — Prerequisites verification + baseline capture (no new code)
Run the §9 prereq checklist first (harness may not run locally yet). Then run the existing
native Node adapter on 1–2 Order Up tasks; capture reference `prompt.json`, `tool_log.json`,
`final_cart.json`, `score.json`, `adapter_A.log`, `adapter_B.log`, `A.wav`, `B.wav`, `combined.wav`.
**Gate:** native eval runs locally; we have a parity target.

### M1 — Runner configurability (FIRST mergeable PR; default unchanged)
Add `--role-b-cmd` (full command, generalizes to a Python interpreter + args) to
`clip_eval/runtime/run_order_up_eval.sh`, **defaulting to the current Node adapter** (byte-for-byte
unchanged when unset). Confirmed exact integration points (read 2026-05-29):
- Role B launch = lines **192–200** (`node .../gptRealtime_adapter.js --role B --signalUrl ... < $TOOLS_FIFO > adapter_B.log`).
  Replace the `node .../gptRealtime_adapter.js` token with `$ROLE_B_CMD`, keeping the
  `--role B --signalUrl ... < FIFO > adapter_B.log` wiring intact (the FIFO carries `configure_tools`;
  writer at line **190**, `configure_tools` JSON built at line **90**).
- Token server = lines **163–164** (started unconditionally) → gate behind whether the chosen adapter
  needs it (Pipecat Role B does NOT).
- Provenance = `prompt.json` snapshot at lines **101/108** (stamps only model today) → add adapter/provider/model.
**Gate:** no-arg run == today's behavior; explicit Node `--role-b-cmd` == same launch path;
`prompt.json` records adapter/provider/model. *(Done early so all later milestones run through the
real harness, not a throwaway launcher.)*

### M2 — Pipecat adapter skeleton (no WebRTC, no model)
`pipecat_role_b_adapter.py`: parse Role B CLI args, async-read stdin, queue `configure_tools`
until runtime ready, clear startup/shutdown logs, clean exit on signals/stdin close.
**Gate:** runner can launch it via `--role-b-cmd`; it receives the same prompt/tool config as the
Node adapter; predictable logs; clean exit.

### M3 — WebRTC transport loopback (no Pipecat pipeline, no model) — riskiest piece, isolated
`orchestrator_transport.py` + `raw_audio_track.py`: WS `/signal`, aiortc PC as offerer,
offer/ICE/answer, receive A's track, echo it back (or emit a test tone) on B's track.
**Gate:** orchestrator accepts the Python adapter as Role B; `A.wav`/`B.wav` produced;
`B.wav` contains the expected echo/tone; clean 48 kHz, no drift/stall over ~60 s.

### M4 — Pipecat + OpenAI Realtime, audio only (no tools)
Wrap transport in base in/out classes; Pipeline `input → OpenAIRealtimeLLMService → output` via
`PipelineWorker`/`WorkerRunner`. Map `--systemPrompt`/`--voice`, `--vadMode`→turn detection,
`audio_in_sample_rate=24000`, `audio_out_sample_rate=48000`.
**Gate:** model hears the customer and speaks a sane order-taking response into `B.wav`;
`combined.wav` sounds like a real conversation attempt. (No cart scoring yet.)

### M5 — stdin/stdout contract
`agent_stop` via a `FrameProcessor` on `BotStoppedSpeakingFrame`; transcripts when
`--enableTranscription`. **`inject` is accepted + logged but NOT wired into the live session
(TODO)** — the Order Up runner never sends it, so it's an unexercised contract gap.
**Gate (met):** stdout emits exactly the contract JSON lines; existing scorer/audit scripts don't break.

### M6 — Tool calls (plumbing gate — NOT a score gate)
On `configure_tools`, convert task tools → Pipecat schemas, register handlers that
`POST {serverUrl}/api/tool` with `{name, arguments}`, read `{result}`, `result_callback(result)`.
**Gate (plumbing only — a model may validly fail the task while plumbing works):**
- `add_to_cart`/`remove_from_cart`/`get_cart`/`submit_order` calls reach `order_up_tool_server.js`
- ≥1 tool call reaches the tool server
- `tool_log.json` and `final_cart.json` written
- `score_order_up_cart.py` runs to completion
(Do NOT gate on a non-zero score — that conflates plumbing with model performance.)

### M7 — OpenAI parity run vs baseline
Run the M0 task set both ways (native Node vs Pipecat+OpenAI). Compare cart score, tool-call
sequence, missed/corrected items, submit timing, latency metrics, interruptions, transcript
quality, audio timing/drift.
**Parity threshold (define before running; proposal):** Pipecat completes full scored evals on the
baseline tasks; cart score within a small agreed delta of native OR all differences explainable;
tool-call sequences functionally equivalent. Native stays default until the team explicitly switches.

**M7 FINDING (2026-05-31, task order_up.Corner_Bistro.546FF09D.g0):**
- Native baseline cart score: **0.75** (complete order).
- Pipecat with interruptions OFF (native's literal `interrupt_response:false` fast config):
  **0.57**, order auto-closed early at 3/7 items — the clip player hit "no agent speech
  within 8000ms" because our path's timing overlaps (agent batches a long reply, ignores
  the next clip while non-interruptible) where native's clean lockstep never overlaps.
- Pipecat with interruptions **ON**: **0.857** and **0.714** over two runs, both completing
  the full order — brackets native's 0.75.
- DECISION: default `_RoleBRealtimeLLM` to **interruptible=ON** (commit ef187d3) — match
  native's BEHAVIOR/outcome, not its literal flag. `ROLE_B_INTERRUPTIBLE=true/false` overrides.
- NOTE: response_rate stayed ~0.5 even at cart 0.857 — it is a misleading proxy; the cart
  score is the parity metric.
- STILL TODO for full confidence: validate across SEVERAL Order Up tasks (only Corner Bistro
  so far) and capture native's own run-to-run variance (only 1 native run).

### M8 — Provider selection
Expose `--provider openai|gemini_live|ultravox|grok` (+ provider-specific `--model`). Provider
behavior lives inside the Pipecat adapter; the runner needs no provider-specific changes.
**Gate:** switching providers changes nothing in task gen / Role A / orchestrator / tool server / scorer.

### M9 — Gemini / Ultravox / xAI smoke tests (future)
One-task smoke test per provider before larger comparisons.
**Ultravox note (verify at this milestone):** the open-weights `fixie-ai/ultravox` model is
audio-in→**text**-out (would need a TTS stage), BUT Pipecat's Ultravox *realtime* service / the
hosted Ultravox WebSocket API emits agent **audio** + supports tools (TTS server-side), so via that
path no separate TTS stage is needed. Confirm the exact mode. Benchmark-validity caveat: if the
audio is server-side TTS rather than native speech-out, the turn-taking we measure is partly the
wrapper's, not the model's.

## 7. Suggested first pull requests
1. **Runner configurability only** (M1) — adapter/provider/model provenance; preserves default. (Safe, mergeable independently.)
2. **Pipecat adapter skeleton** (M2) — CLI/stdin/logging; no WebRTC/model.
3. **Python Role B WebRTC transport** (M3) — connect to orchestrator; echo/silence/tone.
4. **Pipecat OpenAI audio path** (M4) — real model audio in/out; no tools.
5. **Pipecat OpenAI tools** (M6) + stdout contract (M5) — register cart tools; complete scoring.

## 8. Open questions
1. Which 1–2 Order Up tasks are the baseline parity tasks? (need runnable built artifacts)
2. Runner option confirmed as `--role-b-cmd <full command>` (recommended) vs `--adapter-b <path>`?
3. What exact delta counts as "parity enough" for M7 (cart-score tolerance + which metrics are
   advisory vs gating)?
4. Provider order after OpenAI: Gemini Live first (recommended), then Ultravox/xAI?
5. Confirm Python 3.12 for the adapter venv (3.14 unverified for Pipecat).

## 9. Prerequisites — VERIFIED 2026-05-29 (harness IS runnable locally)
- [x] `ffmpeg` on PATH — **ffmpeg 8.1** ✓
- [x] `OPENAI_API_KEY` in `v2/.env` — **present** ✓ (bonus: `GEMINI_API_KEY` also present, ready for M9)
- [x] `v2/node_modules` installed — **yes (224 pkgs)** ✓
- [x] Built task artifacts — **7 tasks built** (`order_up_tasks/*.task.json` + 7 `decision_trees/` + 9 `clips/` dirs) ✓
- [x] `clip_eval/runtime/prompts/order_up.json` split index — **present**, lists task ids ✓
- [x] `uv` available — **uv 0.10.3** ✓
- [x] **Harness has run before** — `v2/artifacts/outputs/OrderUp/` has **7 completed runs** ✓ → M0 does NOT need to build a task.
- [x] Hardcoded `/Users/daniel.fischetti/...` paths are INGESTION-only, not in the eval path ✓

**M0 task pick:** 5 tasks have BOTH built artifacts AND a prior completed run (proven-runnable):
`order_up.Bella_Roma_Trattoria.15F307F2.g1`, `order_up.Blue_Moon_Cafe.58C77CE6.g0`,
`order_up.Corner_Bistro.546FF09D.g0`, `order_up.Ember_Oak.FC5B850B.g3`,
`order_up.Maple_Street_Diner.5D888BDB.g1`. Use 1–2 of these as the baseline/parity tasks.

**Two adjustments flagged by verification:**
- **Node version:** active `node` is **v20.12.2**, but `v2/.nvmrc` pins **22.3.0**. `wrtc`
  (`@roamhq/wrtc`) ships native bindings with a node ABI — run **`nvm use`** (→ 22.3.0) in `v2/`
  before M0 to match how the harness was built. Non-blocking but the safe move.
- **Python:** use the installed **3.11** for the adapter venv (3.12 not present), or `uv python install 3.12`.

## 10. Risks / open technical questions
- aiortc single-track direction may need `force_transceivers_to_send_recv()` (M3).
- Drift control: the Node adapter drops silence to fight OpenAI's ~0.4% fast clock; watch `B.wav`
  length/sync over a full task (M4/M7).
- Routing fairness: Role A routes on a whisper transcript of B's speech, so a parity gap could be
  TTS intelligibility, not ordering logic — note, don't "fix."
- Python 3.14 unverified for Pipecat → separate 3.12 venv sidesteps this.
- Latency metric (`eval_metrics.py`) is anchored on the orchestrator clock with a ~+200 ms bias;
  Pipecat's buffering may shift it — compare with a documented caveat, not 1:1.

## 11. Non-goals (first pass)
- Don't rewrite the existing GPT Realtime adapter.
- Don't make Role A provider-agnostic (stays OpenAI-pinned by design).
- Don't change the scoring model.
- Don't add Gemini/Ultravox/xAI before OpenAI-through-Pipecat finishes a scored eval.
- Don't optimize latency until the basic path works end-to-end.
- Don't make the rest of the repo Python/Pipecat-based — the adapter is just another process that
  satisfies the Role B contract.
```
