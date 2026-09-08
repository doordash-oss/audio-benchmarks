# Role B provider adapter architecture

Role B is the agent under test in the Order Up benchmark. The implementation
uses Pipecat to normalize provider-specific realtime audio, transcription, and
tool-calling behavior behind one local process contract.

## Runtime path

```text
run_order_up_eval.sh
  -> runtime/audio_bridge.js
  -> role_a/clipPlayer_adapter.js
  -> role_b/pipecat/pipecat_role_b_adapter.py
  -> selected speech provider
  -> runtime/order_up_tool_server.js
```

All repository-owned components in this path are released under Apache-2.0.
Pipecat and its transitive dependencies retain their respective upstream
licenses and are installed as package dependencies rather than copied into the
repository.

## Role B process contract

The runner supplies:

- CLI configuration: role, signaling URL, provider, model, voice, system
  prompt, VAD mode, and transcription behavior.
- A newline-delimited `configure_tools` message on stdin containing tool
  schemas and the local tool-server URL.
- A bidirectional 48 kHz mono PCM16 WebRTC track negotiated through the local
  `/signal` endpoint.

The adapter supplies:

- Provider audio on its outgoing WebRTC track.
- Tool requests to `POST {serverUrl}/api/tool` with `{name, arguments}`.
- Newline-delimited transcript and `agent_stop` events on stdout.
- Predictable SIGINT/SIGTERM shutdown.

## Signaling contract

Both roles are WebRTC offerers. After connecting to `/signal`, a role sends:

```json
{"role":"B","type":"offer","sdp":"..."}
```

ICE candidates use:

```json
{"role":"B","type":"candidate","candidate":{"candidate":"...","sdpMid":"0","sdpMLineIndex":0}}
```

The audio bridge replies with role-tagged `answer` and `candidate` messages.
It records outgoing Role A audio to `A.wav`, outgoing Role B audio to `B.wav`,
and creates a stereo `combined.wav` during shutdown.

## Providers

The adapter currently recognizes:

- `openai`
- `gemini_live`
- `ultravox`
- `grok_realtime`
- `cascaded`

New providers should be added behind the same process and tool contracts. A
provider implementation must not alter task generation, Role A routing, cart
semantics, scoring, or output layout.

## Verification gates

Before treating a provider as release-ready:

1. Run `selftest_handshake.py` to verify both WebRTC peers connect and receive
   audio tracks.
2. Run one representative task and confirm `A.wav`, `B.wav`, `combined.wav`,
   `tool_log.json`, `final_cart.json`, and `score.json` are produced.
3. Verify at least one expected tool call reaches the tool server.
4. Review transcript, interruption, timing, and cart-score outputs.
5. Repeat across the release task set and record the exact provider/model
   configuration in the run manifest.
