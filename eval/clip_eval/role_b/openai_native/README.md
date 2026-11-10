# Native OpenAI Realtime adapter

Copyright 2026 DoorDash, Inc.

Licensed under the Apache License, Version 2.0. See the repository's
top-level `LICENSE.txt` file.

This adapter implements the benchmark's Role B contract directly against the
OpenAI Realtime WebSocket API, without Pipecat.

Run one task with the native adapter:

```bash
eval/clip_eval/runtime/run_order_up_eval.sh orderup-0052 90 \
  --role-b openai-native --offline
```

`OPENAI_API_KEY` must be set. The runtime still starts its local token server
because Role A uses it; the native Role B adapter uses the server-side API key
directly.

The audio path is deliberately small:

1. Receive customer PCM from the local WebRTC bridge at 48 kHz.
2. Resample to OpenAI Realtime's 24 kHz PCM format and send 20 ms WebSocket
   chunks.
3. Receive streamed 24 kHz response PCM, resample it to 48 kHz, and pace it
   onto the benchmark WebRTC track in 10 ms frames.

Tools received through `configure_tools` are registered in `session.update`.
Function calls are forwarded to the benchmark tool server, returned as
`function_call_output` conversation items, and followed by one
`response.create` after the tool response completes.
