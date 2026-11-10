# Audio Benchmarks

Order-Up-Bench measures how well a speech agent can take a restaurant order
in a multi-turn spoken conversation. An examiner plays recorded clips of real
customer speech, the agent under test responds and calls cart tools, and the
final cart is compared with the expected order.

This repository contains:

- `eval/` — the evaluation harness: the clip-playing customer simulator
  (Role A), adapters for the model under test (Role B), the cart tool server,
  and the scoring and turn-taking metrics.
- `index.md`, `_includes/`, `_data/`, `assets/` — the source of the benchmark
  report, built with Jekyll into `docs/` by `./bin/build`.

Benchmark tasks and audio are distributed separately as a Hugging Face
dataset, `doordash/doordash-audio-benchmarks`, which the harness downloads at
a pinned revision.

Supported Role B providers: OpenAI Realtime (through Pipecat or a direct
WebSocket adapter), Gemini Live, Grok Voice, Ultravox, and a cascaded
Whisper + LLM + TTS baseline.

## Getting started

See [RUNNING_BENCHMARKS.md](RUNNING_BENCHMARKS.md) for complete setup and
execution instructions.

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License

Original software developed for this repository is licensed under the
[Apache License 2.0](LICENSE.txt).

This license does not relicense third-party or vendored components, benchmark
audio, datasets, or other externally sourced artifacts. Those materials remain
subject to their respective license terms. Bundled third-party software is
identified in [NOTICE.txt](NOTICE.txt).
