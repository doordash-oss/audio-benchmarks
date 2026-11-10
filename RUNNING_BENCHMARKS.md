# Running Order-Up-Bench

This guide explains how to set up and run the Order-Up benchmark from a fresh
clone. Benchmark inputs are downloaded from the versioned Hugging Face dataset
at `doordash/doordash-audio-benchmarks`.

## What the benchmark does

Each run launches four local processes:

1. A tool server that maintains and scores the restaurant cart.
2. A WebRTC orchestrator that connects the two audio participants and records
   `A.wav`, `B.wav`, and `combined.wav`.
3. Role A, the examiner/customer simulator, which selects and plays recorded
   customer clips.
4. Role B, the examinee/model being evaluated, which listens, speaks, and calls
   the cart tools.

The runner downloads the selected task's configuration, decision tree, and
audio clips from Hugging Face into a local cache. At the end it compares
the model's cart with the expected cart and writes behavioral metrics.

## Repository layout

The evaluation harness is rooted under `eval/`:

```text
eval/
├── V1_BENCHMARK_SET.md
├── clip_eval/
│   ├── pipeline/       # Artifact paths and remote cache hydration
│   ├── role_a/         # Recorded-clip customer simulator
│   ├── role_b/         # Model adapters
│   ├── runtime/        # Single-task and dataset runners
│   └── scoring/        # Cart and conversation metrics
└── artifacts/          # Downloaded inputs and generated outputs (gitignored)
```

Run all evaluation commands from `eval/` unless a command says otherwise.

## Prerequisites

You need:

- macOS or Linux.
- Node.js 22 or newer (tested on 22 and 26).
- Python 3.11. Do not use Python 3.14 for the benchmark environment.
- [`uv`](https://docs.astral.sh/uv/) for Python environment management.
- `ffmpeg` for the final stereo recording.
- A Hugging Face account with dataset access while the repository is private.
- An API key for the model provider being evaluated.

On macOS with Homebrew and `nvm`:

```bash
brew install python@3.11 uv ffmpeg
nvm install 22
nvm use 22
```

Confirm the tools:

```bash
node --version
python3.11 --version
uv --version
ffmpeg -version
```

## 1. Install the Node dependencies

From the repository root:

```bash
cd eval
npm install
```

If npm reports root-owned files under `~/.npm`, use a different cache rather
than running npm with `sudo`:

```bash
npm install --cache /tmp/order-up-npm-cache
```

Dependencies are declared in `eval/package.json`; `eval/node_modules/` is
gitignored.

## 2. Create the Python environment

From `eval/`:

```bash
uv venv --python python3.11 .venv

uv pip install --python .venv/bin/python -r requirements.txt
```

The direct `--role-b openai-native` adapter is implemented in Node. The
default `--role-b openai` path and the Gemini, Grok, Ultravox, and cascaded
paths use Pipecat, so install their dependencies with:

```bash
uv pip install --python .venv/bin/python \
  -r clip_eval/role_b/pipecat/requirements.txt
```

Install Pipecat last. It pins an older `websockets` than the block above and
will downgrade it (17.x → 16.1.1); installing in the other order leaves the
Pipecat adapters on an unsupported version.

The benchmark itself needs no audio codec — it reads clips as encoded bytes and
copies them to disk. If you want to inspect the dataset's audio directly, for
example `load_dataset(...)` followed by reading a clip, `datasets` decodes it and
raises `ImportError: To support decoding audio data, please install 'torchcodec'`.
Either install that:

```bash
uv pip install --python .venv/bin/python torchcodec
```

or skip decoding and handle the raw bytes yourself:

```python
ds = load_dataset("doordash/doordash-audio-benchmarks", split="test").decode(False)
```

## 3. Authenticate to Hugging Face

The runtime reads task inputs from:

```text
https://huggingface.co/datasets/doordash/doordash-audio-benchmarks
```

The dataset is currently private. Log in with a Hugging Face user that has
access (authentication will be optional after the dataset becomes public):

```bash
hf auth login
hf auth whoami
```

For automation, set `HF_TOKEN` instead. Never commit the token or paste it into
shell history. `HF_REPO_ID` can point at a fork, and `HF_REVISION` can select a
different immutable dataset commit. The repository defaults pin the reviewed
revision `390df8065efcb2ab74d798d235cf5c42a5ad08b3`.

Verify access from `eval/`:

```bash
PYTHONPATH="$PWD" \
  .venv/bin/python -m clip_eval.pipeline.artifact_store
```

A successful check prints the dataset, resolved commit, privacy state, local
cache root, and several repository files.

The artifact settings are:

| Variable | Default | Purpose |
| --- | --- | --- |
| `HF_REPO_ID` | `doordash/doordash-audio-benchmarks` | Dataset repository |
| `HF_REVISION` | pinned commit | Dataset snapshot to reproduce |
| `HF_TOKEN` | cached login | Noninteractive access token |
| `HF_DATASETS_CACHE` | `eval/artifacts/.hf_cache/datasets` | Download cache |
| `ARTIFACTS_PATH` | `eval/artifacts` | Materialized inputs and run outputs |

To download one task without starting a paid provider run:

```bash
PYTHONPATH="$PWD" .venv/bin/python -c '
from clip_eval.pipeline.artifact_store import cache_pull_stage
cache_pull_stage("orderup-0052")
'
```

This creates the task JSON, decision tree, customer-clip manifest, and WAV
files under `eval/artifacts/`. Repeating the command uses the local cache.

## 4. Configure the model provider

Export the relevant key in your shell or put it in `eval/.env`. The runner loads
`eval/.env` before starting any process, so both paths reach the Node adapters
and the Python scorer and post-run audits. A value already exported in your
shell takes precedence over the same key in `.env`. Never commit the `.env`
file — it is gitignored at any depth.

| Role B | Runner flag | Required credential |
| --- | --- | --- |
| OpenAI Realtime through Pipecat (default) | `--role-b openai` | `OPENAI_API_KEY` |
| OpenAI Realtime direct WebSocket | `--role-b openai-native` | `OPENAI_API_KEY` |
| Gemini Live | `--role-b gemini` | `GEMINI_API_KEY` |
| Grok Voice | `--role-b grok` | `XAI_API_KEY` or `GROK_API_KEY` |
| Ultravox | `--role-b ultravox` | `ULTRAVOX_API_KEY` |
| Cascaded OpenAI baseline | `--role-b cascaded` | `OPENAI_API_KEY` |

Example:

```bash
export OPENAI_API_KEY='...'
```

The default OpenAI model is `gpt-realtime-2`. Override the Role B model with:

```bash
export OPENAI_REALTIME_MODEL='gpt-realtime-2'
```

Other provider overrides include `GEMINI_LIVE_MODEL`,
`GROK_REALTIME_MODEL`, `ULTRAVOX_MODEL`, and `CASCADED_LLM_MODEL`.

## 5. Run one task

From `eval/`:

```bash
OPENAI_REALTIME_MODEL=gpt-realtime-2 \
  bash clip_eval/runtime/run_order_up_eval.sh \
  orderup-0052
```

The runner will:

1. Hydrate the task from Hugging Face.
2. Allocate free local ports.
3. Start the tool server, orchestrator, Role A, and Role B.
4. Stop when the conversation reaches its terminal state or duration limit.
5. Score the final cart and generate behavioral metrics.

The default duration is four times the source conversation length, with a
minimum of 180 seconds. For a short infrastructure probe, pass an explicit
duration:

```bash
SKIP_POST_EVAL=1 \
  bash clip_eval/runtime/run_order_up_eval.sh \
  orderup-0052 \
  30
```

A short probe can confirm connectivity but should not be treated as a valid
benchmark result if it ends at the duration timeout.

### Offline runs

After a task has been cached locally, prevent all remote access with:

```bash
  bash clip_eval/runtime/run_order_up_eval.sh \
  orderup-0052 \
  --offline
```

Offline mode fails when required task or audio files are absent from
`eval/artifacts/`.

## 6. Run a dataset

The dataset runner accepts a plain-text jobs file containing one task ID per
line:

```bash
cat >/tmp/order-up-smoke.txt <<'EOF'
orderup-0052
orderup-0053
EOF

bash clip_eval/runtime/run_order_up_dataset.sh \
  /tmp/order-up-smoke.txt \
  --role-b openai \
  --max-concurrency 2
```

Use conservative concurrency initially. Every task creates two provider
sessions—one for the fixed Role A routing oracle and one for Role B—and incurs
Realtime API usage.

### Generate a jobs file from Hugging Face

The dataset runner accepts Hugging Face task IDs in an explicit jobs file. This
command streams only the dataset records and writes every ID:

```bash
.venv/bin/python -c '
from datasets import load_dataset
rows = load_dataset(
    "doordash/doordash-audio-benchmarks",
    split="test",
    streaming=True,
    revision="390df8065efcb2ab74d798d235cf5c42a5ad08b3",
).decode(False)
for row in rows:
    print(row["task_id"])
' > /tmp/order-up-all.txt

bash clip_eval/runtime/run_order_up_dataset.sh \
  /tmp/order-up-all.txt \
  --role-b openai \
  --max-concurrency 2
```

Dataset logs default to:

```text
/tmp/order_up_dataset/<task_id>.log
/tmp/order_up_dataset/STATUS.txt
```

Set a persistent log directory with `--log-dir PATH`.

## 7. Find and verify results

Each run writes to:

```text
eval/artifacts/outputs/OrderUp/<task_id>/<run_timestamp>/
```

Important files include:

| File | Purpose |
| --- | --- |
| `score.json` | Expected cart, actual cart, detailed diff, tags, and task score |
| `final_cart.json` | Cart as finalized by the tool server |
| `tool_log.json` | Ordered tool calls and results |
| `eval_metrics.json` | Response, interruption, and latency metrics |
| `turn_taking.json` | Deterministic turn-taking analysis |
| `prompt.json` | Exact model, provider, prompt, tools, and duration used |
| `adapter_A.log` | Customer simulator events and clip routing |
| `adapter_B.log` | Model events, transcripts, and tool calls |
| `A.wav`, `B.wav` | Per-role recordings |
| `combined.wav` | Stereo mix, Role A left and Role B right |

Inspect the headline result:

```bash
jq '{
  score: ."Task-specific score",
  submitted,
  totals: .diff.totals,
  tags
}' artifacts/outputs/OrderUp/<task_id>/<run_timestamp>/score.json
```

A clean exact-match run should have:

- `score` equal to `1.0`.
- `submitted` equal to `true`.
- `matched_strict` equal to `expected_qty`.
- `missing`, `extra`, `wrong_item`, and `wrong_option_or_variant` equal to
  zero.
- No failure tags.

Then compare the actual calls and cart:

```bash
jq '.' artifacts/outputs/OrderUp/<task_id>/<run_timestamp>/tool_log.json
jq '.' artifacts/outputs/OrderUp/<task_id>/<run_timestamp>/final_cart.json
jq '.expected_cart' artifacts/order_up_tasks/<task_id>.complex.task.json
```

Do not count an infrastructure failure as a model failure. A run with zero
clips, no submitted cart, or failed adapter processes is invalid even if the
scorer writes a numeric zero.

## Troubleshooting

### `n_clips=0`, both adapters apply SDP, and the run times out

Inspect both adapter logs for ICE state. If ICE remains in `checking` and then
fails, the environment is blocking WebRTC media traffic. HTTP access to
`api.openai.com` is not sufficient: WebRTC also connects to negotiated media
IP addresses over UDP or TCP.

Run from a normal local shell with outbound WebRTC traffic enabled. Sandboxes,
restricted CI runners, VPNs, and corporate firewalls can block the negotiated
media connection.

### `401 Unauthorized` or `Repository Not Found`

The dataset is still private and the active Hugging Face identity lacks access,
or no identity is configured. Request dataset access and then log in again:

```bash
hf auth login
hf auth whoami
```

### `Missing task file` or `Missing decision tree`

The task was not hydrated. Confirm the task ID exists in the Hugging Face
dataset, check authentication, and rerun without `--offline`.

### The default dataset command says there are no tasks

On a fresh cache, use an explicit jobs file as described in “Generate a jobs
file from Hugging Face.” The current no-argument path filters out tasks whose
artifacts are not already present.

### npm reports root-owned cache files

Use `npm install --cache /tmp/order-up-npm-cache`. Do not use `sudo npm` in
the repository and do not recursively change ownership without understanding
what else is stored in the cache.

### Post-run audit reports `No module named openai`

```bash
uv pip install --python .venv/bin/python openai
```

### Timeline calibration reports missing SciPy

```bash
uv pip install --python .venv/bin/python scipy
```

## Known validation caveats

These issues were observed while validating the current repository and should
be resolved or documented before treating the release as fully reproducible:

1. A successful compound-utterance task can report
   `diff.early_submitted=true` because clip-coverage accounting attributes the
   compound clip to one stage item even when it contains and successfully
   produces multiple expected items. The exact cart score is unaffected when
   every item matches, but the field is misleading.
2. The per-run outputs behind the published results are not included in this
   repository. The results page reports the mean of 3 sweeps per model over
   the 248 released tasks; until the run manifest and score files are
   published, those numbers cannot be reconstructed from this repository
   alone.
3. Model aliases can change behavior over time. Record `prompt.json`, retain the
   model identifier, and prefer pinned model snapshots when reproducibility is
   required.
4. Realtime model output is nondeterministic. Validate infrastructure with a
   small smoke set before launching an expensive full sweep, and retain every
   run rather than overwriting prior results.

## Before an open-source release

At minimum:

- Make the Hugging Face dataset public and add its approved data license.
- Add tracked dependency manifests and a one-command bootstrap path.
- Add a preflight command that validates dependencies, provider credentials,
  artifact availability, and WebRTC connectivity.
- Prevent zero-clip infrastructure failures from being recorded as ordinary
  model scores.
- Fix compound-clip coverage accounting.
- Publish the exact run manifests and score inputs used to produce headline
  tables.
- Confirm the audio, transcripts, menus, and metadata are licensed and scrubbed
  appropriately for public distribution.
