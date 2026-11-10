# Contributing to Order-Up-Bench

Thanks for your interest in contributing! This document explains how to get a
change accepted into Order-Up-Bench.

## Contributor License Agreement (CLA)

Before we can merge any contribution, you must accept the Order-Up-Bench
[Contributor License Agreement](CLA.md). The CLA grants DoorDash the rights it
needs to redistribute your contribution under the project's Apache 2.0 license
and protects both you and downstream users.

You only need to sign the CLA once. The same signature covers all of your
future contributions to this repository.

### How to sign

We use [cla-bot](https://github.com/apps/cla-bot) to check signatures on every
pull request. When you open your first PR, the bot will comment if your GitHub
username is not yet on file.

To sign, open a **separate** pull request that adds your GitHub username to the
`contributors` array in [`.clabot`](.clabot), with this exact line in the PR
description:

> I have read and agree to the Order-Up-Bench Contributor License Agreement
> (CLA.md, Version 1.0).

If you are contributing on behalf of an employer, confirm that you have
authority to do so under section 4 of the CLA. Once that PR is merged, cla-bot
will recheck your contribution PRs automatically (you can also comment
`@cla-bot check` to trigger a recheck).

If your contribution includes material that is not your original creation,
follow section 7 of the CLA: submit it separately, identify the source and
license, and mark it conspicuously as "Not a Contribution."

## Reporting Issues

Use GitHub Issues for bug reports and feature requests. Include:

- The commit of this repository you are running (`git rev-parse HEAD`) and the
  dataset revision (`HF_REVISION`, if you overrode the default)
- Your OS and architecture, and your Node.js and Python versions
- The Role B provider and model (`--role-b` value and model ID) and the task ID
- A minimal reproduction (commands run, expected vs. actual behavior), with the
  relevant excerpt of `adapter_A.log` / `adapter_B.log` where applicable

Remove API keys and tokens from anything you paste.

## Development Setup

See [RUNNING_BENCHMARKS.md](RUNNING_BENCHMARKS.md) for prerequisites and the
full setup and execution guide.

```bash
git clone https://github.com/doordash-oss/audio-benchmarks.git
cd audio-benchmarks/eval
npm install
uv venv --python python3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -r clip_eval/role_b/pipecat/requirements.txt
```

## Verification

Run these from `eval/` before every pull request. None of them call a paid
provider API.

```bash
uv pip install --python .venv/bin/python pytest
PYTHONPATH="$PWD" .venv/bin/python -m pytest clip_eval/tests
node clip_eval/role_b/openai_native/selftest.js
bash -n clip_eval/runtime/run_order_up_eval.sh
bash -n clip_eval/runtime/run_order_up_dataset.sh
```

The decision-tree tests under `clip_eval/role_a/tests/` read real task files.
Run them after hydrating a task into `eval/artifacts/` (or point
`ARTIFACTS_PATH` at an existing artifact tree):

```bash
node clip_eval/role_a/tests/test_ack_sentinels.js
node clip_eval/role_a/tests/test_only_one.js
node clip_eval/role_a/tests/test_subclip_fallback.js
```

If your change touches the runner, an adapter, Role A routing, or scoring, also
run at least one task end to end as described in
[RUNNING_BENCHMARKS.md](RUNNING_BENCHMARKS.md) and confirm that `score.json`,
`tool_log.json`, and the recordings are produced. Live runs use provider
credits.

If your change touches the report (`index.md`, `_includes/`, `_data/`,
`assets/`), rebuild the site with `./bin/build` and commit the regenerated
`docs/` output.

## Submitting a Pull Request

1. **Branch naming** — use `feature/<short-name>`, `fix/<short-name>`, or
   `technical/<short-name>`.
2. **Commits** — one logical change per commit; subjects in sentence case,
   imperative mood, under 60 characters.
3. **Tests** — add or update tests as appropriate. New behavior without test
   coverage will generally be asked to add some.
4. **Verification note** — in the PR description, list the checks you ran and
   name any relevant check you intentionally skipped, with a one-sentence
   reason.
5. **Benchmark comparability** — say explicitly if the change can alter scores,
   metrics, Role A behavior, prompts, or task selection. Such changes make new
   results incomparable with published ones and need maintainer agreement
   before merge.
6. **Open the PR** — fill in the description, link any related issue, and
   (until CLA automation is in place) include the CLA acceptance line described
   above.
7. **Review** — a maintainer will review and may request changes. Once approved
   and verification is accepted, a maintainer will merge.

Never commit credentials, `.env` files, audio recordings, run outputs, or
anything under `eval/artifacts/`. Benchmark audio and task data are distributed
through the Hugging Face dataset, not this repository.

## Third-Party Dependencies

If your change adds, removes, or upgrades a dependency, update the relevant
manifest — `eval/package.json` and `eval/package-lock.json`,
`eval/requirements.txt`, `eval/clip_eval/role_b/pipecat/requirements.txt`, or
`Gemfile` and `Gemfile.lock` — and update [NOTICE.txt](NOTICE.txt) if the
dependency is bundled into the repository. New dependency licenses must be
reviewed and approved by DoorDash Legal before the PR can be merged.

## Code of Conduct

Be respectful, assume good faith, and keep discussions focused on the work.
Maintainers may close or moderate threads that drift from these principles.
