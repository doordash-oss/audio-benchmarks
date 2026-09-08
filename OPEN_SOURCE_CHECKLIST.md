# Order-Up-Bench Open-Source Release Checklist

This document tracks the work required to publish Order-Up-Bench as a usable,
reproducible open-source benchmark. The release includes four related
artifacts:

1. The benchmark runtime and scoring code.
2. The audio dataset and generated task artifacts.
3. Reproducible model-run results and aggregate metrics.
4. The public benchmark report website.

The release is not complete until a new user can clone the code, download a
versioned dataset from Hugging Face, run a smoke task, reproduce published
metrics from released result files, and understand the licenses that govern
each artifact.

## Release status

| Area | Current state | Release gate |
| --- | --- | --- |
| Artifact distribution | Runtime hydrates a private, pinned Hugging Face dataset | Make dataset public and approve its license |
| Code licensing | Apache 2.0 added for original repository software | Approved third-party notices and compatibility review |
| Dataset licensing | No dataset card or public data license | Approved audio/data license and documented consent |
| Dependency setup | Local, gitignored Node/Python environments | Tracked, locked dependency manifests |
| Smoke execution | OpenAI task runs successfully locally | Automated preflight and smoke test |
| Provider sweep | Harness supports multiple providers | Reviewed, pinned, reproducible release sweep |
| Release results | Historical internal runs are incomplete and not required for release | Canonical final-sweep manifest and result bundle |
| Public report | Locally built Jekyll site under `docs/` | Published, reviewed, linked to code/data versions |

## Recommended release order

Complete these phases in order. Licensing and data-rights review can block the
entire release and should begin in parallel with the technical work.

1. Establish provenance, rights, and release scope.
2. Finalize artifact distribution on Hugging Face.
3. Package the code and add preflight/smoke automation.
4. Fix known scoring and infrastructure-validity issues.
5. Run and review the final provider sweep.
6. Recompute and independently verify every published result.
7. Publish the repository, Hugging Face dataset, result bundle, and report.
8. Tag immutable versions and complete a clean-room release test.

## 1. Define the release scope and ownership

- [ ] Name a release owner for the code repository.
- [ ] Name a dataset owner responsible for audio rights, removals, and Hugging
  Face maintenance.
- [ ] Name a results owner responsible for run manifests and published tables.
- [x] Decide the language scope of the initial release. English only; the
  unreviewed Spanish manifest is excluded and has been removed.
- [ ] Decide which Role B providers are part of the official headline sweep.
  The current harness supports OpenAI, Gemini, Grok, Ultravox, and a cascaded
  OpenAI baseline.
- [ ] Decide which artifacts are public: raw sessions, stage clips, background
  noise, decision trees, task configurations, prompts, transcripts, run audio,
  and model outputs.
- [ ] Document anything intentionally excluded and why.
- [ ] Assign a public contact for questions, vulnerability reports, and dataset
  takedown requests.

## 2. Confirm provenance, consent, privacy, and legal rights

This is a blocking release gate. Do not publish audio or apply a permissive
license until the organization confirms it has the required rights.

- [x] Remove the known Full-Duplex-Bench-derived sources formerly stored under
  `eval/clip_eval/_vendor/`; the runtime now uses DoorDash-owned Apache-2.0
  components plus the separately declared Pipecat dependency.
- [ ] Finish provenance review for benchmark data, documentation, and any code
  outside the removed vendor directory; replacing those known files does not
  establish provenance for unrelated artifacts.
- [ ] Preserve required upstream copyright and attribution notices for every
  remaining bundled third-party component.
- [ ] Confirm that every recorded speaker consented to public redistribution,
  downstream research use, and the selected dataset license.
- [ ] Confirm whether commercial use is permitted or must remain restricted.
- [x] Review recordings and transcripts for names, addresses, phone numbers,
  payment information, device identifiers, background conversations, and other
  personal or sensitive information.
- [x] Run deterministic PII scans over transcripts and metadata, followed by a
  human review of flagged samples.
- [x] Human-audit a statistically meaningful sample of the audio itself;
  transcript-only review cannot detect every disclosure.
- [ ] Remove internal paths, employee identifiers, session-owner metadata,
  private bucket names, internal project IDs, and debugging payloads that are
  not necessary for the benchmark.
- [ ] Confirm that restaurant/menu content is synthetic, owned, licensed, or
  otherwise safe to redistribute.
- [ ] Document the dataset removal/takedown process and a durable contact.
- [ ] Obtain formal Legal and Privacy approval for the final artifact inventory
  and licenses.

PII review, transcript/metadata scanning, and human audio review have been
completed. Retain the review methodology, date, owner, and approval evidence in
the release record even if those internal review materials are not published.

## 3. Add licenses and release notices

Code, data, report prose, and generated results may require different licenses.
Make the boundary explicit rather than using one ambiguous root license.

### Git repository

- [x] Select Apache License 2.0 for original repository software.
- [x] Add a root `LICENSE` containing the exact Apache License 2.0 text.
- [ ] Complete the top-level `NOTICE.txt` with all bundled components, their
  sources, licenses, and required attribution.
- [ ] Re-run the dependency inventory, update `NOTICE.txt`, and obtain approval
  whenever a release adds or changes bundled third-party material or generated
  site assets.
- [ ] Add `DATA_LICENSE.md` explaining that dataset assets are governed by the
  Hugging Face dataset license rather than necessarily by the code license.
- [ ] State the license for report prose, figures, and tables.
- [x] Add the DoorDash copyright and Apache 2.0 license notice to all
  DoorDash-created source files that support comments.
- [ ] Confirm that every dependency and vendored asset is compatible with the
  intended distribution and use.

### Hugging Face dataset

- [ ] Select the approved dataset license and configure the dataset card's
  license metadata.
- [ ] Add a complete dataset card (`README.md`) with creators, provenance,
  supported tasks, languages, splits, schema, collection process, consent,
  privacy treatment, limitations, intended uses, prohibited/out-of-scope uses,
  citation, and contact information.
- [ ] If redistribution must be gated, configure the Hugging Face repository's
  access terms and document token-based download setup.
- [ ] Include attribution and third-party notices alongside the dataset.
- [ ] Clearly mark files or subsets governed by different terms.

### Repository governance

- [ ] Add `CONTRIBUTING.md`.
- [ ] Add `CODE_OF_CONDUCT.md`.
- [ ] Add `SECURITY.md` with a private vulnerability-reporting path.
- [ ] Add `CITATION.cff` and the benchmark's preferred BibTeX citation.
- [ ] Add issue and pull-request templates.

## 4. Finalize Hugging Face artifact distribution

The current implementation is centralized in
`eval/clip_eval/pipeline/artifact_store.py`. Preserve its local artifact layout
so the runtime and scoring code need minimal changes.

### Design the Hugging Face repository

- [x] Create a Hugging Face **dataset** repository under the final organization.
- [x] Define a row schema containing task config fields, decision trees, and
  encoded clips; the downloader materializes the legacy local artifact layout.
- [ ] Decide which large files use standard Hub storage versus Xet/LFS-backed
  storage.
- [x] Pin the current Hub commit SHA in the downloader.
- [ ] Create an explicit dataset version such as `v1.0.0` after final review.
- [ ] Generate a manifest containing every published file's path, byte size,
  checksum, media type, and logical task/session owner.
- [ ] Validate that the uploaded dataset contains every task referenced by the
  released benchmark manifests.

### Implement the downloader

- [x] Make Hugging Face `datasets` streaming hydration the only remote runtime
  path.
- [x] Add configuration for `HF_REPO_ID`, `HF_REVISION`, and optional
  `HF_TOKEN`.
- [x] Pin the reviewed revision by default rather than silently following a
  mutable `main` branch.
- [x] Preserve `ARTIFACTS_PATH` as the local cache-root override.
- [ ] Offline mode performs no network calls; add a preflight that reports the
  complete list of missing local files rather than failing at the first one.
- [x] Remove automatic remote writes/uploads from normal benchmark execution.
  Dataset publication should be a separate maintainer-only command.
- [x] Remove the legacy cloud-storage client, bucket/project configuration,
  credential requirements, and remote-write paths.
- [ ] Add retries, resumable downloads, checksum validation, and actionable
  errors for missing/gated files.
- [x] Write hydrated artifacts atomically so concurrent workers do not expose
  partial files.

### Prove migration parity

- [ ] Compare the source export and Hugging Face manifests file-for-file and
  checksum-for-checksum before freezing the release dataset.
- [ ] Hydrate representative simple, compound, clarification, correction,
  background-noise, and English tasks from an empty cache.
- [x] Compare a source-exported task with its Hugging Face materialization;
  expected task config, decision tree, clip bytes, and score inputs match.
- [ ] Test public anonymous download and, if applicable, gated authenticated
  download.
- [ ] Test interrupted/resumed downloads and concurrent dataset workers.

## 5. Package dependencies and add one-command setup

The repository currently relies on local gitignored dependency setup. That is
not sufficient for a public benchmark.

- [ ] Add a tracked `eval/package.json` with only required runtime dependencies.
- [ ] Add and commit the corresponding lockfile.
- [ ] Add a tracked `pyproject.toml` with Python 3.11 compatibility and explicit
  core/provider dependency groups.
- [ ] Add and commit `uv.lock`.
- [ ] Separate core OpenAI runtime dependencies from optional Gemini, Grok,
  Ultravox, and cascaded extras.
- [ ] Add `.python-version` and `.nvmrc` or equivalent documented version pins.
- [ ] Add `.env.example` containing variable names and comments, never values.
- [ ] Add a bootstrap command such as `./bin/setup-eval` or a small set of
  documented `make`/`just` targets.
- [ ] Add a preflight command that validates Node, Python, ffmpeg, dependency
  imports, provider credentials, Hugging Face access, task completeness, local
  ports, and WebRTC connectivity.
- [ ] Make preflight failures nonzero and human-readable.
- [x] Update `RUNNING_BENCHMARKS.md` with Hugging Face authentication,
  hydration, configuration, and offline instructions.
- [ ] Test setup on a clean macOS arm64 machine and at least one supported Linux
  environment.

## 6. Harden runtime validity and scoring

Infrastructure failures must not be reported as model-quality failures.

- [ ] Add a run validity state separate from the task score: `valid`,
  `infrastructure_failed`, `timed_out`, `provider_failed`, or `invalid_output`.
- [ ] Reject or quarantine runs where zero customer clips were played.
- [ ] Reject or quarantine runs where either adapter failed ICE/WebRTC setup.
- [ ] Reject or quarantine runs with missing recordings, final cart, prompt
  snapshot, or tool log.
- [ ] Record natural completion separately from duration-watchdog termination.
- [ ] Fix compound-clip coverage accounting. A compound utterance containing
  multiple expected items must not incorrectly produce
  `diff.early_submitted=true` when all covered items were heard and matched.
- [ ] Make the fresh-cache dataset path hydrate tasks before filtering them, or
  remove the local-artifact prefilter.
- [ ] Prevent dataset `STATUS.txt` from labeling an invalid zero-clip run as an
  ordinary successful process exit.
- [ ] Add schema validation for task configs, decision trees, score files,
  prompt snapshots, tool logs, and aggregate results.
- [ ] Add stable machine-readable exit codes for success, model failure, and
  infrastructure failure.

## 7. Add automated tests and CI

- [ ] Unit-test strict cart matching, default options, variants, corrections,
  missing/extra items, and submission state.
- [ ] Add regression coverage for compound clips and the false
  `early_submitted` condition.
- [ ] Unit-test decision-tree routing and terminal behavior.
- [ ] Test artifact resolution with a tiny public fixture dataset.
- [ ] Add an offline end-to-end test using deterministic prerecorded/model
  fixtures so CI does not require paid provider calls.
- [ ] Add an opt-in live provider smoke workflow that uses protected secrets and
  is never required for external pull requests.
- [ ] Add linting/formatting for Python, JavaScript, shell, Markdown, and JSON.
- [ ] Add dependency, secret, and license scanning.
- [ ] Verify the Jekyll report build in CI.
- [ ] Ensure CI never uploads artifacts or exposes API keys in logs.

## 8. Run the final provider sweep

Do not launch the expensive full sweep until setup, validity gates, and scoring
regressions pass.

### Freeze the experiment

- [ ] Freeze the exact task manifest and record its checksum.
- [ ] Freeze the Hugging Face dataset tag and commit SHA.
- [ ] Freeze Role A's model, prompt, voice, VAD, and routing profile logic.
- [ ] Pin Role B provider model snapshots where available.
- [ ] Record provider model IDs, adapter commits, dependency lock hashes,
  runtime flags, concurrency, timeout policy, region, and run date.
- [ ] Decide the retry policy before running. Never selectively retry only poor
  model outcomes.
- [ ] Decide how invalid/infrastructure runs affect denominators before running.
- [ ] Estimate provider spend and set explicit budget/rate-limit safeguards.

### Execute and monitor

- [ ] Run a small cross-provider smoke set covering varied task shapes.
- [ ] Manually listen to and inspect every smoke run.
- [ ] Run OpenAI Realtime across the frozen manifest.
- [ ] Run Gemini Live across the same manifest.
- [ ] Run Grok Voice across the same manifest.
- [ ] Run the cascaded baseline across the same manifest.
- [ ] Run Ultravox only if it is part of the declared release comparison.
- [ ] Monitor invalid-run rates, zero-clip runs, provider errors, timeouts,
  interruption anomalies, and unexpected cost.
- [ ] Preserve every attempt and retry from the **final release sweep** with an
  immutable run ID until the canonical inclusion manifest is frozen.
- [ ] Never overwrite outputs from the final release sweep.

### Review failures

- [ ] Review all infrastructure-invalid runs and rerun according to the frozen
  retry policy.
- [ ] Sample successful and failed conversations for human verification.
- [ ] Verify transcripts and tool logs against the recorded audio for the
  review sample.
- [ ] Check for systemic provider/adaptor differences that make comparisons
  unfair.
- [ ] Confirm that Role A behavior remains fixed across Role B providers.
- [ ] Confirm that every included run used the intended prompt/model/provider;
  use `prompt.json`, not directory names, as the source of truth.

## 9. Recompute and verify published results

- [ ] Create a canonical release run manifest with one included run ID per
  task/provider and explicit exclusions/retries.
- [ ] Aggregate from that manifest rather than globbing every historical output
  directory.
- [ ] Recompute success, recoverability, turn-taking, response, interruption,
  no-response, and latency metrics from raw released files.
- [ ] Independently recompute exact cart success with a second script or review
  implementation.
- [ ] Confirm numerator, denominator, rounding, task intersection, and missing-
  run treatment for every report table and chart.
- [ ] Compare every generated report number against the displayed website text.
- [ ] Fail the report build when generated values and checked-in prose diverge.
- [ ] Publish the aggregate script, release manifest, aggregate JSON/CSV, and
  enough per-run evidence for an external user to reproduce every table.
- [ ] Document any result files that cannot be redistributed and provide a
  reproducible derivation path.
- [ ] Record known limitations and unresolved discrepancies prominently.

Historical development runs do not need to be published. They may include
debugging attempts, invalid infrastructure runs, obsolete schemas, superseded
prompts, or exploratory providers and can create more confusion than
reproducibility. The release must instead publish the canonical final-sweep run
manifest, the included score/metric files, aggregate outputs, exact
configuration snapshots, and the documented retry/exclusion policy. No
historical run may contribute to a published number unless it is explicitly
included in that manifest and released.

The current private internal output archive—438 score files across 145 distinct
tasks—is therefore not itself a release blocker, but it is also not a valid
source for reconstructing the reported 254-task, four-provider comparison.

## 10. Final repository sweep and review

### Security and repository hygiene

- [ ] Scan the entire Git history, not only the current tree, for API keys,
  Google credentials, tokens, internal URLs, private bucket names, employee
  identifiers, and absolute workstation paths.
- [ ] Remove or rotate every exposed secret before making the repository
  public.
- [ ] Remove internal-only scripts, comments, prompts, and debugging payloads
  that should not ship.
- [ ] Remove generated outputs and large datasets from Git; host them on
  Hugging Face or in a dedicated release bundle.
- [ ] Confirm `.gitignore` covers `.env`, credentials, local artifacts, caches,
  recordings, and provider logs.
- [ ] Check every tracked audio clip and generated site asset against the
  approved public artifact inventory.
- [ ] Replace absolute paths and machine-specific assumptions.
- [ ] Review dependencies for known vulnerabilities and unsupported packages.

### Documentation and usability

- [ ] Rewrite the root README with the benchmark purpose, architecture, quick
  start, data link, report link, supported providers, headline results,
  limitations, license boundary, and citation.
- [ ] Verify every command in `RUNNING_BENCHMARKS.md` from a clean clone.
- [ ] Document expected costs and which providers require paid accounts.
- [ ] Document supported operating systems and known WebRTC/firewall issues.
- [ ] Add architecture and data-flow diagrams only where they clarify the
  runtime.
- [ ] Add a changelog and versioning policy for code, dataset, and results.
- [ ] Have an engineer unfamiliar with the project complete the setup without
  private guidance; record and fix every stumbling block.

### Engineering review

- [ ] Conduct a focused code review of artifact downloading, process cleanup,
  concurrency, scoring, aggregation, and secret handling.
- [ ] Review prompts and decision-tree behavior for benchmark leakage or unfair
  provider-specific assumptions.
- [ ] Verify that optional audits cannot silently change headline scores.
- [ ] Verify that all generated outputs capture the exact configuration used.
- [ ] Resolve or explicitly document every release-blocking TODO.

## 11. Publish the report

The repository already contains a Jekyll report and a local build command that
writes the static site into `docs/`.

- [ ] Choose the public canonical report URL and owner.
- [ ] Decide whether to continue using GitHub Pages or another approved static
  hosting path.
- [ ] Build the report from release aggregates rather than manually maintained
  numbers wherever possible.
- [ ] Add prominent links to the Git repository, Hugging Face dataset, released
  result bundle, licenses, citation, and methodology.
- [ ] Display exact code tag, dataset revision, run-manifest version, and report
  build date.
- [ ] Include limitations, consent/privacy summary, provider costs, and
  nondeterminism caveats.
- [ ] Verify charts have accessible labels, text alternatives, and source data.
- [ ] Test desktop/mobile layout, audio playback, deep links, and browser
  compatibility.
- [ ] Check that no report asset exposes private audio, metadata, paths, or
  credentials.
- [ ] Add analytics only if approved and disclose them appropriately.
- [ ] Configure redirects and a custom domain if required.
- [ ] Run link, accessibility, and static-site security checks.
- [ ] Obtain Communications/Brand review if required.

Build locally with:

```bash
./bin/build
```

## 12. Release and clean-room verification

- [ ] Create immutable and matching versions for code, dataset, run manifest,
  results, and report.
- [ ] Create a signed Git tag and GitHub release notes.
- [ ] Publish the Hugging Face dataset at the pinned revision.
- [ ] Publish aggregate and per-run result artifacts with checksums.
- [ ] Publish the report and verify all cross-links.
- [ ] From a machine/account without DoorDash internal access, clone the public
  repository and follow only the public documentation.
- [ ] Download the dataset anonymously or through the documented gated flow.
- [ ] Run preflight, one smoke task, scoring, and aggregate verification.
- [ ] Confirm no command reaches internal hosts or private repositories.
- [ ] Confirm an external user can identify the exact inputs behind every
  headline number.
- [ ] Archive final approval evidence and announce the release.

## Definition of done

Order-Up-Bench is ready to open-source when all of the following are true:

- [ ] Code, data, report, and third-party license boundaries are approved and
  unambiguous.
- [ ] The public dataset is complete, versioned, checksummed, documented, and
  downloadable from Hugging Face without internal infrastructure.
- [ ] A fresh clone installs from tracked lockfiles and passes preflight.
- [ ] A new user can run and score a task without DoorDash access.
- [ ] Infrastructure-invalid runs cannot contaminate model metrics.
- [ ] The final provider sweep is frozen, complete, reviewable, and represented
  by a canonical run manifest.
- [ ] Every published number is generated from released artifacts and
  independently reconciled.
- [ ] The report is public, accessible, versioned, and linked to the exact code,
  dataset, and results releases.
- [ ] A clean-room external test passes using only public documentation.
