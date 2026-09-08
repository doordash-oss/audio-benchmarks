# Copyright 2026 DoorDash, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Artifact store for the clip_eval pipeline.

Every pipeline artifact has a canonical local path under ``ARTIFACTS_PATH``.
Public benchmark inputs are hydrated from a row in the Hugging Face dataset.

This module unifies access:

  * ``read_*`` and ``write_*`` operate on the local artifact cache.
  * ``cache_pull_stage(tid)`` hydrates a runnable task from Hugging Face.

Configuration (all env vars are optional)::

    HF_REPO_ID             default: doordash/doordash-audio-benchmarks
    HF_REVISION            immutable dataset revision used by default
    HF_TOKEN               optional for public datasets; required while the
                           dataset is private (cached ``hf auth login`` works)
    HF_DATASETS_CACHE      optional datasets cache directory
    ARTIFACTS_PATH         local cache root (existing override)
    CLIP_EVAL_OFFLINE      "1" / "true" → no remote calls; reads must
                           hit the local cache
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from clip_eval.pipeline.paths import ARTIFACTS_PATH


# ── Configuration ───────────────────────────────────────────────────────

DEFAULT_HF_REPO_ID = "doordash/doordash-audio-benchmarks"
# Pin the reviewed snapshot so benchmark inputs do not silently drift.
DEFAULT_HF_REVISION = "390df8065efcb2ab74d798d235cf5c42a5ad08b3"


def hf_repo_id() -> str:
    return os.environ.get("HF_REPO_ID") or DEFAULT_HF_REPO_ID


def hf_revision() -> str:
    return os.environ.get("HF_REVISION") or DEFAULT_HF_REVISION


def hf_datasets_cache() -> Path:
    configured = os.environ.get("HF_DATASETS_CACHE")
    return Path(configured) if configured else ARTIFACTS_PATH / ".hf_cache" / "datasets"


def is_offline() -> bool:
    return (os.environ.get("CLIP_EVAL_OFFLINE") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# ── OverwritePolicy ─────────────────────────────────────────────────────


class OverwritePolicy(str, Enum):
    """Controls how :func:`write_bytes` / :func:`write_json` handles
    an existing artifact.

    OVERWRITE        — unconditional local replacement.
    SKIP_IF_EXISTS   — no-op if a local copy exists. Used for
                       once-only backups like ``transcription.json.original``.
    SKIP_IF_LOCAL    — no-op if a local copy exists. Used for cache-style
                       outputs like
                       ``transcription.json`` post-first-transcribe and
                       per-file ``background_noise/noise_*.wav``.
    MERGE_JSON       — read-modify-write: load existing local JSON,
                       dict/list-merge the new payload on top, and write.
    """

    OVERWRITE = "overwrite"
    SKIP_IF_EXISTS = "skip_if_exists"
    SKIP_IF_LOCAL = "skip_if_local"
    MERGE_JSON = "merge_json"


# ── ArtifactRef ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to one artifact.

    ``path`` is relative to ``ARTIFACTS_PATH``. Helpers below construct refs
    so callers rarely build one by hand.
    """

    path: Path

    def local_path(self) -> Path:
        return ARTIFACTS_PATH / self.path


def _ref(*parts) -> ArtifactRef:
    return ArtifactRef(Path(*parts))


# ── Existence / fetch / write ───────────────────────────────────────────


def _exists_local(ref: ArtifactRef) -> bool:
    p = ref.local_path()
    return p.exists() and (p.is_dir() or p.stat().st_size > 0)


def exists(ref: ArtifactRef) -> bool:
    return _exists_local(ref)


def cache_pull(ref: ArtifactRef) -> Path:
    """Return an existing local artifact or explain how to hydrate its task."""
    local = ref.local_path()
    if _exists_local(ref):
        return local
    if is_offline():
        raise FileNotFoundError(
            f"artifact missing locally with CLIP_EVAL_OFFLINE=1: {local}"
        )
    raise FileNotFoundError(
        f"artifact missing locally: {local}. Hugging Face stores complete "
        "tasks; call cache_pull_stage(task_id) to hydrate one."
    )


def read_bytes(ref: ArtifactRef, *, prefer_local: bool = True) -> bytes:
    if prefer_local and _exists_local(ref):
        return ref.local_path().read_bytes()
    return cache_pull(ref).read_bytes()


def read_json(ref: ArtifactRef, *, prefer_local: bool = True) -> dict:
    return json.loads(read_bytes(ref, prefer_local=prefer_local).decode("utf-8"))


def write_bytes(
    ref: ArtifactRef,
    data: bytes,
    *,
    policy: OverwritePolicy = OverwritePolicy.OVERWRITE,
) -> None:
    """Write ``data`` to the local artifact cache according to ``policy``."""
    if policy == OverwritePolicy.SKIP_IF_EXISTS:
        if _exists_local(ref):
            return
    elif policy == OverwritePolicy.SKIP_IF_LOCAL:
        if _exists_local(ref):
            return
    elif policy == OverwritePolicy.MERGE_JSON:
        raise ValueError("MERGE_JSON only valid for write_json")
    _atomic_write(ref.local_path(), data)


def write_json(
    ref: ArtifactRef,
    obj,
    *,
    policy: OverwritePolicy = OverwritePolicy.OVERWRITE,
    indent: Optional[int] = 2,
) -> None:
    if policy == OverwritePolicy.MERGE_JSON:
        # Read existing payload (prefer local) then deep-merge dicts /
        # extend lists. Intentionally simple — suits repair_state.json's
        # "append to disabled_items / disabled_options" pattern. Non-dict
        # / non-list values are replaced by the incoming payload.
        existing = {}
        try:
            existing = read_json(ref)
        except FileNotFoundError:
            existing = {}
        merged = _deep_merge(existing, obj)
        data = json.dumps(merged, indent=indent).encode("utf-8")
        _atomic_write(ref.local_path(), data)
        return
    data = json.dumps(obj, indent=indent).encode("utf-8")
    write_bytes(ref, data, policy=policy)


def _deep_merge(base, new):
    if isinstance(base, dict) and isinstance(new, dict):
        out = dict(base)
        for k, v in new.items():
            out[k] = _deep_merge(out.get(k), v) if k in out else v
        return out
    if isinstance(base, list) and isinstance(new, list):
        seen = {json.dumps(x, sort_keys=True, default=str) for x in base}
        out = list(base)
        for item in new:
            key = json.dumps(item, sort_keys=True, default=str)
            if key not in seen:
                out.append(item)
                seen.add(key)
        return out
    return new


def delete(ref: ArtifactRef) -> None:
    p = ref.local_path()
    if p.exists():
        if p.is_dir():
            import shutil
            shutil.rmtree(p)
        else:
            p.unlink()


# ── Bulk cache hydration ────────────────────────────────────────────────


def _json_field(row: dict, name: str):
    """Decode a dataset field that is stored as JSON text in Parquet."""
    value = row.get(name)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in Hugging Face field {name!r}") from exc
    return value


def _atomic_write(path: Path, data: bytes) -> None:
    """Write a complete artifact before making it visible to other workers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_write_json(path: Path, value) -> None:
    _atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8"))


# ── Loose files: task id map, background noise, shared clips ────────────
#
# The Parquet rows carry per-task inputs. Three things live outside them as
# plain files in the dataset repo:
#
#   * ``task_id_map.json`` — the published ``orderup-NNNN`` ids carry no
#     session identity, but the runtime needs it: ``run_order_up_eval.sh``
#     resolves the background-noise directory and the source game window
#     (auto-duration) from ``session_id`` / ``game_index``.
#   * ``background_noise/<session_id>/noise_*.wav`` — ambient loops the
#     clipPlayer cycles between clips. Every decision tree references these.
#   * ``clips/order_up_shared/*.wav`` — the shared clip pool the decision
#     tree falls back to when a task has no per-task copy.

_TASK_ID_MAP_FILE = "task_id_map.json"
_SHARED_CLIPS_PREFIX = "clips/order_up_shared/"

# Every shared-clip basename the customer side can inject. Mirrors
# SHARED_CLIP_NAMES in the internal build pipeline; kept here so hydration can
# assert completeness rather than trusting whatever the dataset happens to
# hold. Adding a clip to role_a/decision_tree.js without adding it here is
# caught by tests/test_shared_clip_pool.py.
SharedClipName = Literal[
    "yes.wav",
    "no.wav",
    "thats_all.wav",
    "just_the_defaults.wav",
    "only_one.wav",
    "the_first_option.wav",
    "the_second_option.wav",
]
SHARED_CLIP_NAMES: tuple[str, ...] = (
    "yes.wav",
    "no.wav",
    "thats_all.wav",
    "just_the_defaults.wav",
    "only_one.wav",
    "the_first_option.wav",
    "the_second_option.wav",
)

# Coarse conversational role. Mirrors ClipRole in the build pipeline. Typing
# only — this repo has no pydantic and validates nothing.
ClipRole = Literal["main", "opt", "restatement", "corr", "yes", "no"]
_CLIP_ROLE_VALUES = frozenset(("main", "opt", "restatement", "corr", "yes", "no"))

# Distinguishes "field absent from this revision's schema" from "field present
# and legitimately null". 102 published clips have a genuinely null clip_role.
_ABSENT = object()
_NOISE_PREFIX = "background_noise/"

_hf_cache_lock = threading.Lock()
_task_id_map_cache: Optional[dict] = None
_repo_files_cache: Optional[list] = None


def _hf_token():
    return os.environ.get("HF_TOKEN") or None


def _download_repo_file(filename: str) -> bytes:
    """Fetch one file from the pinned dataset revision."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face hydration requires the 'huggingface_hub' package"
        ) from exc
    path = hf_hub_download(
        repo_id=hf_repo_id(),
        filename=filename,
        repo_type="dataset",
        revision=hf_revision(),
        token=_hf_token(),
        cache_dir=str(hf_datasets_cache()),
    )
    return Path(path).read_bytes()


def _repo_files() -> list:
    """File listing for the pinned revision, fetched once per process."""
    global _repo_files_cache
    with _hf_cache_lock:
        if _repo_files_cache is None:
            try:
                from huggingface_hub import HfApi
            except ImportError as exc:
                raise RuntimeError(
                    "Hugging Face hydration requires the 'huggingface_hub' package"
                ) from exc
            _repo_files_cache = HfApi(token=_hf_token()).list_repo_files(
                hf_repo_id(), repo_type="dataset", revision=hf_revision()
            )
        return _repo_files_cache


def task_id_map() -> dict:
    """``{task_id: {legacy_task_id, session_id, game_index}}``.

    Read from the local cache when present so offline runs work; otherwise
    pulled from the pinned dataset revision.
    """
    global _task_id_map_cache
    with _hf_cache_lock:
        if _task_id_map_cache is not None:
            return _task_id_map_cache
    local = ARTIFACTS_PATH / _TASK_ID_MAP_FILE
    if local.exists() and local.stat().st_size > 0:
        data = json.loads(local.read_text())
    else:
        if is_offline():
            raise FileNotFoundError(
                f"{_TASK_ID_MAP_FILE} missing locally with CLIP_EVAL_OFFLINE=1: {local}"
            )
        raw = _download_repo_file(_TASK_ID_MAP_FILE)
        _atomic_write(local, raw)
        data = json.loads(raw.decode("utf-8"))
    with _hf_cache_lock:
        _task_id_map_cache = data
    return data


def task_source_session(tid: str) -> tuple[Optional[str], Optional[int]]:
    """``(session_id, game_index)`` for a task, or ``(None, None)``."""
    entry = task_id_map().get(tid) or {}
    return entry.get("session_id"), entry.get("game_index")


def task_source_window_seconds(tid: str) -> Optional[float]:
    """Length of the source conversation this task was cut from.

    The runner scales this to pick a run duration. It is published as a bare
    number rather than shipping the session's ``stage_alignment.json``, which
    carries full-session transcripts and internal paths.
    """
    entry = task_id_map().get(tid) or {}
    window = entry.get("source_window_seconds")
    return float(window) if window else None


# Loose-file hydration is latency-bound, not bandwidth-bound: the background
# noise is ~1580 files of ~250 KB each, so a serial loop spends nearly all its
# time waiting on round trips (~4-5 files/sec, ~6 minutes for one full pull).
# Downloads are independent, so a small thread pool collapses that to well
# under a minute. Kept modest to stay polite to the Hub and because several
# eval workers may hydrate concurrently.
_HYDRATE_WORKERS = 8


def _hydrate_repo_prefix(prefix: str, local_root: Path) -> int:
    """Copy every dataset file under ``prefix`` into ``local_root``.

    Returns the number of files now present locally. Already-present files
    are left alone so repeat runs and concurrent workers stay cheap.
    """
    names = [f for f in _repo_files() if f.startswith(prefix)]
    todo: list[tuple[str, Path]] = []
    present = 0
    for name in names:
        rel = name[len(prefix):]
        if not rel or rel.endswith("/"):
            continue
        dest = local_root / rel
        if dest.is_file() and dest.stat().st_size > 0:
            present += 1
            continue
        todo.append((name, dest))

    if not todo:
        return present
    if len(todo) == 1:
        name, dest = todo[0]
        _atomic_write(dest, _download_repo_file(name))
        return present + 1

    from concurrent.futures import ThreadPoolExecutor

    def fetch(item: tuple[str, Path]) -> None:
        name, dest = item
        _atomic_write(dest, _download_repo_file(name))

    with ThreadPoolExecutor(max_workers=_HYDRATE_WORKERS) as pool:
        # list() forces evaluation so an exception in any download surfaces
        # here rather than being swallowed by the generator.
        list(pool.map(fetch, todo))
    return present + len(todo)


def cache_pull_background_noise(session_id: str) -> int:
    """Hydrate ``background_noise/<session_id>/`` and return the file count.

    Raises when the session has no noise locally and none can be fetched.
    Silent absence is what made the missing-noise regression invisible: the
    clipPlayer falls back to digital silence with only a warning, which
    changes turn-taking behaviour without failing the run.
    """
    # Guard the same way task_id and clip_id are guarded below: session_id
    # comes from a dataset file and is interpolated into both a repo path and
    # a local filesystem path.
    if not isinstance(session_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", session_id):
        raise ValueError(f"unsafe session_id in {_TASK_ID_MAP_FILE}: {session_id!r}")
    local_root = session_background_noise_dir(session_id)
    existing = sorted(local_root.glob("noise_*.wav")) if local_root.is_dir() else []
    if is_offline():
        if not existing:
            raise FileNotFoundError(
                f"background noise missing locally with CLIP_EVAL_OFFLINE=1: {local_root}"
            )
        return len(existing)
    count = _hydrate_repo_prefix(f"{_NOISE_PREFIX}{session_id}/", local_root)
    if count == 0:
        raise FileNotFoundError(
            f"no background noise for session {session_id!r} in "
            f"{hf_repo_id()} at revision {hf_revision()}; the benchmark's "
            "decision trees require it"
        )
    return count


def cache_pull_shared_clips() -> int:
    """Hydrate ``clips/order_up_shared/`` and return the file count.

    Asserts the pool is *complete*, not merely non-empty. Each name in
    ``SHARED_CLIP_NAMES`` is a response the customer side can play — yes, no,
    declining an extra unit, picking the first or second option offered,
    accepting defaults, closing out. ``availableClipBasename`` in
    ``role_a/decision_tree.js`` returns null for a missing file and the choice
    is quietly dropped from ``valid_choices``, so the model is never asked the
    question it was meant to answer and nothing errors.

    That is not hypothetical: the previously published revision shipped 6 of
    the 7 (no ``only_one.wav``), and before that was missing the pool
    entirely, costing three more responses. See commit c457555 — "none of it
    failed loudly".
    """
    local_root = ARTIFACTS_PATH / "clips" / "order_up_shared"
    if not is_offline():
        _hydrate_repo_prefix(_SHARED_CLIPS_PREFIX, local_root)
    have = {p.name for p in local_root.glob("*.wav")} if local_root.is_dir() else set()
    missing = [n for n in SHARED_CLIP_NAMES if n not in have]
    if missing:
        raise FileNotFoundError(
            f"shared clip pool incomplete in {local_root}: missing {missing}. "
            f"Every name in SHARED_CLIP_NAMES is injectable by "
            f"role_a/decision_tree.js; a missing one silently removes a "
            f"customer response instead of failing. If HF_REVISION is set it "
            f"may predate the shared-pool republish."
        )
    return len(have)


def _derive_clip_role(clip_kind) -> Optional[str]:
    """Coarse role from the fine-grained kind, for pre-extension revisions.

    Branch order is load-bearing and mirrors the build pipeline:
    ``*_post_revision`` resolves to None BEFORE the ``_corr`` test (otherwise
    ``opt1_corr_post_revision`` becomes "corr"), and ``_corr`` resolves before
    the prefix tests (otherwise ``opt1_corr`` becomes "opt" and ``main_corr``
    becomes "main").

    Replaces the old inline expression, which emitted the raw kind — so
    ``opt1`` / ``restatement1`` / ``opt1_corr_post_revision`` were bucketed
    under values ClipRole cannot hold, and scoring saw a different role
    vocabulary on Hugging Face than on the internal artifacts.
    """
    if not isinstance(clip_kind, str) or not clip_kind:
        return None
    if clip_kind.endswith("_post_revision"):
        return None
    if "_corr" in clip_kind:
        return "corr"
    for prefix, role in (
        ("main", "main"), ("opt", "opt"), ("restatement", "restatement"),
        ("yes", "yes"), ("no", "no"),
    ):
        if clip_kind.startswith(prefix):
            return role
    return None


def _materialize_hf_row(row: dict, expected_tid: str) -> None:
    tid = row.get("task_id")
    if tid != expected_tid:
        raise ValueError(f"dataset returned task {tid!r}; expected {expected_tid!r}")
    if not isinstance(tid, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", tid):
        raise ValueError(f"unsafe task_id in Hugging Face dataset: {tid!r}")

    tree = _json_field(row, "decision_tree")
    if not isinstance(tree, dict):
        raise ValueError(f"decision_tree for {tid!r} is not a JSON object")
    tree["task_id"] = tid
    tree["clips_base_dir"] = str(stage_clips_dir(tid).resolve())

    clips = row.get("clips") or []
    if not isinstance(clips, list) or not clips:
        raise ValueError(f"task {tid!r} contains no audio clips")

    nodes = tree.get("nodes") or {}
    clips_manifest = []
    # Fields this revision could not supply, recorded on the manifest so a
    # scoring run can tell it is reading a degraded shim instead of guessing.
    lossy: set[str] = set()
    # {clip_source: [filenames written]} — audio the tree references that has
    # no manifest row of its own.
    audio_only: dict[str, list[str]] = {}
    for clip in clips:
        clip_id = clip.get("clip_id")
        if not isinstance(clip_id, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", clip_id):
            raise ValueError(f"unsafe clip_id in task {tid!r}: {clip_id!r}")
        audio = clip.get("audio") or {}
        encoded = audio.get("bytes") if isinstance(audio, dict) else None
        if encoded is None and isinstance(audio, dict) and audio.get("path"):
            source_path = Path(audio["path"])
            if source_path.is_file():
                encoded = source_path.read_bytes()
        if not isinstance(encoded, bytes) or not encoded:
            raise ValueError(f"clip {clip_id!r} in task {tid!r} has no encoded audio")

        audio_path = stage_clips_dir(tid) / f"{clip_id}.wav"
        # Always write the WAV: sidecar rows exist precisely to deliver audio
        # the tree references (`<clip_id>.fallback.wav` etc.), and writing to
        # `<clip_id>.wav` reproduces that filename exactly.
        _atomic_write(audio_path, encoded)

        # `clip_source` is the schema-version probe. Presence means the row
        # carries real metadata; absence means a revision published before the
        # struct was extended, and the fields must be reconstructed. Probing
        # `clip_role is None` instead would be wrong — 102 published clips have
        # a legitimately null role.
        source = clip.get("clip_source", _ABSENT)
        if source is _ABSENT:
            node = nodes.get(clip_id, {}) if isinstance(nodes, dict) else {}
            covered = node.get("covered_item_indices") or []
            lossy.update(("customer_spans", "audio_start", "audio_end",
                          "trigger_kind", "muted_intervals", "item_index"))
            clips_manifest.append({
                "clip_id": clip_id,
                "clip_kind": clip.get("clip_kind"),
                "clip_role": _derive_clip_role(clip.get("clip_kind")),
                "audio_path": str(audio_path.resolve()),
                "transcript": clip.get("transcript"),
                # Synthesized, not published, at this revision: item coverage
                # only, with no label/text/times.
                "customer_spans": [{"item_indices": covered}] if covered else [],
            })
            continue

        if source != "manifest":
            # Audio-delivery rows only. `customer_clips` has no row for any of
            # these, so emitting one would inject clips the internal manifest
            # does not have — polluting eval_metrics' per-clip lookup and
            # turn_taking_analysis' durations, and making an HF-hydrated run
            # score against a different clip set than a GCS-hydrated one:
            #   fallback/option/sub_clip — internally these hang off their
            #     parent as fallback_audio_path / option_spans[] / sub_clips[]
            #   tts_filler — yes/no/thats_all, copied into each task dir by
            #     the tree builder, never manifest rows
            #   orphan — tree-reachable audio whose manifest row was lost
            #     upstream; it must ship, but inventing a row would paper over
            #     the drift
            audio_only.setdefault(source, []).append(audio_path.name)
            continue

        role = clip.get("clip_role")
        if role is not None and role not in _CLIP_ROLE_VALUES:
            role = _derive_clip_role(clip.get("clip_kind"))
        duration = clip.get("audio_duration")
        clips_manifest.append({
            "clip_id": clip_id,
            "clip_kind": clip.get("clip_kind"),
            "clip_role": role,
            "clip_source": source,
            "trigger_kind": clip.get("trigger_kind"),
            "audio_path": str(audio_path.resolve()),
            "transcript": clip.get("transcript"),
            "item_index": clip.get("item_index"),
            "stage_index": clip.get("stage_index"),
            # Published times are clip-relative, so the clip's own timeline
            # starts at zero. Consumers computing `audio_end - audio_start`
            # (eval_metrics, turn_taking_*) get the true duration.
            "audio_start": 0.0,
            "audio_end": duration,
            "timing_unreliable": bool(clip.get("timing_unreliable")),
            "muted_intervals": clip.get("muted_intervals") or [],
            "post_revision_offset_t": clip.get("post_revision_offset_t"),
            "customer_spans": list(clip.get("customer_spans") or []),
        })

    tree_path = stage_decision_tree_ref(tid).local_path()
    # session_id / game_index are not columns in the dataset — the published
    # task ids dropped them. The runtime still needs both: the noise dir is
    # keyed by session, and auto-duration reads that session's game window.
    session_id, game_index = task_source_session(tid)
    task_config = {
        "task_id": tid,
        "split": "OrderUp",
        "session_id": session_id,
        "game_index": game_index,
        "source_window_seconds": task_source_window_seconds(tid),
        "restaurant_name": row.get("restaurant"),
        "language": row.get("language"),
        "decision_tree": str(tree_path.resolve()),
        "examinee_system_prompt": row.get("system_prompt"),
        "tools": _json_field(row, "tools"),
        "menu_sample": _json_field(row, "menu"),
        "expected_cart": _json_field(row, "expected_cart"),
        "_variant": "complex",
        "_source": {
            "repository": hf_repo_id(),
            "revision": hf_revision(),
            "split": "test",
        },
    }
    manifest = {
        "task_id": tid,
        # session_id / game_index come from task_id_map, not the row — the
        # published ids dropped them. Both are required by the internal
        # manifest schema and are what locate the session's noise.
        "session_id": session_id,
        "game_index": game_index,
        "restaurant_name": row.get("restaurant"),
        "clips_dir": str(stage_clips_dir(tid).resolve()),
        "initial_noise_rotation": tree.get("initial_noise_rotation") or [],
        "clips": clips_manifest,
        "_source": {
            "repository": hf_repo_id(),
            "revision": hf_revision(),
            "split": "test",
            # Empty on a revision that publishes the full per-clip struct.
            "_lossy_fields": sorted(lossy),
        },
        # Audio written without a manifest row of its own; the tree
        # references these filenames directly.
        "_audio_only": {k: sorted(v) for k, v in sorted(audio_only.items())},
    }
    _atomic_write_json(tree_path, tree)
    _atomic_write_json(stage_task_config_ref(tid).local_path(), task_config)
    _atomic_write_json(stage_task_config_ref(tid, complex=True).local_path(), task_config)
    _atomic_write_json(stage_customer_clips_ref(tid).local_path(), manifest)


def cache_is_current(tid: str) -> bool:
    """Is the local cache for ``tid`` usable as-is?

    Three conditions, and two of them used to be wrong:

    * **Provenance.** A manifest with no ``_source.repository`` was built
      locally by the internal pipeline. It is authoritative and must never be
      overwritten from Hugging Face — that is data loss, not a cache refresh.
    * **Revision.** A cache built at a different revision must be rebuilt, or
      every existing user's warm cache pins them to the old data forever and a
      republish is invisible to exactly the people who already ran the
      benchmark. A cache flagged ``_lossy_fields`` was written by a consumer
      predating the full per-clip struct and is likewise rebuilt.
    * **Completeness.** Relative ``audio_path`` values are resolved against
      ``ARTIFACTS_PATH``'s parent, matching how ``clips_base_dir`` is anchored
      in ``role_a/decision_tree.js``, so the answer no longer depends on the
      current working directory.
    """
    required = (
        stage_task_config_ref(tid, complex=True),
        stage_decision_tree_ref(tid),
        stage_customer_clips_ref(tid),
    )
    if not all(_exists_local(ref) for ref in required):
        return False
    try:
        manifest = json.loads(
            stage_customer_clips_ref(tid).local_path().read_text())
        source = manifest.get("_source") or {}
        if not source.get("repository"):
            return True  # locally built; authoritative
        if source.get("revision") != hf_revision():
            return False
        if source.get("_lossy_fields"):
            return False
        root = ARTIFACTS_PATH.parent
        paths = [
            p if p.is_absolute() else root / p
            for p in (Path(c["audio_path"]) for c in manifest.get("clips") or [])
        ]
        return bool(paths) and all(
            p.is_file() and p.stat().st_size > 0 for p in paths)
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def _cache_pull_stage_huggingface(tid: str) -> None:
    """Materialize a task from the dataset unless the local cache already has it.

    Three things decide "already has it", and two of them used to be wrong:

    * **Provenance.** A manifest with no ``_source.repository`` was built
      locally by the internal pipeline. It is authoritative and must never be
      overwritten from Hugging Face — doing so is data loss, not a cache
      refresh. The old code reached that path whenever a locally built
      manifest's relative ``audio_path`` failed to resolve from the current
      working directory.
    * **Revision.** A cache built at a different revision has to be rebuilt,
      or every existing user's warm cache silently pins them to the old data
      forever and a republish is invisible to exactly the people who already
      ran the benchmark.
    * **Completeness.** Paths are resolved relative to ``ARTIFACTS_PATH``'s
      parent, matching how ``clips_base_dir`` is anchored in
      ``role_a/decision_tree.js``, so the check no longer depends on cwd.
    """
    if not cache_is_current(tid):
        # Single-tid streaming (`_hf_row`, one call per task) is gone: it
        # linear-scanned the dataset per task and transferred audio for every
        # row it skipped past, which moved ~37 GB to hydrate a 300 MB split
        # one task at a time. Route through the same shard-download path
        # `cache_pull_stages` uses, even for a set of one, so there is only
        # one hydration mechanism to keep correct.
        _materialize_from_shards({tid})


def _noise_for(tid: str) -> str:
    session_id, _ = task_source_session(tid)
    if not session_id:
        raise FileNotFoundError(
            f"no session_id for task {tid!r} in {_TASK_ID_MAP_FILE}; cannot "
            "locate its background noise. Every published task has one — a "
            "miss here means the map and the dataset have drifted apart."
        )
    return session_id


def cache_pull_stage(tid: str) -> None:
    """Hydrate all local inputs required to run one benchmark task."""
    if is_offline():
        return
    _cache_pull_stage_huggingface(tid)
    # Row hydration covers per-task inputs only. The shared clip pool and the
    # owning session's background noise are loose files in the dataset and
    # need their own pull. Both sit outside the early-return above on purpose:
    # a cache that already holds the row may still be missing them.
    cache_pull_shared_clips()
    cache_pull_background_noise(_noise_for(tid))


def _materialize_from_shards(wanted: set[str]) -> None:
    """Materialize `wanted` by reading the parquet shards directly.

    Measured against `load_dataset(streaming=True)`: streaming moves ~0.8 MB/s
    because it fetches row groups over HTTP range requests, so one pass over
    the ~300 MB split costs ~370s. Downloading the same shards with
    `hf_hub_download` and reading them with pyarrow runs at ~5 MB/s and does
    the whole 248-task split in ~60s.

    The shards are also what the Hub caches and dedupes, so a re-run or a
    second model's sweep pays nothing.
    """
    from concurrent.futures import ThreadPoolExecutor

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    names = sorted(f for f in _repo_files()
                   if f.startswith("data/") and f.endswith(".parquet"))
    if not names:
        raise FileNotFoundError(
            f"no data/*.parquet in {hf_repo_id()} at revision {hf_revision()}"
        )
    token = os.environ.get("HF_TOKEN") or None

    def fetch(name: str) -> str:
        return hf_hub_download(
            repo_id=hf_repo_id(), filename=name, repo_type="dataset",
            revision=hf_revision(), token=token,
            cache_dir=str(hf_datasets_cache()),
        )

    with ThreadPoolExecutor(max_workers=min(_HYDRATE_WORKERS, len(names))) as pool:
        local_shards = list(pool.map(fetch, names))

    for shard in local_shards:
        if not wanted:
            break
        table = pq.read_table(shard)
        ids = table.column("task_id").to_pylist()
        hits = [i for i, t in enumerate(ids) if t in wanted]
        if not hits:
            continue
        # Slice per hit so a shard's audio blobs are not all held at once.
        for i in hits:
            row = {c: table.column(c)[i].as_py() for c in table.schema.names}
            _materialize_hf_row(row, row["task_id"])
            wanted.discard(row["task_id"])
    if wanted:
        raise FileNotFoundError(
            f"{len(wanted)} task(s) not found in {hf_repo_id()} at revision "
            f"{hf_revision()}: {sorted(wanted)[:5]}"
        )


def cache_pull_stages(tids: list[str]) -> int:
    """Hydrate many tasks in a single pass over the dataset.

    Prefer this to calling :func:`cache_pull_stage` in a loop when you know
    the full set of tasks up front: hydrating N tasks one at a time downloads
    the parquet shards N times (they are re-fetched from the Hub cache, not
    re-downloaded, but each call still re-reads every shard until its task is
    found). One call here reads each shard exactly once no matter how many of
    the requested tasks it satisfies.

    Returns the number of tasks requested (already-current caches are skipped
    without a download).
    """
    if is_offline():
        return 0
    wanted = {t for t in tids}
    if not wanted:
        return 0

    # Skip tasks whose cache is already good, so a partial re-run is cheap and
    # we can avoid the download entirely when nothing is missing.
    remaining = {tid for tid in sorted(wanted) if not cache_is_current(tid)}
    if remaining:
        _materialize_from_shards(remaining)

    cache_pull_shared_clips()
    for session_id in sorted({_noise_for(t) for t in wanted}):
        cache_pull_background_noise(session_id)
    return len(wanted)


# tid format: "order_up.<restaurant>.<sess_short>.g<N>.s<M>"
_TID_RE = re.compile(
    r"^order_up\.[A-Za-z0-9_]+\.([0-9A-F]{8})\.g(\d+)\.s(\d+)$",
)


def parse_tid(tid: str) -> Optional[tuple[str, int, int]]:
    """Parse a per-stage task id into (sess_short_8, gi, si). Returns None
    for legacy game-level tids (no `.s<M>` suffix)."""
    m = _TID_RE.match(tid)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


# ── Typed convenience helpers ───────────────────────────────────────────
#
# Each helper returns an ArtifactRef. Use ref.local_path() for the
# concrete Path, or pass to read_*/write_*/cache_pull. None of these
# helpers do an implicit pull. Keeps hot loops local and deterministic.


# Session-level ----------------------------------------------------------

def session_dir(session_id: str) -> Path:
    return ARTIFACTS_PATH / "sessions/order_up" / session_id


def _sess(session_id: str, *parts) -> ArtifactRef:
    return _ref("sessions/order_up", session_id, *parts)


def session_audio_ref(session_id: str) -> ArtifactRef:
    return _sess(session_id, "audio.wav")


def session_transcription_ref(session_id: str) -> ArtifactRef:
    return _sess(session_id, "transcription.json")


def session_transcription_original_ref(session_id: str) -> ArtifactRef:
    return _sess(session_id, "transcription.json.original")


def session_stage_alignment_ref(session_id: str) -> ArtifactRef:
    return _sess(session_id, "stage_alignment.json")


def session_game_data_ref(session_id: str) -> ArtifactRef:
    return _sess(session_id, "game_data.json")


def session_metadata_ref(session_id: str) -> ArtifactRef:
    return _sess(session_id, "session_metadata.json")


def session_meta_ref(session_id: str) -> ArtifactRef:
    return _sess(session_id, "_meta.json")


def session_content_match_ref(session_id: str) -> ArtifactRef:
    return _sess(session_id, "content_match.json")


def session_repair_state_ref(session_id: str) -> ArtifactRef:
    return _sess(session_id, "repair_state.json")


def session_timeline_ref(session_id: str) -> ArtifactRef:
    return _sess(session_id, "timeline.json")


def session_timeline_md_ref(session_id: str) -> ArtifactRef:
    return _sess(session_id, "TIMELINE.md")


def session_game_speaker_roles_ref(session_id: str, gi: int) -> ArtifactRef:
    return _sess(session_id, f"_game_speaker_roles_g{gi:02d}.json")


def session_background_noise_manifest_ref(session_id: str) -> ArtifactRef:
    return _ref("background_noise", session_id, "manifest.json")


def session_background_noise_clip_ref(session_id: str, name: str) -> ArtifactRef:
    """``name`` like ``noise_001.wav`` or ``_cand_002_sub05.wav``."""
    return _ref("background_noise", session_id, name)


def session_background_noise_dir(session_id: str) -> Path:
    return ARTIFACTS_PATH / "background_noise" / session_id


# Session-stage (session-scoped per-stage outputs) -----------------------

def stage_classified_spans_ref(session_id: str, gi: int, si: int) -> ArtifactRef:
    return _sess(session_id, "classified_spans", f"g{gi:02d}_s{si:02d}.json")


def stage_audit_ref(session_id: str, gi: int, si: int) -> ArtifactRef:
    return _sess(session_id, "audit", f"g{gi:02d}_s{si:02d}.json")


def stage_categorization_ref(session_id: str, gi: int, si: int) -> ArtifactRef:
    return _sess(session_id, "stage_categorization", f"g{gi:02d}_s{si:02d}.json")


# Stage (tid-keyed; by-kind dirs) ----------------------------------------

def stage_customer_clips_ref(tid: str) -> ArtifactRef:
    return _ref("customer_clips", f"{tid}.json")


def stage_decision_tree_ref(tid: str) -> ArtifactRef:
    return _ref("decision_trees", f"{tid}.json")


def stage_task_config_ref(tid: str, *, complex: bool = False) -> ArtifactRef:
    suffix = ".complex.task.json" if complex else ".task.json"
    return _ref("order_up_tasks", f"{tid}{suffix}")


def stage_refiner_log_ref(tid: str) -> ArtifactRef:
    return _ref("refiner_logs", f"{tid}.json")


def stage_clips_dir(tid: str) -> Path:
    """Per-stage clips dir. Replaces the legacy per-game dir
    (``clips/order_up.<r>.<s>.g<g>/``)."""
    return ARTIFACTS_PATH / "clips" / tid


def stage_clip_ref(tid: str, clip_id: str, *, fallback: bool = False) -> ArtifactRef:
    suffix = ".fallback.wav" if fallback else ".wav"
    return _ref("clips", tid, f"{clip_id}{suffix}")


# Shared -----------------------------------------------------------------

def shared_clip_ref(name: SharedClipName) -> ArtifactRef:
    if name not in SHARED_CLIP_NAMES:
        raise ValueError(
            f"unknown shared clip {name!r}; add it to SHARED_CLIP_NAMES so "
            f"hydration and the completeness check both cover it"
        )
    return _ref("clips/order_up_shared", name)



def shared_compliance_ref() -> ArtifactRef:
    return _ref("classification_compliance.json")


# ── Round-trip helper, callable directly for ops ────────────────────────


def _self_check() -> None:
    """Health check for Hugging Face access and the pinned dataset revision.

    Usage: ``python -m clip_eval.pipeline.artifact_store``
    """
    if is_offline():
        print(f"CLIP_EVAL_OFFLINE=1; local cache root: {ARTIFACTS_PATH}")
        return
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face hydration requires the 'huggingface_hub' package"
        ) from exc
    info = HfApi(token=os.environ.get("HF_TOKEN") or None).dataset_info(
        hf_repo_id(), revision=hf_revision()
    )
    print(f"dataset: https://huggingface.co/datasets/{hf_repo_id()}")
    print(f"revision: {info.sha}")
    print(f"private: {info.private}")
    print(f"local cache root: {ARTIFACTS_PATH}")
    for sibling in (info.siblings or [])[:5]:
        print(f"  {sibling.rfilename}")


if __name__ == "__main__":
    _self_check()
