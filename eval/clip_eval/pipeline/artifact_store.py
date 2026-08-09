"""GCS-backed artifact store for the clip_eval pipeline.

Every pipeline artifact (transcription, classified spans, audit envelopes,
customer clips, decision trees, task configs, refiner logs, background
noise, …) has a canonical local path under ``ARTIFACTS_PATH`` and a
corresponding GCS object under ``gs://<bucket>/<prefix>/``. The local
layout and the GCS object names are identical relative to those roots,
so an artifact at ``customer_clips/<tid>.json`` locally lives at
``<prefix>/customer_clips/<tid>.json`` in GCS.

This module unifies access:

  * ``read_*`` ensures a local copy exists (fetching from GCS if needed)
    and returns the content.
  * ``write_*`` writes locally and uploads to GCS in one call, gated by
    a per-step :class:`OverwritePolicy`.
  * ``cache_pull_session(session_id)`` / ``cache_pull_stage(tid)`` hydrate
    every artifact a downstream step needs in one batched fetch, so the
    inner loop never blocks on GCS round-trips.

Configuration (all env vars are optional)::

    CLIP_EVAL_GCS_PROJECT  GCP project ID for the storage client. When
                           unset the google-cloud-storage SDK derives
                           the project from ADC.
    CLIP_EVAL_GCS_BUCKET   default: gathered-sensor-data
    CLIP_EVAL_GCS_PREFIX   default: eval-datasets/clip-eval/v2-artifacts
    ARTIFACTS_PATH         local cache root (existing override)
    CLIP_EVAL_OFFLINE      "1" / "true" → no GCS calls; reads must hit
                           local cache, writes skip upload
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

from clip_eval.pipeline.paths import (
    SESSIONS_ROOT,
    ARTIFACTS_PATH,
)


# ── Configuration ───────────────────────────────────────────────────────

DEFAULT_BUCKET = "gathered-sensor-data"
DEFAULT_PREFIX = "eval-datasets/clip-eval/v2-artifacts"


def gcs_project() -> Optional[str]:
    """GCP project ID for the storage client. When None, the SDK
    derives it from ADC (gcloud config, GOOGLE_CLOUD_PROJECT,
    metadata server, etc.)."""
    return os.environ.get("CLIP_EVAL_GCS_PROJECT") or None


def gcs_bucket_name() -> str:
    return os.environ.get("CLIP_EVAL_GCS_BUCKET") or DEFAULT_BUCKET


def gcs_prefix() -> str:
    return (os.environ.get("CLIP_EVAL_GCS_PREFIX") or DEFAULT_PREFIX).strip("/")


def is_offline() -> bool:
    return (os.environ.get("CLIP_EVAL_OFFLINE") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )


# ── Lazy GCS client ─────────────────────────────────────────────────────
#
# google-cloud-storage import is heavy (~1s) and ADC discovery hits the
# filesystem + metadata server, so we defer both until the first call
# that actually needs the bucket. Offline mode never instantiates it.

_client = None
_client_lock = threading.Lock()


def _get_bucket():
    global _client
    if is_offline():
        raise RuntimeError(
            "GCS access requested in offline mode (CLIP_EVAL_OFFLINE=1). "
            "Cached local copy must already exist for reads."
        )
    with _client_lock:
        if _client is None:
            from google.cloud import storage
            project = gcs_project()
            _client = storage.Client(project=project) if project else storage.Client()
    return _client.bucket(gcs_bucket_name())


# ── OverwritePolicy ─────────────────────────────────────────────────────


class OverwritePolicy(str, Enum):
    """Controls how :func:`write_bytes` / :func:`write_json` handles
    an existing artifact.

    OVERWRITE        — unconditional replace (local + GCS).
    SKIP_IF_EXISTS   — no-op if any copy exists (local or GCS). Used for
                       once-only backups like ``transcription.json.original``.
    SKIP_IF_LOCAL    — no-op if a local copy exists, regardless of GCS.
                       Still uploads on first write if local exists but
                       GCS doesn't. Used for cache-style outputs like
                       ``transcription.json`` post-first-transcribe and
                       per-file ``background_noise/noise_*.wav``.
    MERGE_JSON       — read-modify-write: load existing JSON (local or
                       GCS), dict/list-merge the new payload on top,
                       write. Used for ``repair_state.json``.
    """

    OVERWRITE = "overwrite"
    SKIP_IF_EXISTS = "skip_if_exists"
    SKIP_IF_LOCAL = "skip_if_local"
    MERGE_JSON = "merge_json"


# ── ArtifactRef ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to one artifact.

    ``path`` is the path relative to ``ARTIFACTS_PATH`` locally and to
    ``gcs_prefix()`` in GCS. Helpers below construct refs so callers
    rarely build one by hand.
    """

    path: Path

    def local_path(self) -> Path:
        return ARTIFACTS_PATH / self.path

    def gcs_object(self) -> str:
        return f"{gcs_prefix()}/{self.path.as_posix()}"

    def gcs_uri(self) -> str:
        return f"gs://{gcs_bucket_name()}/{self.gcs_object()}"


def _ref(*parts) -> ArtifactRef:
    return ArtifactRef(Path(*parts))


# ── Existence / fetch / write ───────────────────────────────────────────


def _exists_local(ref: ArtifactRef) -> bool:
    p = ref.local_path()
    return p.exists() and (p.is_dir() or p.stat().st_size > 0)


def _exists_gcs(ref: ArtifactRef) -> bool:
    if is_offline():
        return False
    return _get_bucket().blob(ref.gcs_object()).exists()


def exists(ref: ArtifactRef, *, where: Literal["local", "gcs", "either"] = "either") -> bool:
    if where == "local":
        return _exists_local(ref)
    if where == "gcs":
        return _exists_gcs(ref)
    return _exists_local(ref) or _exists_gcs(ref)


def cache_pull(ref: ArtifactRef) -> Path:
    """Ensure a local copy exists; download from GCS if missing. Returns
    the local Path. In offline mode this is a no-op when the local file
    already exists, and raises otherwise.
    """
    local = ref.local_path()
    if _exists_local(ref):
        return local
    if is_offline():
        raise FileNotFoundError(
            f"artifact missing locally and CLIP_EVAL_OFFLINE=1: {ref.gcs_uri()}"
        )
    blob = _get_bucket().blob(ref.gcs_object())
    if not blob.exists():
        raise FileNotFoundError(f"artifact not in GCS: {ref.gcs_uri()}")
    local.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local))
    return local


def read_bytes(ref: ArtifactRef, *, prefer_local: bool = True) -> bytes:
    if prefer_local and _exists_local(ref):
        return ref.local_path().read_bytes()
    return cache_pull(ref).read_bytes()


def read_json(ref: ArtifactRef, *, prefer_local: bool = True) -> dict:
    return json.loads(read_bytes(ref, prefer_local=prefer_local).decode("utf-8"))


def _upload(ref: ArtifactRef, data: bytes) -> None:
    if is_offline():
        return
    blob = _get_bucket().blob(ref.gcs_object())
    blob.upload_from_string(data)


def write_bytes(
    ref: ArtifactRef,
    data: bytes,
    *,
    policy: OverwritePolicy = OverwritePolicy.OVERWRITE,
    upload: bool = True,
) -> None:
    """Write ``data`` to the local cache and (unless ``upload=False`` or
    offline) the corresponding GCS object, per ``policy``."""
    if policy == OverwritePolicy.SKIP_IF_EXISTS:
        if _exists_local(ref) or (upload and _exists_gcs(ref)):
            return
    elif policy == OverwritePolicy.SKIP_IF_LOCAL:
        if _exists_local(ref):
            # Back-fill GCS if local exists but GCS doesn't.
            if upload and not is_offline() and not _exists_gcs(ref):
                _upload(ref, ref.local_path().read_bytes())
            return
    elif policy == OverwritePolicy.MERGE_JSON:
        raise ValueError("MERGE_JSON only valid for write_json")
    local = ref.local_path()
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    if upload:
        _upload(ref, data)


def write_json(
    ref: ArtifactRef,
    obj,
    *,
    policy: OverwritePolicy = OverwritePolicy.OVERWRITE,
    upload: bool = True,
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
        local = ref.local_path()
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        if upload:
            _upload(ref, data)
        return
    data = json.dumps(obj, indent=indent).encode("utf-8")
    write_bytes(ref, data, policy=policy, upload=upload)


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


def delete(ref: ArtifactRef, *, local: bool = True, gcs: bool = True) -> None:
    if local:
        p = ref.local_path()
        if p.exists():
            if p.is_dir():
                import shutil
                shutil.rmtree(p)
            else:
                p.unlink()
    if gcs and not is_offline():
        blob = _get_bucket().blob(ref.gcs_object())
        if blob.exists():
            blob.delete()


# ── Bulk cache hydration ────────────────────────────────────────────────


def cache_pull_session(
    session_id: str,
    *,
    include_audio: bool = True,
    include_background_noise: bool = True,
    include_stages: bool = False,
) -> None:
    """Pull every session-level artifact for one session in a batched fetch.

    ``include_audio=False`` skips the multi-MB audio.wav when the caller
    doesn't need it. ``include_stages=True`` also pulls every per-stage
    classified_spans/audit/stage_categorization JSON under the session.
    Background noise lives at top-level ``background_noise/<sid>/``,
    not under the session dir.
    """
    if is_offline():
        return
    bucket = _get_bucket()
    base = f"{gcs_prefix()}/sessions/order_up/{session_id}/"
    for blob in bucket.list_blobs(prefix=base):
        rel = blob.name[len(base):]
        if not rel:
            continue
        if not include_audio and rel == "audio.wav":
            continue
        if not include_stages and (
            rel.startswith("classified_spans/")
            or rel.startswith("audit/")
            or rel.startswith("stage_categorization/")
        ):
            continue
        local = ARTIFACTS_PATH / "sessions/order_up" / session_id / rel
        if local.exists() and local.stat().st_size > 0:
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local))
    # Background noise (top-level layout).
    if include_background_noise:
        bn_base = f"{gcs_prefix()}/background_noise/{session_id}/"
        for blob in bucket.list_blobs(prefix=bn_base):
            rel = blob.name[len(bn_base):]
            if not rel:
                continue
            local = ARTIFACTS_PATH / "background_noise" / session_id / rel
            if local.exists() and local.stat().st_size > 0:
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local))


def cache_pull_stage(tid: str) -> None:
    """Pull every artifact a per-stage eval needs: the four per-stage
    artifacts (customer_clips, decision_tree, task_config, refiner_log)
    + the per-stage clips/<tid>/ dir + the owning session's transcription
    + background_noise + matching classified_spans/audit + shared TTS
    clips.
    """
    if is_offline():
        return
    bucket = _get_bucket()

    # 1) The four per-stage JSONs.
    for ref in (
        stage_customer_clips_ref(tid),
        stage_decision_tree_ref(tid),
        stage_task_config_ref(tid),
        stage_refiner_log_ref(tid),
    ):
        if not _exists_local(ref) and _exists_gcs(ref):
            cache_pull(ref)
    # complex task variant is optional
    cref = stage_task_config_ref(tid, complex=True)
    if not _exists_local(cref) and _exists_gcs(cref):
        cache_pull(cref)

    # 2) The per-stage clips dir.
    clip_prefix = f"{gcs_prefix()}/clips/{tid}/"
    for blob in bucket.list_blobs(prefix=clip_prefix):
        rel = blob.name[len(clip_prefix):]
        if not rel:
            continue
        local = ARTIFACTS_PATH / "clips" / tid / rel
        if local.exists() and local.stat().st_size > 0:
            continue
        local.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local))

    # 3) The owning session — extract from tid.
    parsed = parse_tid(tid)
    if parsed is not None:
        sess_short, gi, si = parsed
        sess_id = _resolve_full_session_id(sess_short)
        if sess_id:
            cache_pull_session(
                sess_id,
                include_audio=False,
                include_background_noise=True,
                include_stages=False,
            )
            # Pull this stage's classified_spans + audit + categorization.
            for ref in (
                stage_classified_spans_ref(sess_id, gi, si),
                stage_audit_ref(sess_id, gi, si),
                stage_categorization_ref(sess_id, gi, si),
            ):
                if not _exists_local(ref) and _exists_gcs(ref):
                    cache_pull(ref)
            ref = session_game_speaker_roles_ref(sess_id, gi)
            if not _exists_local(ref) and _exists_gcs(ref):
                cache_pull(ref)

    # 4) Shared TTS clips.
    for name in ("yes.wav", "no.wav", "thats_all.wav"):
        ref = shared_clip_ref(name)
        if not _exists_local(ref) and _exists_gcs(ref):
            cache_pull(ref)


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


def _resolve_full_session_id(sess_short: str) -> Optional[str]:
    """Resolve "18456B1A" → "18456B1A-6DD6-4DA1-B888-25480241D4DF"."""
    cached = _session_id_cache.get(sess_short)
    if cached:
        return cached
    if SESSIONS_ROOT.exists():
        for child in SESSIONS_ROOT.iterdir():
            if child.is_dir() and child.name.startswith(sess_short):
                _session_id_cache[sess_short] = child.name
                return child.name
    if is_offline():
        return None
    bucket = _get_bucket()
    base = f"{gcs_prefix()}/sessions/order_up/"
    page_iter = bucket.list_blobs(prefix=base, delimiter="/").pages
    for page in page_iter:
        for p in page.prefixes:
            name = p[len(base):].rstrip("/")
            if name.startswith(sess_short):
                _session_id_cache[sess_short] = name
                return name
    return None


_session_id_cache: dict[str, str] = {}


# ── Typed convenience helpers ───────────────────────────────────────────
#
# Each helper returns an ArtifactRef. Use ref.local_path() for the
# concrete Path, or pass to read_*/write_*/cache_pull. None of these
# helpers do an implicit pull — that's the caller's job (via cache_pull
# or cache_pull_*). Keeps hot loops fast and makes any unexpected GCS
# touch easy to spot.


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

def shared_clip_ref(name: Literal["yes.wav", "no.wav", "thats_all.wav"]) -> ArtifactRef:
    return _ref("clips/order_up_shared", name)


def shared_compliance_ref() -> ArtifactRef:
    return _ref("classification_compliance.json")


# ── Round-trip helper, callable directly for ops ────────────────────────


def _self_check() -> None:
    """Health check: list one blob and print bucket+prefix.

    Usage: ``python -m clip_eval.pipeline.artifact_store``
    """
    if is_offline():
        print(f"CLIP_EVAL_OFFLINE=1; local cache root: {ARTIFACTS_PATH}")
        return
    bucket = _get_bucket()
    print(f"bucket: gs://{gcs_bucket_name()}/{gcs_prefix()}/")
    print(f"local cache root: {ARTIFACTS_PATH}")
    n = 0
    for blob in bucket.list_blobs(prefix=gcs_prefix() + "/", max_results=5):
        print(f"  {blob.name}  ({blob.size or 0} bytes)")
        n += 1
    if n == 0:
        print("  (no objects under prefix yet)")


if __name__ == "__main__":
    _self_check()
