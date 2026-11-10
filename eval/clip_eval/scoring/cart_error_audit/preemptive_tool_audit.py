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

"""Post-eval audit for ``extra-item`` runs: classify each extra item's
INTENT.

The audit runs in two passes:

1. **Deterministic preemptive check** — did the agent fire a tool call
   BEFORE the first customer clip even finished? That happens when B's
   server-side VAD trips on a phantom signal (agent's own greeting
   echo, noise floor, jitter) and the agent fabricates an order
   without hearing anything. Signals:
     - ``first_b_vad_t``   first ``vad_b_of_a`` event
     - ``first_clip_t``    first ``customer_speech`` event
     - ``first_tool_call_t`` first ``tool_call`` event
   A run is ``preemptive`` when first_b_vad_t < first_clip_t AND
   first_tool_call_t < first_clip_t_end. This is the original audit
   semantic.

2. **LLM intent classifier** — for each extra item in the diff,
   classify the agent's intent into one of:
     - ``instead_of``     — agent added this EXTRA in PLACE OF an
                            intended item. The intended item never
                            gets added AND the agent never engages
                            with it in the transcript. Pairs the extra
                            with the missing intended item.
     - ``in_addition_to`` — agent added this extra ON TOP of the
                            intended items. Intended items are added
                            (initially or later), OR the agent at
                            least asks a follow-up about them.
     - ``preemptive``     — fabricated before the customer's first
                            clip played (cross-references the
                            deterministic check above).
     - ``unclear``        — not enough signal to decide.

The ``instead_of`` classification short-circuits the cart-diff tag
machinery: the paired (missing, extra) entries are pulled out of the
``missing-item`` / ``extra-item`` counts and surfaced as a single
``wrong-item`` tag — symmetric to how ``wrong-option-or-variant``
handles same-base mismatches.

Writes ``preemptive_tool_audit.json`` next to ``score.json``. Schema:
  {
    "preemptive": bool,           # deterministic check
    "vad_before_clip": bool,
    "tool_before_clip_end": bool,
    "first_b_vad_t": float | null,
    "first_clip_t": float | null,
    "first_clip_t_end": float | null,
    "first_tool_call_t": float | null,
    "first_tool_name": str | null,
    "first_tool_args": dict | null,
    "items": [                    # one per extra in the diff
      {
        "extra_item_id": str,
        "extra_options": dict,
        "intent": "instead_of"|"in_addition_to"|"preemptive"|"unclear",
        "paired_missing_item_id": str | null,
        "rationale": str,
        "agent_quote": str | null,
        "turn_t": float | null,
        "read_back": bool,        # did the agent say this extra's name aloud?
        "read_back_quote": str | null
      }
    ],
    "judged": bool,
    "model": str | null,
    "skip_reason": str | null
  }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from clip_eval.pipeline.paths import ARTIFACTS_PATH
from clip_eval.lib.timeline import build as build_viz

TRIGGER_TAG = "extra-item"
# Runs whose extras have been judged ``instead_of`` lose the ``extra-item``
# tag (it gets replaced by ``wrong-item`` after diff-tag re-attachment).
# Both tags should re-trigger the audit so backfills + re-runs are
# idempotent.
TRIGGER_TAGS = {"extra-item", "wrong-item"}


def _first_of_kind(events: list[dict], kind: str) -> dict | None:
    for e in events:
        if e.get("kind") == kind:
            return e
    return None


def _render_cart_items(items: list[dict]) -> str:
    """Render a cart-item list as a bulleted block for the LLM prompt."""
    if not items:
        return "  (none)"
    lines = []
    for it in items:
        item_id = it.get("item_id") or "?"
        opts = it.get("options")
        if isinstance(opts, dict) and opts:
            opt_str = "; ".join(
                f"{k}={','.join(v) if isinstance(v, list) else v}"
                for k, v in opts.items()
            )
            lines.append(f"  - {item_id}  [{opt_str}]")
        else:
            lines.append(f"  - {item_id}")
    return "\n".join(lines)


def _render_transcript(record_dir: Path) -> str:
    """Interleave agent + customer speech events into one timestamped
    role-tagged transcript (same shape as off_policy_audit uses)."""
    tl = build_viz(record_dir)
    rows: list[tuple[float, str, str]] = []
    for e in (tl.get("events") or []):
        k = e.get("kind")
        if k not in ("agent_speech", "customer_speech"):
            continue
        text = (e.get("text") or "").strip()
        if not text:
            continue
        rows.append((float(e.get("t") or 0.0), k, text))
    rows.sort(key=lambda r: r[0])
    return "\n".join(
        f"[{t:6.1f}s] {'AGENT' if kind == 'agent_speech' else 'CUSTOMER'}: {text}"
        for t, kind, text in rows
    )


def _load_stage_envelope(task_id: str) -> dict | None:
    """Pull the per-stage envelope from stage_alignment.json so the LLM
    sees the intended flow (stage_type, has_correction, the items the
    customer ASKED for, the reference cart_actions sequence, etc.).

    Per-stage tids look like ``order_up.<Restaurant>.<8HEXPREFIX>.gN.sM``.
    The 8-hex is a prefix of the session uuid. We resolve the full uuid
    by scanning ``artifacts/sessions/order_up/``.
    """
    if not task_id:
        return None
    parts = task_id.split(".")
    if len(parts) < 5:
        return None
    short = parts[-3]   # e.g. "FC5B850B"
    gi_str = parts[-2]  # "g8"
    si_str = parts[-1]  # "s2"
    if not (gi_str.startswith("g") and si_str.startswith("s")):
        return None
    try:
        gi = int(gi_str[1:])
        si = int(si_str[1:])
    except ValueError:
        return None
    sessions_root = ARTIFACTS_PATH / "sessions" / "order_up"
    if not sessions_root.exists():
        return None
    for d in sessions_root.iterdir():
        if d.is_dir() and d.name.upper().startswith(short.upper()):
            ap = d / "stage_alignment.json"
            if not ap.exists():
                continue
            try:
                a = json.loads(ap.read_text())
                g = a.get("games", [])[gi]
                stages = g.get("stages") or []
                if si >= len(stages):
                    return None
                s = stages[si]
                return {
                    "stage_index": s.get("stage_index"),
                    "stage_type": s.get("stage_type"),
                    "has_correction": s.get("has_correction"),
                    "is_confirmation": s.get("is_confirmation"),
                    "expected_items": s.get("expected_items"),
                    "expected_item_ids": s.get("expected_item_ids"),
                    "items": s.get("items"),
                    "cart_actions": s.get("cart_actions"),
                    "transcript": s.get("transcript"),
                }
            except Exception:
                continue
    return None


def _render_stage_envelope(env: dict | None) -> str:
    if not env:
        return "  (stage envelope not available)"
    lines: list[str] = [
        f"  stage_type: {env.get('stage_type')}",
        f"  has_correction: {env.get('has_correction')}",
        f"  is_confirmation: {env.get('is_confirmation')}",
    ]
    eids = env.get("expected_item_ids") or []
    enames = env.get("expected_items") or []
    if eids:
        pairs = ", ".join(
            f"{n} ({i})" for n, i in zip(enames, eids)
        )
        lines.append(f"  expected (post-correction): {pairs}")
    items = env.get("items") or []
    if items:
        lines.append("  items detail:")
        for it in items:
            lines.append(
                f"    - {it.get('id')}: type={it.get('item_type')}, "
                f"display={it.get('display_text')}"
            )
    cas = env.get("cart_actions") or []
    if cas:
        lines.append("  reference cart_actions (human playthrough):")
        for ca in cas:
            lines.append(
                f"    - [{ca.get('t'):.1f}s] {ca.get('action')} "
                f"{ca.get('item')} ({ca.get('item_id')})"
            )
    ref_tx = (env.get("transcript") or "").strip()
    if ref_tx:
        # Cap to 600 chars — just for high-level context.
        clipped = ref_tx if len(ref_tx) <= 600 else ref_tx[:600] + " …"
        lines.append(f"  reference transcript excerpt: {clipped!r}")
    return "\n".join(lines)


JUDGE_SYSTEM = """\
You are auditing a voice-AI ordering agent. The agent's final cart \
contains EXTRA items the customer never explicitly ordered. For each \
extra item, classify the agent's INTENT.

You have access to:
  - INTENDED ITEMS: what the customer actually ordered (the expected cart)
  - ACTUAL ITEMS: what ended up in the agent's cart
  - EXTRA ITEMS: items in ACTUAL but not in INTENDED — these are the ones to classify
  - MISSING ITEMS: items in INTENDED but not in ACTUAL — candidates for instead_of pairing
  - STAGE FLOW: the intended trajectory from the source stage (type, correction info, expected items, reference cart_actions)
  - PREEMPTIVE FLAG: a deterministic check that fires when the first tool call landed before the customer's first clip even finished. Trust this for "preemptive" — only override when the transcript shows a different pattern.
  - TRANSCRIPT: the role-tagged conversation

For each extra item, choose exactly one intent:

* **instead_of** — the agent added this extra IN PLACE OF an intended (missing) item. Two conditions must hold:
    (i) the intended item NEVER appears in ACTUAL (it's in MISSING), AND
    (ii) the agent NEVER asks a follow-up about it or attempts to add it anywhere in the transcript.
  When you pick this, identify the specific missing intended item it replaced (the one most likely confused with the extra — similar audio, similar role, or context).

* **in_addition_to** — the agent added this extra ON TOP of the intended order. ANY of these qualifies:
    - the intended items DO appear in ACTUAL, OR
    - the agent asks a follow-up question about an intended item (e.g. "What about the X?"), OR
    - the agent attempts to add an intended item later in the conversation (even if it doesn't end up in the final cart for some other reason).
  The extra is a hallucinated addition, not a substitution.

* **preemptive** — the agent fabricated this item BEFORE the customer's first clip played. Use this when the PREEMPTIVE FLAG is true AND the transcript supports it. Don't invent this when the flag is false.

* **unclear** — genuine ambiguity. Use sparingly.

ALSO, for EVERY extra item (regardless of intent), flag whether the agent VERBALLY READ BACK or CONFIRMED the name of the incorrect item to the customer at any point in the transcript. "Read back" means the agent said the item's name (or an obvious synonym / menu phrasing) out loud — e.g. "Got it, mango lassi added", "So that's a chicken burrito", or summarizing the order with the item in it. Generic acknowledgments like "Got it", "Okay", or "Anything else?" do NOT count. The point is to know whether the customer had a chance to catch the mistake — set `read_back: true` only when the spoken name of THIS specific extra appears in the agent's speech.

Output strict JSON:
{
  "items": [
    {
      "extra_item_id": "<BARE item id only, e.g. 'mango-lassi' — never include the options block>",
      "intent": "instead_of" | "in_addition_to" | "preemptive" | "unclear",
      "paired_missing_item_id": "<BARE missing item id, or null>",
      "rationale": "<one sentence citing the transcript evidence>",
      "agent_quote": "<verbatim quote from the transcript that supports the intent call, or null>",
      "turn_t": <number from the [t] prefix, or null>,
      "read_back": <true | false>,
      "read_back_quote": "<verbatim agent quote that names the item, or null when read_back is false>"
    },
    ...
  ]
}

Item ids: use ONLY the hyphenated identifier (the first token in the "- <id>" line of EXTRA / MISSING ITEMS). DO NOT include the bracketed options string in extra_item_id or paired_missing_item_id. Example: write "mango-lassi", NOT "mango-lassi  [Dairy=Whole Milk Yogurt; Sweetness=Regular]".

Include exactly one entry per EXTRA item. Quote the agent verbatim — no paraphrasing.
"""


JUDGE_USER_TMPL = """\
INTENDED ITEMS (the customer's expected cart)
{intended_block}

ACTUAL ITEMS (the agent's final cart)
{actual_block}

EXTRA ITEMS (classify each of these)
{extra_block}

MISSING ITEMS (candidates for instead_of pairing)
{missing_block}

STAGE FLOW
{stage_block}

PREEMPTIVE FLAG: {preemptive}
  first_b_vad_t={first_b_vad_t}, first_clip_t={first_clip_t}, \
first_clip_t_end={first_clip_t_end}, first_tool_call_t={first_tool_call_t}

TRANSCRIPT
{transcript}
"""


def call_judge(
    intended: list[dict], actual: list[dict],
    extras: list[dict], missing: list[dict],
    stage_env: dict | None, transcript: str,
    preemptive: bool, timing: dict,
    *, model: str = "gpt-5-mini",
) -> dict:
    """One LLM call. Returns the parsed JSON object."""
    from openai import OpenAI
    client = OpenAI()
    user_msg = JUDGE_USER_TMPL.format(
        intended_block=_render_cart_items(intended),
        actual_block=_render_cart_items(actual),
        extra_block=_render_cart_items(extras),
        missing_block=_render_cart_items(missing),
        stage_block=_render_stage_envelope(stage_env),
        preemptive=str(preemptive).lower(),
        first_b_vad_t=timing.get("first_b_vad_t"),
        first_clip_t=timing.get("first_clip_t"),
        first_clip_t_end=timing.get("first_clip_t_end"),
        first_tool_call_t=timing.get("first_tool_call_t"),
        transcript=transcript or "(no transcript)",
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def _coerce_id(raw_id: str | None, candidates: set[str]) -> str | None:
    """LLMs sometimes echo back the rendered cart line (e.g.
    ``"mango-lassi  [Dairy=Whole Milk Yogurt; Sweetness=Regular]"``)
    instead of just the bare id. Strip any whitespace + bracketed
    content and match against the real id set.
    """
    if not raw_id:
        return None
    if raw_id in candidates:
        return raw_id
    stripped = raw_id.split("[", 1)[0].strip()
    if stripped in candidates:
        return stripped
    # Last-ditch: tolerate slight tail garbage (trailing punctuation,
    # double spaces) by checking each candidate as a prefix of stripped.
    for c in candidates:
        if stripped.startswith(c) and (
            len(stripped) == len(c) or not stripped[len(c)].isalnum()
        ):
            return c
    return None


def _normalize_judge_items(
    judge_items: list[dict],
    extras: list[dict],
    missing: list[dict],
) -> list[dict]:
    """Sanitize the LLM output: align entries with the actual extras list,
    drop spurious extra_item_ids, ensure paired_missing_item_id refers
    to a real missing item.
    """
    extra_ids = {e.get("item_id") for e in extras if e.get("item_id")}
    missing_ids = {m.get("item_id") for m in missing if m.get("item_id")}
    by_extra: dict[str, dict] = {}
    for it in (judge_items or []):
        eid = _coerce_id(it.get("extra_item_id"), extra_ids)
        if eid and eid not in by_extra:
            paired = _coerce_id(it.get("paired_missing_item_id"), missing_ids)
            intent = it.get("intent")
            if intent not in ("instead_of", "in_addition_to", "preemptive", "unclear"):
                intent = "unclear"
            # instead_of without a valid pair → demote to unclear
            if intent == "instead_of" and not paired:
                intent = "unclear"
            read_back = bool(it.get("read_back"))
            read_back_quote = it.get("read_back_quote") or None
            # Drop read_back_quote when the LLM reported read_back=false
            # but still echoed a quote (cleanest schema for downstream).
            if not read_back:
                read_back_quote = None
            by_extra[eid] = {
                "extra_item_id": eid,
                "extra_options": next(
                    (e.get("options") for e in extras if e.get("item_id") == eid),
                    None,
                ),
                "intent": intent,
                "paired_missing_item_id": paired if intent == "instead_of" else None,
                "rationale": (it.get("rationale") or "")[:400],
                "agent_quote": (it.get("agent_quote") or None),
                "turn_t": it.get("turn_t"),
                "read_back": read_back,
                "read_back_quote": read_back_quote,
            }
    # Fill in any extras the LLM omitted as "unclear"
    out = []
    for e in extras:
        eid = e.get("item_id")
        if eid in by_extra:
            out.append(by_extra[eid])
        else:
            out.append({
                "extra_item_id": eid,
                "extra_options": e.get("options"),
                "intent": "unclear",
                "paired_missing_item_id": None,
                "rationale": "judge omitted this extra from its response",
                "agent_quote": None,
                "turn_t": None,
                "read_back": False,
                "read_back_quote": None,
            })
    return out


def run_audit(
    record_dir: Path, *,
    force: bool = False,
    skip_llm: bool = False,
    model: str = "gpt-5-mini",
) -> dict:
    """Audit one run dir. Returns a result dict; the caller writes it
    to disk. ``force=True`` skips the tag gate (useful for demos);
    ``skip_llm=True`` only runs the deterministic preemptive check.
    """
    score_path = record_dir / "score.json"
    out: dict = {
        "record_dir": str(record_dir),
        "trigger_tag": TRIGGER_TAG,
        "judged": False,
        "skip_reason": None,
        "preemptive": False,
        "items": [],
        "model": None,
    }
    if not score_path.exists():
        out["skip_reason"] = "no score.json"
        return out
    score = json.loads(score_path.read_text())
    tags = list(score.get("tags") or [])
    out["tags"] = tags
    out["task_id"] = score.get("task_id") or record_dir.parent.name
    out["variant"] = "complex"
    if not force and not (TRIGGER_TAGS & set(tags)):
        out["skip_reason"] = f"none of {sorted(TRIGGER_TAGS)} on this run"
        return out

    tl = build_viz(record_dir)
    events = tl.get("events") or []

    # Deterministic preemptive check (unchanged).
    first_vad = _first_of_kind(events, "vad_b_of_a")
    first_clip = _first_of_kind(events, "customer_speech")
    first_tool = _first_of_kind(events, "tool_call")
    out["first_b_vad_t"] = first_vad.get("t") if first_vad else None
    out["first_clip_t"] = first_clip.get("t") if first_clip else None
    out["first_clip_t_end"] = first_clip.get("t_end") if first_clip else None
    out["first_tool_call_t"] = first_tool.get("t") if first_tool else None
    out["first_tool_name"] = first_tool.get("tool") if first_tool else None
    out["first_tool_args"] = first_tool.get("args") if first_tool else None
    out["judged"] = True
    if first_vad is not None and first_clip is not None and first_tool is not None:
        vad_before_clip = first_vad["t"] < first_clip["t"]
        tool_before_clip_end = first_tool["t"] < (
            first_clip.get("t_end") or first_clip["t"]
        )
        out["vad_before_clip"] = bool(vad_before_clip)
        out["tool_before_clip_end"] = bool(tool_before_clip_end)
        out["preemptive"] = bool(vad_before_clip and tool_before_clip_end)

    # Per-extra intent classification (LLM).
    diff = score.get("diff") or {}
    extras = list(diff.get("extra") or [])
    missing = list(diff.get("missing") or [])
    intended = list(score.get("expected_cart") or [])
    actual = list(score.get("actual_cart") or [])

    if skip_llm:
        out["skip_reason"] = "skip_llm=True (deterministic only)"
        return out
    if not extras:
        out["skip_reason"] = "no extras in diff (tag stale?)"
        return out

    transcript = _render_transcript(record_dir)
    stage_env = _load_stage_envelope(out["task_id"])

    timing = {
        "first_b_vad_t": out["first_b_vad_t"],
        "first_clip_t": out["first_clip_t"],
        "first_clip_t_end": out["first_clip_t_end"],
        "first_tool_call_t": out["first_tool_call_t"],
    }
    judge_out = call_judge(
        intended, actual, extras, missing, stage_env, transcript,
        out["preemptive"], timing, model=model,
    )
    out["items"] = _normalize_judge_items(
        judge_out.get("items") or [], extras, missing,
    )
    out["model"] = model
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("record_dir", type=Path)
    ap.add_argument("--force", action="store_true",
                    help=f"run even if {TRIGGER_TAG!r} isn't on this run")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--skip-llm", action="store_true",
                    help="only run the deterministic preemptive check")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the prompt that would go to the LLM, but don't call it")
    ap.add_argument("--out", type=Path,
                    help="output path (default: "
                         "<record_dir>/preemptive_tool_audit.json)")
    args = ap.parse_args()

    if args.dry_run:
        # Build the same inputs the audit would feed to the LLM, then dump.
        score = json.loads((args.record_dir / "score.json").read_text())
        diff = score.get("diff") or {}
        extras = list(diff.get("extra") or [])
        missing = list(diff.get("missing") or [])
        intended = list(score.get("expected_cart") or [])
        actual = list(score.get("actual_cart") or [])
        transcript = _render_transcript(args.record_dir)
        stage_env = _load_stage_envelope(score.get("task_id"))
        msg = JUDGE_USER_TMPL.format(
            intended_block=_render_cart_items(intended),
            actual_block=_render_cart_items(actual),
            extra_block=_render_cart_items(extras),
            missing_block=_render_cart_items(missing),
            stage_block=_render_stage_envelope(stage_env),
            preemptive="(deterministic check not run in dry-run)",
            first_b_vad_t="(n/a)", first_clip_t="(n/a)",
            first_clip_t_end="(n/a)", first_tool_call_t="(n/a)",
            transcript=transcript or "(no transcript)",
        )
        print("=== SYSTEM ===")
        print(JUDGE_SYSTEM)
        print()
        print("=== USER ===")
        print(msg)
        return 0

    out = run_audit(
        args.record_dir, force=args.force,
        skip_llm=args.skip_llm, model=args.model,
    )
    out_path = args.out or (args.record_dir / "preemptive_tool_audit.json")
    out_path.write_text(json.dumps(out, indent=2))
    if not out.get("judged"):
        print(f"[extra_item_audit] {args.record_dir.name}: skipped "
              f"({out.get('skip_reason')})")
    else:
        items = out.get("items") or []
        verdict = "PREEMPTIVE" if out["preemptive"] else "non-preemptive"
        intents = ", ".join(
            f"{it['extra_item_id']}={it['intent']}" for it in items
        ) or "(no extras)"
        print(f"[extra_item_audit] {args.record_dir.name}: {verdict}  "
              f"intents: {intents} → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
