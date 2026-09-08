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

"""Post-eval audit for ``missing-item`` runs: classify the agent's
ROOT CAUSE for each missing item into one of:

1. ``premature_submit`` — the agent called ``submit_order`` before the
   customer signalled they were done (no "that's all" / "no" in
   response to "anything else?"). Submitting early truncates the order
   and items the customer hadn't yet been asked about disappear.

2. ``compound_line`` — the customer named TWO OR MORE items in a
   single utterance, and the agent registered fewer than all of them.
   E.g. customer says "I'll have a margherita pizza and a Caesar
   salad" and the agent only adds the pizza.

3. ``clarify_abandoned`` — the agent asked a clarifying question
   about one item, but the customer's next utterance moved on to a
   DIFFERENT item. The agent (mis-)attributed the new utterance to
   the still-open clarification rather than treating it as a fresh
   request, so the item the customer named gets dropped.

4. ``other`` — none of the above (agent silenced the intro clip,
   misheard, hallucinated a substitution, etc.). Use when the failure
   mode doesn't match the three above.

Writes ``missing_item_intent_audit.json`` next to ``score.json``. One
entry per missing item. Schema mirrors ``preemptive_tool_audit.json``:
  {
    "judged": bool,
    "model": str | null,
    "skip_reason": str | null,
    "task_id": str,
    "variant": str,
    "tags": [...],
    "submit_order_t": float | null,
    "submit_order_args": dict | null,
    "last_customer_speech_t": float | null,
    "items": [
      {
        "missing_item_id": str,
        "missing_options": dict | null,
        "intent": "premature_submit" | "compound_line" |
                  "clarify_abandoned" | "other",
        "rationale": str,
        "evidence_quote": str | null,
        "turn_t": float | null
      }
    ]
  }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from clip_eval.scoring.cart_error_audit.preemptive_tool_audit import (
    _render_cart_items,
    _render_stage_envelope,
    _render_transcript,
    _load_stage_envelope,
    _coerce_id,
)
from clip_eval.lib.timeline import build as build_viz


TRIGGER_TAG = "missing-item"


def _render_tool_calls(events: list[dict]) -> str:
    """Compact summary of every tool_call event."""
    rows = []
    for e in events:
        if e.get("kind") != "tool_call":
            continue
        t = float(e.get("t") or 0.0)
        tool = e.get("tool") or "?"
        args = e.get("args") or {}
        rows.append(f"  [{t:6.1f}s] {tool}({json.dumps(args, default=str)[:160]})")
    if not rows:
        return "  (no tool calls)"
    return "\n".join(rows)


def _find_submit_order(events: list[dict]) -> tuple[float | None, dict | None]:
    """Find the first ``submit_order`` tool_call. Returns (t, args)."""
    for e in events:
        if e.get("kind") == "tool_call" and (e.get("tool") or "") == "submit_order":
            return float(e.get("t") or 0.0), e.get("args")
    return None, None


def _last_customer_speech_t(events: list[dict]) -> float | None:
    last = None
    for e in events:
        if e.get("kind") == "customer_speech":
            last = float(e.get("t_end") or e.get("t") or 0.0)
    return last


JUDGE_SYSTEM = """\
You are auditing a voice-AI ordering agent. The agent's final cart \
is MISSING items the customer asked for. For each missing item, \
classify the ROOT CAUSE into exactly ONE of four buckets.

You have:
  - INTENDED ITEMS: what the customer actually ordered
  - ACTUAL ITEMS: what the agent's cart contained at submit time
  - MISSING ITEMS: items in INTENDED but not in ACTUAL — classify each
  - STAGE FLOW: the intended trajectory (stage_type, expected items, reference cart_actions)
  - TOOL CALLS: every tool the agent fired, with timestamps
  - SUBMIT TIMING: when submit_order fired and the last customer utterance
  - TRANSCRIPT: the role-tagged conversation with timestamps

IMPORTANT OVERRIDE — **premature_submit supersedes everything else**. \
First decide whether the AGENT submitted the order without the \
customer signalling they were done. The customer "signalled done" \
ONLY when, in response to one of the agent's "anything else?" / \
"is that everything?" prompts, the customer affirmatively closed \
the order (e.g. "that's all", "no", "no thanks", "I'm good", \
"nope, that's it"). A SILENT customer between the agent's last \
"anything else?" and submit_order is NOT closure. If the agent \
submitted without that closure signal, classify EVERY missing \
item in this run as **premature_submit** — including missing \
items that never appear in the transcript (they didn't appear \
because the agent didn't give the customer the chance to ask). \
Only fall through to the other buckets when the customer DID \
signal closure and items are still missing.

Buckets:

* **premature_submit** — the agent called ``submit_order`` BEFORE \
the customer's order was complete. Use this when ANY of:
    - the agent never received a closure signal between its last \
"anything else?" prompt and submit_order, OR
    - submit_order timestamp precedes the customer's last named item \
in the transcript, OR
    - the missing item appears in the customer's transcript AFTER the \
submit_order tool call.
  When this applies, ALL missing items in this run get this bucket — \
the premature submit is the ROOT cause; downstream individual-item \
failure modes are subordinate.

* **compound_line** — the customer named TWO OR MORE items in a \
single utterance/turn and the agent registered fewer than all of them. \
Signals:
    - one customer turn mentions multiple items (e.g. "pizza and a \
salad", "burger, fries, and a coke"), AND
    - the agent's immediate response acknowledges only a subset (often \
with a clarifying question about ONE of the named items), AND
    - the omitted item from that same turn ends up MISSING.
  This is "agent dropped one of N items the customer said together."

* **clarify_abandoned** — the agent asks a clarifying question about \
ONE item, the customer's NEXT utterance does NOT answer that \
question (the customer names a different item, or otherwise moves \
on), and the clarification context falls apart so an item gets \
DROPPED. The dropped item can be EITHER side of the exchange:
    - **item A dropped** (the clarified item): agent asked clarification \
about A, customer moved on without answering, agent never circles \
back to add A → A ends up MISSING. (e.g. agent: "BLT with what \
spread?" → customer: "Mango sticky rice" → agent adds mango, A=BLT \
gets forgotten.)
    - **item B dropped** (the new item): agent asked clarification \
about A, customer named B, agent treats B's words as A's \
clarification answer and never registers B → B ends up MISSING. \
(e.g. agent: "Chai hot or iced?" → customer: "Milkshake" → agent \
applies "milkshake" as a property of the chai, milkshake gets \
forgotten.)
  The unifying signal: a clarifying question was outstanding when \
the customer changed direction, and something got dropped as a \
result. Use this for either direction.

* **other** — none of the above. Includes: agent silenced the intro \
clip (clipPlayer timeout — the agent never spoke after the customer \
introduced the item), misheard the item as a different one (which \
became an EXTRA, not a missing), substitution patterns, or just \
forgot the item mid-conversation.

For EACH missing item, output exactly one bucket. Quote the \
transcript verbatim to back your choice. Use timestamps from the \
[t] prefix.

Output strict JSON:
{
  "items": [
    {
      "missing_item_id": "<BARE item id only, e.g. 'bbq-burger'>",
      "intent": "premature_submit" | "compound_line" | "clarify_abandoned" | "other",
      "rationale": "<one sentence citing the transcript evidence>",
      "evidence_quote": "<verbatim transcript line that supports the call, or null>",
      "turn_t": <number from the [t] prefix, or null>
    },
    ...
  ]
}

Include exactly one entry per MISSING item. Item ids: hyphenated \
identifier only — NEVER include the bracketed options string.
"""


JUDGE_USER_TMPL = """\
INTENDED ITEMS (the customer's expected cart)
{intended_block}

ACTUAL ITEMS (the agent's final cart)
{actual_block}

MISSING ITEMS (classify each of these)
{missing_block}

STAGE FLOW
{stage_block}

TOOL CALLS
{tool_calls_block}

SUBMIT TIMING
  submit_order_t: {submit_order_t}
  last_customer_speech_t: {last_customer_speech_t}

TRANSCRIPT
{transcript}
"""


def call_judge(
    intended: list[dict], actual: list[dict], missing: list[dict],
    stage_env: dict | None, transcript: str,
    tool_calls_block: str, submit_t: float | None,
    last_customer_t: float | None,
    *, model: str = "gpt-5-mini",
) -> dict:
    from openai import OpenAI
    client = OpenAI()
    user_msg = JUDGE_USER_TMPL.format(
        intended_block=_render_cart_items(intended),
        actual_block=_render_cart_items(actual),
        missing_block=_render_cart_items(missing),
        stage_block=_render_stage_envelope(stage_env),
        tool_calls_block=tool_calls_block,
        submit_order_t=submit_t,
        last_customer_speech_t=last_customer_t,
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


_VALID_INTENTS = (
    "premature_submit", "compound_line", "clarify_abandoned", "other",
)


def _normalize_items(
    judge_items: list[dict], missing: list[dict],
) -> list[dict]:
    missing_ids = {m.get("item_id") for m in missing if m.get("item_id")}
    by_id: dict[str, dict] = {}
    for it in (judge_items or []):
        mid = _coerce_id(it.get("missing_item_id"), missing_ids)
        if not mid or mid in by_id:
            continue
        intent = it.get("intent")
        if intent not in _VALID_INTENTS:
            intent = "other"
        by_id[mid] = {
            "missing_item_id": mid,
            "missing_options": next(
                (m.get("options") for m in missing if m.get("item_id") == mid),
                None,
            ),
            "intent": intent,
            "rationale": (it.get("rationale") or "")[:400],
            "evidence_quote": it.get("evidence_quote") or None,
            "turn_t": it.get("turn_t"),
        }
    out = []
    for m in missing:
        mid = m.get("item_id")
        if mid in by_id:
            out.append(by_id[mid])
        else:
            out.append({
                "missing_item_id": mid,
                "missing_options": m.get("options"),
                "intent": "other",
                "rationale": "judge omitted this missing item from its response",
                "evidence_quote": None,
                "turn_t": None,
            })
    return out


def run_audit(
    record_dir: Path, *,
    force: bool = False,
    model: str = "gpt-5-mini",
) -> dict:
    score_path = record_dir / "score.json"
    out: dict = {
        "record_dir": str(record_dir),
        "trigger_tag": TRIGGER_TAG,
        "judged": False,
        "skip_reason": None,
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
    if not force and TRIGGER_TAG not in tags:
        out["skip_reason"] = f"tag {TRIGGER_TAG!r} not on this run"
        return out

    diff = score.get("diff") or {}
    missing_all = list(diff.get("missing") or [])
    # Drop missing entries that the scorer already paired into a
    # wrong-option-or-variant pair — those are the SAME item (same
    # base_id) with different options, so the "missing" entry is really
    # an option-level diff, not an item the agent dropped. Same for
    # wrong-item pairs (different items the LLM judged instead_of).
    # Auditing the residuals only avoids classifying the same failure
    # twice under different lenses.
    paired_keys: set[tuple] = set()
    def _key(it: dict) -> tuple:
        opts = it.get("options")
        if isinstance(opts, dict):
            opts_t = tuple(sorted(
                (k, tuple(v) if isinstance(v, list) else v)
                for k, v in opts.items()
            ))
        else:
            opts_t = ()
        return (it.get("item_id"), opts_t)
    for w in (diff.get("wrong_option_or_variant") or []):
        m = w.get("missing")
        if m: paired_keys.add(_key(m))
    for w in (diff.get("wrong_item") or []):
        m = w.get("missing")
        if m: paired_keys.add(_key(m))
    missing = [m for m in missing_all if _key(m) not in paired_keys]
    out["n_missing_after_pair_filter"] = len(missing)
    out["n_missing_pre_pair_filter"] = len(missing_all)
    if not missing:
        out["skip_reason"] = (
            "all missing entries already accounted for by wrong-option-or-variant "
            "or wrong-item pairing; nothing to classify"
        )
        return out

    tl = build_viz(record_dir)
    events = tl.get("events") or []
    submit_t, submit_args = _find_submit_order(events)
    last_customer_t = _last_customer_speech_t(events)
    out["submit_order_t"] = submit_t
    out["submit_order_args"] = submit_args
    out["last_customer_speech_t"] = last_customer_t

    transcript = _render_transcript(record_dir)
    stage_env = _load_stage_envelope(out["task_id"])
    tool_calls_block = _render_tool_calls(events)

    intended = list(score.get("expected_cart") or [])
    actual = list(score.get("actual_cart") or [])

    judge_out = call_judge(
        intended, actual, missing, stage_env, transcript,
        tool_calls_block, submit_t, last_customer_t, model=model,
    )
    out["items"] = _normalize_items(judge_out.get("items") or [], missing)
    out["judged"] = True
    out["model"] = model
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("record_dir", type=Path)
    ap.add_argument("--force", action="store_true",
                    help="run even if missing-item isn't on this run")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--out", type=Path,
                    help="output path (default: <record_dir>/"
                         "missing_item_intent_audit.json)")
    args = ap.parse_args()

    out = run_audit(args.record_dir, force=args.force, model=args.model)
    out_path = args.out or (args.record_dir / "missing_item_intent_audit.json")
    out_path.write_text(json.dumps(out, indent=2))
    if not out.get("judged"):
        print(f"[missing_item_intent_audit] {args.record_dir.name}: skipped "
              f"({out.get('skip_reason')})")
    else:
        intents = ", ".join(
            f"{it['missing_item_id']}={it['intent']}" for it in out.get("items") or []
        ) or "(no missing)"
        print(f"[missing_item_intent_audit] {args.record_dir.name}: "
              f"intents: {intents} → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
