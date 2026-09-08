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

"""Post-eval LLM check: did the agent ask about option groups it
shouldn't have?

Triggered only on runs tagged ``missing-answer-missing-item``. Those
are the cases where the customer's clip pool didn't carry an answer
for some question the agent posed, the agent stalled on it, and the
eval timed out missing that item. If the question itself was
unnecessary — the option group had a default or was optional — the
agent went off-policy and is responsible for the miss.

An option group is OFF-POLICY to ask about when:
  - ``required: false`` (optional — agent shouldn't bother), OR
  - ``required: true`` AND ``default`` is non-null (default fills if the
    customer doesn't specify — agent shouldn't bother).

Groups with ``required: true`` and no default are ON-POLICY: customer
MUST answer, agent must ask.

For each expected_cart item we look up its option_groups from the
complex menu (``menu_sample.agent_menu``), filter to the off-policy
set, render that into a structured block, and ask gpt-5 to decide if
the agent's transcript contains a question about any of them.

Writes ``off_policy_audit.json`` next to ``score.json``.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from clip_eval.pipeline.paths import ARTIFACTS_PATH
from clip_eval.lib.timeline import build as build_viz

TRIGGER_TAG = "missing-answer-missing-item"


@dataclass
class OffPolicyChoice:
    name: str


@dataclass
class OffPolicyGroup:
    name: str
    required: bool
    default: str | None
    multi_select: bool
    reason: str            # 'optional' | 'has_default'
    choices: list[str] = field(default_factory=list)


@dataclass
class OffPolicyItem:
    item_id: str
    item_name: str
    off_policy_groups: list[OffPolicyGroup] = field(default_factory=list)


def _find_item_in_agent_menu(agent_menu: list[dict], item_id: str) -> dict | None:
    """Walk the agent_menu list and return the first item whose
    ``base_id`` equals ``item_id`` (which is the resolved variant id,
    e.g. ``hawaiian-pizza-small``)."""
    for cat in agent_menu or []:
        for it in (cat.get("items") or []):
            if it.get("base_id") == item_id:
                return it
    return None


def build_off_policy_structure(
    expected_cart: list[dict], agent_menu: list[dict],
) -> list[OffPolicyItem]:
    """For each item in the expected_cart, return the off-policy option
    groups (optional or default-backed). Items the menu doesn't carry
    are returned with an empty group list — the audit just skips them.
    """
    out: list[OffPolicyItem] = []
    for entry in expected_cart:
        item_id = entry.get("resolved_id") or entry.get("item_id")
        if not item_id:
            continue
        menu_item = _find_item_in_agent_menu(agent_menu, item_id)
        item_name = (menu_item or {}).get("name") or item_id
        groups = (menu_item or {}).get("option_groups") or []
        off_groups: list[OffPolicyGroup] = []
        for og in groups:
            required = bool(og.get("required"))
            default = og.get("default")
            if required and default in (None, ""):
                # On-policy: agent MUST ask. Skip.
                continue
            reason = "optional" if not required else "has_default"
            off_groups.append(OffPolicyGroup(
                name=og.get("name") or "",
                required=required,
                default=default,
                multi_select=bool(og.get("multi_select")),
                reason=reason,
                choices=[c.get("name") or "" for c in (og.get("choices") or [])],
            ))
        out.append(OffPolicyItem(
            item_id=item_id, item_name=item_name, off_policy_groups=off_groups,
        ))
    return out


def render_off_policy_block(items: list[OffPolicyItem]) -> str:
    """Render the structured off-policy list as a prompt-friendly block."""
    lines: list[str] = []
    for it in items:
        if not it.off_policy_groups:
            lines.append(f'- "{it.item_name}" ({it.item_id}): no off-policy groups')
            continue
        lines.append(f'- "{it.item_name}" ({it.item_id}):')
        for og in it.off_policy_groups:
            tag = (
                f'optional' if og.reason == "optional"
                else f'required, default="{og.default}"'
            )
            ms = ", multi-select" if og.multi_select else ""
            choices = ", ".join(og.choices)
            lines.append(
                f'  · "{og.name}" ({tag}{ms}) — choices: {choices}'
            )
    return "\n".join(lines)


def extract_agent_turns(record_dir: Path) -> list[dict]:
    """Pull agent_speech events out of the timeline, returning a list
    of ``{t, text}`` dicts in chronological order. We skip empty
    transcripts (e.g. cancelled responses)."""
    tl = build_viz(record_dir)
    out: list[dict] = []
    for e in (tl.get("events") or []):
        if e.get("kind") != "agent_speech":
            continue
        text = (e.get("text") or "").strip()
        if not text:
            continue
        out.append({"t": float(e.get("t") or 0.0), "text": text})
    return out


def extract_customer_turns(record_dir: Path) -> list[dict]:
    """Same shape, customer side — included for prompt context so the
    judge can see what the agent was responding to."""
    tl = build_viz(record_dir)
    out: list[dict] = []
    for e in (tl.get("events") or []):
        if e.get("kind") != "customer_speech":
            continue
        text = (e.get("text") or "").strip()
        if not text:
            continue
        out.append({"t": float(e.get("t") or 0.0), "text": text})
    return out


def render_transcript(record_dir: Path) -> str:
    """Interleave agent + customer turns into a single role-tagged
    transcript that mirrors what a human reading the timeline would
    see. Each line is timestamped so the judge can quote ranges."""
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
    out: list[str] = []
    for t, kind, text in rows:
        role = "AGENT" if kind == "agent_speech" else "CUSTOMER"
        out.append(f'[{t:6.1f}s] {role}: {text}')
    return "\n".join(out)


def load_inputs_for_run(record_dir: Path) -> dict:
    """Bundle everything the audit needs from one record dir.

    Returns a dict with:
      - score: parsed score.json (or None)
      - tags: list of strings from score.json
      - task_id: tid from prompt.json
      - variant: 'simple' | 'complex' | None
      - expected_cart: list of dicts
      - agent_menu: list of category dicts (from complex.task.json)
      - off_policy: list[OffPolicyItem]
      - transcript_text: interleaved agent+customer transcript

    Raises FileNotFoundError when score.json is missing — without it
    we can't filter by tag, so the audit can't decide whether to
    run."""
    score_path = record_dir / "score.json"
    if not score_path.exists():
        raise FileNotFoundError(f"no score.json at {score_path}")
    score = json.loads(score_path.read_text())
    prompt = {}
    pp = record_dir / "prompt.json"
    if pp.exists():
        try:
            prompt = json.loads(pp.read_text())
        except Exception:
            prompt = {}
    task_id = prompt.get("task_id") or record_dir.parent.name
    variant = "complex"

    # Find the complex task config for menu lookup. The off-policy
    # check fundamentally depends on having default + required flags,
    # which only the complex menu carries.
    complex_tc = ARTIFACTS_PATH / "order_up_tasks" / f"{task_id}.complex.task.json"
    agent_menu: list[dict] = []
    expected_cart: list[dict] = list(score.get("expected_cart") or [])
    if complex_tc.exists():
        try:
            tc = json.loads(complex_tc.read_text())
            agent_menu = ((tc.get("menu_sample") or {}).get("agent_menu") or [])
            # Use the complex task's expected_cart so option keys match
            # menu group names. score.json's expected_cart is the simple
            # variant's shape on simple runs; on complex runs it equals
            # this anyway.
            expected_cart = list(tc.get("expected_cart") or expected_cart)
        except Exception:
            pass

    off_policy = build_off_policy_structure(expected_cart, agent_menu)
    transcript_text = render_transcript(record_dir)

    return {
        "task_id": task_id,
        "variant": variant,
        "tags": list(score.get("tags") or []),
        "score": score.get("Task-specific score"),
        "expected_cart": expected_cart,
        "agent_menu_present": bool(agent_menu),
        "off_policy": off_policy,
        "transcript_text": transcript_text,
    }


# ── LLM judge ───────────────────────────────────────────────────────


JUDGE_SYSTEM = """\
You are auditing a voice-AI ordering agent. The agent took a customer's
order over the phone. We want to know whether the agent asked the
customer about any option group that the policy says the agent should
NOT ask about — either because the group is optional, or because the
group has a default value that fills automatically when the customer
doesn't specify.

You will receive (1) a list of items the customer ordered, each with
its "off-policy" option groups (the ones the agent should NOT have
asked about), and (2) the role-tagged transcript of the conversation.

For each off-policy group, decide if the agent asked the customer
about it. An "ask" means the agent surfaced the group's name or a
specific choice and invited the customer to pick — e.g. "what crust
would you like?" or "BBQ Sauce or Mustard?". A passing mention while
confirming what the customer ALREADY said is not an ask.

Reply with strict JSON of the form:
{
  "asks": [
    {
      "item_id": "<resolved_id>",
      "group_name": "<exact group name from the input>",
      "agent_quote": "<the relevant agent utterance, verbatim>",
      "turn_t": <number, the [t] timestamp prefix from the transcript line>,
      "rationale": "<one short sentence>"
    },
    ...
  ]
}

Include only confirmed asks. Empty list when the agent stayed on
policy. Quote the agent utterance verbatim — no paraphrasing.
"""


JUDGE_USER_TMPL = """\
EXPECTED ITEMS + OFF-POLICY OPTION GROUPS
{off_policy_block}

TRANSCRIPT
{transcript}
"""


def call_judge(off_policy_block: str, transcript: str,
               model: str = "gpt-5-mini") -> dict:
    """One LLM call. Returns the parsed JSON object."""
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER_TMPL.format(
                off_policy_block=off_policy_block, transcript=transcript,
            )},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def run_audit(record_dir: Path, *, model: str = "gpt-5-mini",
              force: bool = False) -> dict:
    """Full pipeline: load → filter by tag → render → judge → write."""
    inputs = load_inputs_for_run(record_dir)
    out: dict = {
        "task_id": inputs["task_id"],
        "variant": inputs["variant"],
        "tags": inputs["tags"],
        "trigger_tag": TRIGGER_TAG,
        "judged": False,
        "skip_reason": None,
        "off_policy": [asdict(o) for o in inputs["off_policy"]],
        "asks": [],
    }
    if not force and TRIGGER_TAG not in inputs["tags"]:
        out["skip_reason"] = f"tag {TRIGGER_TAG!r} not on this run"
        return out
    if not inputs["agent_menu_present"]:
        out["skip_reason"] = "no complex menu available for this task_id"
        return out
    n_off_policy = sum(len(i.off_policy_groups) for i in inputs["off_policy"])
    if n_off_policy == 0:
        out["skip_reason"] = "no off-policy groups across any expected item"
        return out
    if not inputs["transcript_text"].strip():
        out["skip_reason"] = "empty transcript (no agent/customer speech events)"
        return out

    block = render_off_policy_block(inputs["off_policy"])
    judge_out = call_judge(block, inputs["transcript_text"], model=model)
    out["judged"] = True
    out["model"] = model
    out["asks"] = judge_out.get("asks") or []
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("record_dir", type=Path)
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--force", action="store_true",
                    help="run even if the run is not tagged "
                         f"{TRIGGER_TAG!r} (useful for demos)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the off-policy block + transcript + "
                         "skip decision, but don't call the LLM")
    ap.add_argument("--out", type=Path,
                    help="output path (default: <record_dir>/off_policy_audit.json)")
    args = ap.parse_args()

    if args.dry_run:
        inputs = load_inputs_for_run(args.record_dir)
        print(f"task_id: {inputs['task_id']}  variant: {inputs['variant']}")
        print(f"tags: {inputs['tags']}")
        n_off = sum(len(i.off_policy_groups) for i in inputs["off_policy"])
        print(f"off-policy groups: {n_off}")
        print()
        print("=== OFF-POLICY BLOCK ===")
        print(render_off_policy_block(inputs["off_policy"]))
        print()
        print("=== TRANSCRIPT ===")
        print(inputs["transcript_text"])
        return 0

    out = run_audit(args.record_dir, model=args.model, force=args.force)
    out_path = args.out or (args.record_dir / "off_policy_audit.json")
    out_path.write_text(json.dumps(out, indent=2))
    asks = out.get("asks") or []
    if out.get("judged"):
        print(f"[off_policy_audit] {args.record_dir.name}: "
              f"{len(asks)} off-policy ask(s) → {out_path}")
    else:
        print(f"[off_policy_audit] {args.record_dir.name}: "
              f"skipped ({out.get('skip_reason')}) → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
