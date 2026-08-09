"""Post-eval audit for ``wrong-option-or-variant`` runs: per pair, did
the agent verbally NAME the wrong option/variant out loud?

Same shape as the read-back flag in ``preemptive_tool_audit.json`` but
scoped to base-id-matched pairs where the agent added the right item
with the wrong option (e.g. customer asked for Cheddar, agent says
"got it, Pepper Jack added") or wrong variant (e.g. "small" vs
"large" suffix on the same base id).

Distinction:
  - read_back=true  — agent's speech contains the wrong option/variant
                      name (customer had a chance to catch the mismatch)
  - read_back=false — agent silently set the option/variant; the
                      customer never heard which value the agent picked

Writes ``wrong_option_audit.json`` next to ``score.json``. Schema:
  {
    "judged": bool,
    "model": str | null,
    "skip_reason": str | null,
    "task_id": str,
    "variant": str,
    "tags": [...],
    "pairs": [
      {
        "base_item_id": str,
        "kind": "option" | "variant",
        "expected_options": dict | null,
        "actual_options": dict | null,
        "expected_variant_suffix": str | null,
        "actual_variant_suffix": str | null,
        "read_back": bool,
        "read_back_quote": str | null,
        "rationale": str
      }
    ]
  }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from clip_eval.scoring.cart_error_audit.preemptive_tool_audit import (
    _render_transcript,
    _render_stage_envelope,
    _load_stage_envelope,
)

TRIGGER_TAG = "wrong-option-or-variant"


def _render_pair(pair: dict, idx: int) -> str:
    """Render one wrong-option-or-variant pair as a block for the LLM."""
    m = pair.get("missing") or {}
    e = pair.get("extra") or {}
    kind = pair.get("kind") or "?"
    base = m.get("item_id") if kind == "option" else (
        m.get("item_id", "").rsplit("-", 1)[0]
    )
    lines = [f"  [{idx}] base={base!r}  kind={kind}"]
    if kind == "option":
        lines.append(f"    expected_options: {m.get('options')}")
        lines.append(f"    actual_options:   {e.get('options')}")
    else:
        lines.append(f"    expected_item_id (variant): {m.get('item_id')!r}")
        lines.append(f"    actual_item_id   (variant): {e.get('item_id')!r}")
    return "\n".join(lines)


JUDGE_SYSTEM = """\
You are auditing a voice-AI ordering agent. The agent added the \
CORRECT base item but with the WRONG OPTION or WRONG VARIANT. For \
each wrong-option-or-variant pair, decide whether the agent VERBALLY \
NAMED the wrong option/variant aloud.

"Read back" means the agent's speech contains the spoken name (or an \
obvious synonym / menu phrasing) of the WRONG option/variant value \
that ended up in the cart — e.g. "Got it, with Pepper Jack" when the \
customer wanted Cheddar; "OK, large Mushroom Swiss" when the customer \
wanted small. Generic acknowledgments like "Got it", "OK", "added" do \
NOT count — the SPECIFIC wrong value must appear in the agent's speech.

For each pair, output:
  - read_back: true | false
  - read_back_quote: the verbatim agent line that names the wrong \
value (null when read_back is false)
  - rationale: one sentence citing the transcript evidence

Output strict JSON:
{
  "pairs": [
    {
      "pair_index": <int from the [N] prefix>,
      "read_back": true | false,
      "read_back_quote": "<verbatim quote, or null>",
      "rationale": "<one sentence>"
    },
    ...
  ]
}

Include exactly one entry per pair, in input order.
"""


JUDGE_USER_TMPL = """\
WRONG-OPTION-OR-VARIANT PAIRS (audit each)
{pairs_block}

STAGE FLOW
{stage_block}

TRANSCRIPT
{transcript}
"""


def call_judge(
    pairs: list[dict], stage_env: dict | None, transcript: str,
    *, model: str = "gpt-5-mini",
) -> dict:
    from openai import OpenAI
    client = OpenAI()
    user_msg = JUDGE_USER_TMPL.format(
        pairs_block="\n".join(_render_pair(p, i) for i, p in enumerate(pairs)),
        stage_block=_render_stage_envelope(stage_env),
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


def _normalize_pairs(
    judge_pairs: list[dict], score_pairs: list[dict],
) -> list[dict]:
    by_idx: dict[int, dict] = {}
    for jp in (judge_pairs or []):
        try:
            i = int(jp.get("pair_index"))
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(score_pairs) and i not in by_idx:
            by_idx[i] = jp
    out = []
    for i, sp in enumerate(score_pairs):
        jp = by_idx.get(i, {})
        m = sp.get("missing") or {}
        e = sp.get("extra") or {}
        kind = sp.get("kind") or "?"
        read_back = bool(jp.get("read_back"))
        rb_quote = jp.get("read_back_quote") or None
        if not read_back:
            rb_quote = None
        out.append({
            "base_item_id": (
                m.get("item_id") if kind == "option"
                else (m.get("item_id", "").rsplit("-", 1)[0] or None)
            ),
            "kind": kind,
            "expected_options": m.get("options"),
            "actual_options": e.get("options"),
            "expected_variant_suffix": (
                m.get("item_id") if kind == "variant" else None
            ),
            "actual_variant_suffix": (
                e.get("item_id") if kind == "variant" else None
            ),
            "read_back": read_back,
            "read_back_quote": rb_quote,
            "rationale": (jp.get("rationale") or "")[:400],
        })
    return out


def run_audit(
    record_dir: Path, *, force: bool = False, model: str = "gpt-5-mini",
) -> dict:
    score_path = record_dir / "score.json"
    out: dict = {
        "record_dir": str(record_dir),
        "trigger_tag": TRIGGER_TAG,
        "judged": False,
        "skip_reason": None,
        "pairs": [],
        "model": None,
    }
    if not score_path.exists():
        out["skip_reason"] = "no score.json"
        return out
    score = json.loads(score_path.read_text())
    tags = list(score.get("tags") or [])
    out["tags"] = tags
    out["task_id"] = score.get("task_id") or record_dir.parent.name
    out["variant"] = score.get("variant") or "simple"
    if not force and TRIGGER_TAG not in tags:
        out["skip_reason"] = f"tag {TRIGGER_TAG!r} not on this run"
        return out

    diff = score.get("diff") or {}
    pairs = list(diff.get("wrong_option_or_variant") or [])
    if not pairs:
        out["skip_reason"] = "no wrong_option_or_variant pairs (tag stale?)"
        return out

    transcript = _render_transcript(record_dir)
    stage_env = _load_stage_envelope(out["task_id"])

    judge_out = call_judge(pairs, stage_env, transcript, model=model)
    out["pairs"] = _normalize_pairs(judge_out.get("pairs") or [], pairs)
    out["judged"] = True
    out["model"] = model
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("record_dir", type=Path)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    out = run_audit(args.record_dir, force=args.force, model=args.model)
    out_path = args.out or (args.record_dir / "wrong_option_audit.json")
    out_path.write_text(json.dumps(out, indent=2))
    if not out.get("judged"):
        print(f"[wrong_option_audit] {args.record_dir.name}: skipped "
              f"({out.get('skip_reason')})")
    else:
        rb = ", ".join(
            f"{p['base_item_id']}={'read-back' if p['read_back'] else 'silent'}"
            for p in out.get("pairs") or []
        ) or "(no pairs)"
        print(f"[wrong_option_audit] {args.record_dir.name}: {rb} → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
