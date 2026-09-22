#!/usr/bin/env python3
"""Ask two or more layad daemons the same question and diff the answers.

The same state and question does not score the same under MLX and under PyTorch, so a
threshold fitted on your laptop is not valid on your homelab box. This measures that gap
rather than leaving you to discover it in production:

    layad serve &                                    # MLX, on the Mac
    ssh homelab layad serve &                        # torch, on the homelab
    python scripts/compare-runtime.py http://127.0.0.1:8918 http://homelab:8918

Exits non-zero if any answer differs by more than --tolerance, so it can gate a rollout.
"""

from __future__ import annotations

import argparse
import json
import sys

import httpx

STATE = (
    "PagerDuty: checkout-api p99 latency 4200ms, error rate 12%, started 6 minutes ago. "
    "Three customers have written in."
)
QUESTIONS = {
    "is_urgent": {
        "type": "noul",
        "instructions": "The message describes an outage that needs attention now.",
    },
    "domain": {
        "type": "choice",
        "instructions": "Which team owns this?",
        "criteria": {"infra": "servers and networking", "billing": "payments", "app": "product"},
    },
    "severity": {
        "type": "score",
        "instructions": "How severe is this?",
        "criteria": ["none", "low", "medium", "high", "critical"],
    },
}


def scalars(answers: dict) -> dict:
    """The one number (or label) each answer turns on, flattened for comparison."""
    out = {}
    for qid, answer in answers.items():
        key = {"noul": "noul", "choice": "choice", "score": "score"}[answer["type"]]
        out[qid] = answer[key]
        if answer["type"] != "noul":
            out[f"{qid}.confidence"] = answer["confidence"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("endpoints", nargs="+", help="layad base URLs to compare")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--state", default=STATE)
    parser.add_argument("--questions", type=argparse.FileType(), help="JSON question pack")
    args = parser.parse_args()

    questions = json.load(args.questions) if args.questions else QUESTIONS
    results = {}
    for endpoint in args.endpoints:
        base = endpoint.rstrip("/")
        body = httpx.post(
            f"{base}/ai/run",
            json={"state": args.state, "questions": questions},
            timeout=180.0,
        )
        body.raise_for_status()
        body = body.json()
        usage = body["usage"]
        results[endpoint] = body
        print(
            f"{endpoint}\n  runtime={usage.get('runtime')} device={usage.get('device')} "
            f"dtype={usage.get('dtype')} revision={(usage.get('revision') or '?')[:12]} "
            f"model={body['model']}"
        )

    print()
    header = ["field", *(f"{e.split('//')[-1]:>22}" for e in args.endpoints)]
    print(" | ".join(header))
    worst = 0.0
    mismatched = []
    reference = scalars(results[args.endpoints[0]]["answers"])
    for field in reference:
        row = [f"{field:<18}"]
        values = [scalars(results[e]["answers"])[field] for e in args.endpoints]
        for value in values:
            row.append(f"{value:>22.4f}" if isinstance(value, float) else f"{value:>22}")
        print(" | ".join(row))
        if all(isinstance(v, float) for v in values):
            spread = max(values) - min(values)
            worst = max(worst, spread)
            if spread > args.tolerance:
                mismatched.append((field, spread))
        elif len(set(values)) > 1:
            mismatched.append((field, float("inf")))

    print(f"\nlargest numeric spread: {worst:.4f} (tolerance {args.tolerance})")
    if mismatched:
        print(
            "\nThese runtimes do not agree. A threshold calibrated against one of them is "
            "not valid against the other:"
        )
        for field, spread in mismatched:
            print(f"  {field}: {'different label' if spread == float('inf') else f'{spread:.4f}'}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
