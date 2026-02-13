"""Sanitize raw API samples → safe fixtures for tests (no tokens, truncated).

Usage:
    python scripts/sanitize_sample.py data/api_samples/funnel_*.json tests/fixtures/api/funnel_sample.json
    python scripts/sanitize_sample.py data/api_samples/ads_*.json tests/fixtures/api/ads_sample.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def sanitize(obj, max_items: int = 3, depth: int = 0, max_depth: int = 5):
    """Recursively truncate lists and redact sensitive strings."""
    if depth >= max_depth:
        return "..."
    if isinstance(obj, dict):
        return {k: sanitize(v, max_items, depth + 1, max_depth) for k, v in obj.items()}
    if isinstance(obj, list):
        truncated = [sanitize(item, max_items, depth + 1, max_depth) for item in obj[:max_items]]
        if len(obj) > max_items:
            truncated.append(f"... ({len(obj) - max_items} more items)")
        return truncated
    if isinstance(obj, str) and len(obj) > 200:
        return obj[:50] + "..."
    return obj


def main():
    if len(sys.argv) < 3:
        print("Usage: sanitize_sample.py <input.json> <output.json>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    if not input_path.exists():
        print(f"Input not found: {input_path}")
        sys.exit(1)

    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    safe = sanitize(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(safe, f, ensure_ascii=False, indent=2)

    print(f"Sanitized: {input_path} → {output_path}")


if __name__ == "__main__":
    main()
