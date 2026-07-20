"""Crash-safe result writing.

Colab sessions die without warning. Pushing a thousand traces through the
relation model will outlive at least one of them. These helpers mean a dead
session costs you the item in flight and nothing else: rerun the same cell and
it picks up where it stopped.

Use it for anything that loops over items and takes more than a few minutes.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one result and force it to disk before returning."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read results back, skipping any half-written final line."""
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue      # a killed session truncated this one


def done_ids(path: str | Path, key: str = "id") -> set[str]:
    return {str(r[key]) for r in read_jsonl(path) if key in r}


def resumable(
    items: Iterable[Any],
    out_path: str | Path,
    work_fn: Callable[[Any], dict[str, Any]],
    id_fn: Callable[[Any], str] = lambda it: str(it["id"]),
    on_error: Callable[[Any, Exception], dict[str, Any]] | None = None,
    progress: bool = True,
) -> int:
    """Run work_fn over items, skipping anything already finished.

    Pass on_error when a failure is itself data worth keeping (a trace the
    relation model chokes on is a finding for the characterisation study, not
    a reason to lose the other 900 results).
    """
    items = list(items)
    seen = done_ids(out_path)
    processed = 0

    iterator: Iterable[Any] = items
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(items, desc=Path(out_path).name)
        except ImportError:
            pass

    for item in iterator:
        item_id = id_fn(item)
        if item_id in seen:
            continue
        try:
            record = work_fn(item)
        except Exception as exc:  # noqa: BLE001
            if on_error is None:
                raise
            record = on_error(item, exc)
        record.setdefault("id", item_id)
        append_jsonl(out_path, record)
        seen.add(item_id)
        processed += 1

    return processed
