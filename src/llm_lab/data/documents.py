from __future__ import annotations

import json
import gzip
from collections.abc import Iterable, Iterator
from pathlib import Path


TEXT_SUFFIXES = {".txt", ".text", ".md"}
JSONL_SUFFIXES = {".jsonl", ".jsonl.gz"}


def expand_input_paths(paths: Iterable[str | Path], *, recursive: bool = False) -> list[Path]:
    expanded: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.iterdir()
            expanded.extend(p for p in iterator if p.is_file() and _is_supported_text_path(p))
        elif path.is_file():
            if not _is_supported_text_path(path):
                raise ValueError(f"unsupported input file suffix: {path}")
            expanded.append(path)
        else:
            raise FileNotFoundError(path)
    return sorted(expanded)


def iter_text_documents(
    paths: Iterable[str | Path],
    *,
    jsonl_field: str = "text",
    recursive: bool = False,
    max_docs: int | None = None,
) -> Iterator[str]:
    emitted = 0
    for path in expand_input_paths(paths, recursive=recursive):
        for text in _iter_path_documents(path, jsonl_field=jsonl_field):
            if text == "":
                continue
            yield text
            emitted += 1
            if max_docs is not None and emitted >= max_docs:
                return


def _iter_path_documents(path: Path, *, jsonl_field: str) -> Iterator[str]:
    if _is_jsonl_path(path):
        file_obj = (
            gzip.open(path, "rt", encoding="utf-8")
            if path.name.endswith(".gz")
            else path.open("r", encoding="utf-8")
        )
        with file_obj as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
                if jsonl_field not in record:
                    raise ValueError(f"missing JSONL field {jsonl_field!r} at {path}:{line_number}")
                value = record[jsonl_field]
                if not isinstance(value, str):
                    raise TypeError(f"JSONL field {jsonl_field!r} must be a string at {path}:{line_number}")
                yield value
    else:
        yield path.read_text(encoding="utf-8")


def _is_supported_text_path(path: Path) -> bool:
    return path.suffix in TEXT_SUFFIXES or _is_jsonl_path(path)


def _is_jsonl_path(path: Path) -> bool:
    return path.suffix in JSONL_SUFFIXES or path.name.endswith(".jsonl.gz")
