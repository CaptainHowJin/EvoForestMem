from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..log.message import Message


def message_bytes(message: "Message") -> bytes:

    return json.dumps(
        message.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def advance_corpus_digest(previous: bytes, message: "Message") -> bytes:

    return hashlib.sha256(previous + b"\0" + message_bytes(message)).digest()


def corpus_fingerprint(messages) -> str:

    digest = bytes(32)
    for message in messages:
        digest = advance_corpus_digest(digest, message)
    return digest.hex()


def atomic_write(path: Path, text: str) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
