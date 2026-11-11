"""Content-addressed per-refit checkpoints with a single atomic completion file."""

from io import BytesIO
import hashlib
import json
from pathlib import Path

import pandas as pd

from .archive import file_hash
from .experiments import atomic_write, canonical_bytes, content_hash


class CheckpointStore:
    def __init__(self, directory):
        self.directory = Path(directory)

    def load(self, key):
        directory = self.directory/content_hash(key)
        receipt_path = directory/"completion.json"
        if not receipt_path.exists():
            return None
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["Key"] != key:
                return None
            if set(receipt["Files"]) != {"scores.parquet", "audit.json"}:
                return None
            contents = {}
            for name, digest in receipt["Files"].items():
                contents[name] = (directory/name).read_bytes()
                if Path(name).name != name or hashlib.sha256(contents[name]).hexdigest() != digest:
                    return None
            scores = pd.read_parquet(BytesIO(contents["scores.parquet"]))
            audit = json.loads(contents["audit.json"])
            return scores, audit
        except (OSError, ValueError, KeyError):
            return None

    def save(self, key, scores, audit):
        directory = self.directory/content_hash(key)
        directory.mkdir(parents=True, exist_ok=True)
        buffer = BytesIO()
        scores.to_parquet(buffer, index=False)
        # Identical keys imply deterministic inputs. Serialize writers through the
        # experiment store lock so a reader never sees a mixed completed pair.
        from .experiments import ExperimentStore
        with ExperimentStore(directory).transaction():
            existing = self.load(key)
            if existing is not None:
                return directory
            (directory/"completion.json").unlink(missing_ok=True)
            atomic_write(directory/"scores.parquet", buffer.getvalue())
            atomic_write(directory/"audit.json", canonical_bytes(audit)+b"\n")
            receipt = {"Key": key, "Files": {name: file_hash(directory/name) for name in ["scores.parquet", "audit.json"]}}
            atomic_write(directory/"completion.json", canonical_bytes(receipt)+b"\n")
        return directory
