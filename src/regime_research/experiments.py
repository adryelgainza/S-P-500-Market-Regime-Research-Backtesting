"""Hash-linked experiment events with serialized, atomic append transactions."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid


ZERO_HASH = "0"*64
STAGES = {"retrospective", "development", "transfer", "forward"}
TERMINAL_STATES = {"completed", "failed", "abandoned"}


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def content_hash(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path, value):
    """Replace a complete file while retaining the previous version on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(7):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 6:
                    raise
                time.sleep(.05*2**attempt)
    finally:
        Path(temporary).unlink(missing_ok=True)


def publish_directory(source, destination, *, root):
    """Publish a staged directory within its declared root, retrying sharing locks."""
    source, destination, root = Path(source).resolve(), Path(destination).resolve(), Path(root).resolve()
    if not source.is_relative_to(root) or not destination.is_relative_to(root) or destination == root:
        raise ValueError("Directory publication escapes its intended root")
    for attempt in range(7):
        if destination.exists():
            raise FileExistsError(f"A completed directory already exists at {destination}")
        try:
            os.rename(source, destination)
            return destination
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(.05*2**attempt)


class ExperimentLogError(ValueError):
    """The experiment history cannot satisfy its declared integrity contract."""


def _experiment_spec(spec):
    required = {"Hypothesis_ID", "Configuration_SHA256", "Source_Revision", "Dataset_IDs", "Stage", "Seed", "Purpose"}
    if not required.issubset(spec) or any(not spec[key] for key in required-{"Seed"}):
        raise ExperimentLogError("Experiment registration requires complete provenance and a hypothesis")
    if spec["Stage"] not in STAGES or not isinstance(spec["Seed"], int) or isinstance(spec["Seed"], bool):
        raise ExperimentLogError("Invalid experiment stage or seed")
    if not isinstance(spec["Dataset_IDs"], list) or not spec["Dataset_IDs"]:
        raise ExperimentLogError("Experiments require explicit dataset identifiers")


def _apply_event(event, states, event_ids):
    kind, identifier, payload = event["Type"], event["Experiment_ID"], event["Payload"]
    if event["Event_ID"] in event_ids:
        raise ExperimentLogError("Duplicate event identifier")
    if kind in {"experiment_registered", "experiment_imported"}:
        if not identifier or identifier in states:
            raise ExperimentLogError("Experiment identifiers must be unique")
        _experiment_spec(payload["Specification"])
        parent = payload["Specification"].get("Parent_Experiment_ID")
        if parent is not None and parent not in states:
            raise ExperimentLogError("Experiment parent is absent from the known history")
        state = "registered"
        if kind == "experiment_imported":
            state = payload["Imported_State"]
            if state not in TERMINAL_STATES or not payload.get("Import_Basis"):
                raise ExperimentLogError("Imported experiments need a terminal state and import basis")
        states[identifier] = dict(Specification=payload["Specification"], State=state,
                                  Outcome=payload.get("Outcome"), Corrections=[])
    elif kind in {"experiment_started", "experiment_completed", "experiment_failed", "experiment_abandoned"}:
        if identifier not in states:
            raise ExperimentLogError("Experiment transition precedes registration")
        state = states[identifier]["State"]
        destination = kind.removeprefix("experiment_")
        allowed = {"registered"} if destination == "started" else {"started"}
        if destination == "abandoned":
            allowed.add("registered")
        if state not in allowed:
            raise ExperimentLogError(f"Invalid experiment transition {state} to {destination}")
        if destination in {"failed", "abandoned"} and not payload.get("Reason"):
            raise ExperimentLogError("Failed or abandoned experiments need a reason")
        states[identifier]["State"] = destination
        if destination in TERMINAL_STATES:
            states[identifier]["Outcome"] = payload
    elif kind == "experiment_corrected":
        if (identifier not in states or not payload.get("Reason")
                or event_ids.get(payload.get("Corrects_Event_ID")) != identifier):
            raise ExperimentLogError("Corrections require a known experiment, event, and reason")
        states[identifier]["Corrections"].append(payload)
    elif kind in {"family_imported", "protocol_frozen", "forward_decision_recorded", "evaluation_recorded"}:
        if identifier is not None and identifier not in states:
            raise ExperimentLogError("Research event references an unknown experiment")
    else:
        raise ExperimentLogError(f"Unsupported experiment event type {kind}")
    event_ids[event["Event_ID"]] = identifier


class ExperimentStore:
    def __init__(self, directory="experiments"):
        self.directory = Path(directory)
        self.path = self.directory/"events.jsonl"

    @contextmanager
    def transaction(self, timeout=10):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory/".writer.lock").open("a+b") as stream:
            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            deadline = time.monotonic()+timeout
            while True:
                try:
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("The experiment register is busy")
                    time.sleep(.025)
            try:
                yield self
            finally:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _read(self, anchor=None):
        raw = self.path.read_bytes() if self.path.exists() else b""
        if raw and not raw.endswith(b"\n"):
            raise ExperimentLogError("The event log has an incomplete record")
        events, states, identifiers, keys = [], {}, {}, set()
        previous = ZERO_HASH
        try:
            for sequence, line in enumerate(raw.splitlines(), 1):
                event = json.loads(line)
                timestamp = datetime.fromisoformat(event["Recorded_UTC"])
                if timestamp.utcoffset() is None or (events and timestamp < datetime.fromisoformat(events[-1]["Recorded_UTC"])):
                    raise ExperimentLogError("Event recording timestamps must be aware and ordered")
                digest = event["Event_SHA256"]
                body = {key:value for key,value in event.items() if key != "Event_SHA256"}
                if event["Sequence"] != sequence or event["Previous_SHA256"] != previous or content_hash(body) != digest:
                    raise ExperimentLogError("Experiment hash chain or sequence mismatch")
                if event["Idempotency_Key"] in keys:
                    raise ExperimentLogError("Duplicate event idempotency key")
                _apply_event(event, states, identifiers)
                keys.add(event["Idempotency_Key"])
                previous = digest
                events.append(event)
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ExperimentLogError(f"Malformed experiment event: {error}") from error
        if anchor is not None:
            sequence = anchor["Sequence"]
            actual = events[sequence-1]["Event_SHA256"] if 0 < sequence <= len(events) else ZERO_HASH
            if sequence > len(events) or actual != anchor["Event_SHA256"]:
                raise ExperimentLogError("Experiment history differs from the trusted anchor")
        return events, states, raw

    def _append_many(self, specifications, expected_head=None):
        events, states, raw = self._read()
        previous = events[-1]["Event_SHA256"] if events else ZERO_HASH
        if expected_head is not None and previous != expected_head:
            raise ExperimentLogError("Experiment head changed before commit")
        identifiers = {event["Event_ID"]:event["Experiment_ID"] for event in events}
        existing = {event["Idempotency_Key"]:event for event in events}
        additions, returned = [], []
        for spec in specifications:
            key = spec["Idempotency_Key"]
            if key in existing:
                prior = existing[key]
                if any(prior[name] != spec.get(name) for name in ["Type", "Experiment_ID", "Payload"]):
                    raise ExperimentLogError("An idempotency key was reused with different content")
                returned.append(prior)
                continue
            event = dict(Sequence=len(events)+len(additions)+1, Event_ID=spec.get("Event_ID",uuid.uuid4().hex),
                Recorded_UTC=spec.get("Recorded_UTC",utc_now()), Type=spec["Type"],
                Experiment_ID=spec.get("Experiment_ID"), Idempotency_Key=key,
                Previous_SHA256=previous, Payload=spec["Payload"])
            prior_event = additions[-1] if additions else (events[-1] if events else None)
            timestamp = datetime.fromisoformat(event["Recorded_UTC"])
            if timestamp.utcoffset() is None or (prior_event and timestamp < datetime.fromisoformat(prior_event["Recorded_UTC"])):
                raise ExperimentLogError("Event recording timestamps must be aware and ordered")
            _apply_event(event, states, identifiers)
            event["Event_SHA256"] = content_hash(event)
            previous = event["Event_SHA256"]
            additions.append(event)
            existing[key] = event
            returned.append(event)
        if additions:
            atomic_write(self.path, raw+b"".join(canonical_bytes(event)+b"\n" for event in additions))
        return returned

    def append_many(self, specifications, expected_head=None):
        with self.transaction():
            return self._append_many(specifications, expected_head)

    def append(self, kind, experiment_id, payload, *, idempotency_key, expected_head=None):
        return self.append_many([dict(Type=kind, Experiment_ID=experiment_id, Payload=payload,
                                      Idempotency_Key=idempotency_key)], expected_head)[0]

    def verify(self, anchor=None):
        with self.transaction():
            events, states, raw = self._read(anchor)
            return dict(Events=len(events), Experiments=len(states), Sequence=len(events),
                Event_SHA256=events[-1]["Event_SHA256"] if events else ZERO_HASH,
                Log_SHA256=hashlib.sha256(raw).hexdigest(), States=states)

    def save_anchor(self):
        with self.transaction():
            events, _, raw = self._read()
            anchor = dict(Sequence=len(events), Event_SHA256=events[-1]["Event_SHA256"] if events else ZERO_HASH,
                          Log_SHA256=hashlib.sha256(raw).hexdigest(), Created_UTC=utc_now())
            name=f"anchor-{anchor['Sequence']:06d}-{anchor['Event_SHA256']}.json"
            path=self.directory/"anchors"/name
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
            atomic_write(path, canonical_bytes(anchor)+b"\n")
            return anchor

    @contextmanager
    def attempt(self, specification, experiment_id=None):
        """Register before work and retain ordinary failures and interruptions."""
        identifier = experiment_id or uuid.uuid4().hex
        with self.transaction():
            _, states, _ = self._read()
            if identifier in states:
                raise ExperimentLogError("A repeated attempt requires a new experiment ID and the same hypothesis")
            self._append_many([
                {"Type": "experiment_registered", "Experiment_ID": identifier,
                 "Payload": {"Specification": specification}, "Idempotency_Key": f"register-{identifier}"},
                {"Type": "experiment_started", "Experiment_ID": identifier,
                 "Payload": {}, "Idempotency_Key": f"start-{identifier}"}])
        outcome = {"Artifacts": []}
        try:
            yield identifier, outcome
        except BaseException as error:
            self.append("experiment_failed", identifier,
                        {"Reason": f"{type(error).__name__}: {error}", "Artifacts": outcome["Artifacts"]},
                        idempotency_key=f"failed-{identifier}")
            raise
        else:
            self.append("experiment_completed", identifier, outcome, idempotency_key=f"complete-{identifier}")
