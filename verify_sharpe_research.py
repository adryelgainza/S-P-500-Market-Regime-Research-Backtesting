"""Compatibility entry point for verifying the immutable historical study."""

from pathlib import Path
import tempfile

from regime_research.archive import replay_baseline


def main():
    project = Path(__file__).resolve().parent
    scratch = project/"tmp/legacy_verification"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temporary:
        _, result = replay_baseline(project, Path(temporary))
        print((Path(temporary)/"baseline_verification.log").read_text(encoding="utf-8"))
        print(f"Archived layers 1–5 verified. Candidate trials {result['Inference']['Registered_Trials']}")


if __name__ == "__main__":
    main()
