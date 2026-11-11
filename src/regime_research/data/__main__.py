"""Explicit data import/refresh and offline verification commands."""

import argparse
import json
from pathlib import Path

from . import (HistoryRequirements, build_eligible_sample, import_index_snapshot,
               load_dataset, refresh_spy_snapshot)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="datasets", help="Snapshot directory")
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("import-index", help="Preserve the existing index CSV")
    index.add_argument("--csv", default="sp500_ohlcv.csv")
    refresh = commands.add_parser("refresh-spy", help="Download a new SPY/actions/cash snapshot")
    refresh.add_argument("--start")
    refresh.add_argument("--end", help="Exclusive end date")
    refresh.add_argument("--credited-spread-bps", type=float, default=0.0)
    verify = commands.add_parser("verify", help="Verify a saved snapshot entirely offline")
    verify.add_argument("dataset_id")
    eligible = commands.add_parser("eligible", help="Save eligibility and exclusion tables offline")
    eligible.add_argument("dataset_id")
    eligible.add_argument("--output", required=True)
    eligible.add_argument("--training-years", type=int, default=20)
    eligible.add_argument("--warmup-sessions", type=int, default=200)
    eligible.add_argument("--calibration-sessions", type=int, default=252)
    eligible.add_argument("--label-horizon", type=int, default=5)
    args = parser.parse_args(argv)
    if args.command == "import-index":
        dataset = import_index_snapshot(args.csv, root=args.root)
    elif args.command == "refresh-spy":
        dataset = refresh_spy_snapshot(root=args.root, start=args.start, end=args.end,
                                      credited_spread_bps=args.credited_spread_bps)
    else:
        dataset = load_dataset(args.dataset_id, root=args.root)
    if args.command == "eligible":
        sample = build_eligible_sample(dataset, HistoryRequirements(
            training_years=args.training_years, feature_warmup_sessions=args.warmup_sessions,
            calibration_sessions=args.calibration_sessions, label_horizon_sessions=args.label_horizon))
        output = Path(args.output)
        resolved = output.resolve()
        if resolved.is_relative_to(Path(args.root).resolve()):
            parser.error("Eligibility reports must be saved outside the immutable snapshot directory")
        output.mkdir(parents=True, exist_ok=True)
        sample.eligible.to_csv(output / "eligible_decisions.csv", index=False)
        sample.exclusions.to_csv(output / "exclusions.csv", index=False)
        print(f"Eligible decisions: {len(sample.eligible):,}; exclusions: {len(sample.exclusions):,}")
    print(json.dumps({"Dataset_ID": dataset.dataset_id, "Rows": len(dataset.bars),
                      "Start": dataset.bars.Session.iloc[0], "End": dataset.bars.Session.iloc[-1],
                      "Track": dataset.manifest["Metadata"]["Track"],
                      "Quality_Warnings": [{key: value for key, value in item.items() if key != "Dates"}
                                           for item in dataset.quality_report["Warnings"]]}, indent=2))


if __name__ == "__main__":
    main()
