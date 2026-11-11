"""Frozen Monte Carlo checks for the Layer 3 multiplicity procedure."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy.stats import beta, binom

from .statistics import max_t_adjustment


def _simulate(protocol, scenario, seed):
    rng = np.random.default_rng(seed)
    n, burn, count = protocol["Observations"], protocol["Burn_In"], protocol["Trial_Count"]
    sigma = protocol["Daily_Volatility"]
    shocks = rng.normal(size=(n+burn, count))
    if scenario == "correlated_trials":
        rho = protocol["Cross_Trial_Correlation"]
        shocks = np.sqrt(rho)*rng.normal(size=(n+burn, 1))+np.sqrt(1-rho)*shocks
    values = sigma*shocks
    if scenario == "ar1_normal":
        phi = protocol["AR1_Phi"]
        values[0] = 0
        for index in range(1, len(values)):
            values[index] = phi*values[index-1]+sigma*np.sqrt(1-phi**2)*shocks[index]
    elif scenario == "garch_normal":
        alpha, persistence = protocol["GARCH_Alpha"], protocol["GARCH_Beta"]
        variance = np.full(count, sigma**2)
        omega = sigma**2*(1-alpha-persistence)
        for index in range(1, len(values)):
            variance = omega+alpha*values[index-1]**2+persistence*variance
            values[index] = np.sqrt(variance)*shocks[index]
    values = values[burn:]
    if scenario == "planted_edge":
        values[:, 0] += protocol["Planted_Daily_Edge"]
    return pd.DataFrame(values, columns=[f"trial_{number}" for number in range(count)])


def run_calibration(protocol_path, output):
    protocol_path, output = Path(protocol_path), Path(output)
    if output.exists():
        raise FileExistsError(f"Calibration output already exists at {output}")
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    nulls = protocol["Null_Scenarios"]
    if set(nulls) != {"iid_normal", "ar1_normal", "garch_normal", "correlated_trials"}:
        raise ValueError("Calibration requires the four declared null scenarios")
    replications = protocol["Outer_Replications"]
    tail = protocol["Calibration_Family_Error"]/(2*len(nulls))
    lower_count = int(binom.ppf(tail, replications, protocol["Nominal_Familywise_Alpha"]))
    upper_count = int(binom.ppf(1-tail, replications, protocol["Nominal_Familywise_Alpha"]))
    sources = [Path(__file__), Path(__file__).with_name("statistics.py")]
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    rows = []
    for number, scenario in enumerate([*nulls, "planted_edge"]):
        rejections = 0
        for replication in range(replications):
            seed = protocol["Seed"]+100000*number+replication
            data = _simulate(protocol, scenario, seed)
            result = max_t_adjustment(data, np.zeros(len(data)),
                repetitions=protocol["Bootstrap_Replications"], block_length=protocol["Block_Length"],
                maximum_lag=protocol["HAC_Lag"], seed=seed+1000000)
            p_value = result.MaxT_Adjusted_P.iloc[0] if scenario == "planted_edge" else result.MaxT_Adjusted_P.min()
            rejections += int(p_value <= protocol["Nominal_Familywise_Alpha"])
        rate = rejections/replications
        power_lower = (float(beta.ppf(1-protocol["Power_Lower_Bound_Confidence"], rejections,
                                     replications-rejections+1)) if rejections else 0.0)
        passed = (power_lower > protocol["Minimum_Planted_Power_Lower_Bound"] if scenario == "planted_edge"
                  else lower_count <= rejections <= upper_count)
        rows.append(dict(Scenario=scenario, Replications=replications, Rejections=rejections, Rejection_Rate=rate,
            Monte_Carlo_SE=float(np.sqrt(rate*(1-rate)/replications)),
            Acceptance_Lower_Count=None if scenario == "planted_edge" else lower_count,
            Acceptance_Upper_Count=None if scenario == "planted_edge" else upper_count,
            Power_Lower_Bound=power_lower if scenario == "planted_edge" else None, Passed=bool(passed),
            Seed=protocol["Seed"]+100000*number))
        print(f"Calibration {scenario}: {rejections}/{replications}, passed={passed}", flush=True)
    for path in sources:
        if hashlib.sha256(path.read_bytes()).hexdigest() != source_hashes[path.name]:
            raise RuntimeError("Statistical source changed during calibration")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pending-calibration-", dir=output.parent) as temporary:
        pending = Path(temporary)/output.name
        pending.mkdir()
        (pending/"protocol.json").write_bytes(protocol_bytes)
        pd.DataFrame(rows).to_csv(pending/"null_calibration.csv", index=False)
        manifest = dict(Completed_UTC=datetime.now(timezone.utc).isoformat(), All_Passed=all(row["Passed"] for row in rows),
            Protocol_SHA256=hashlib.sha256(protocol_bytes).hexdigest(), Source_SHA256=source_hashes,
            Files={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in pending.iterdir()})
        (pending/"completion.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        pending.rename(output)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="protocols/layer3_null_calibration.json")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = run_calibration(arguments.protocol, arguments.output)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
