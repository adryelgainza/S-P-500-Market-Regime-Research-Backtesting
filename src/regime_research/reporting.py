"""Notebook-facing tables and figures with persisted chart data and assumptions."""

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class StudyResult:
    summary: dict
    metrics: pd.DataFrame
    inference: pd.DataFrame
    predictive: pd.DataFrame
    figures: dict
    directory: Path


def render_figures(daily, metrics, output, caption):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output/"wealth_and_exposure.csv", index=False, lineterminator="\n")
    metrics.to_csv(output/"metric_comparison.csv", index=False, lineterminator="\n")
    with plt.rc_context({"font.family": "serif", "axes.spines.top": False, "axes.spines.right": False}):
        figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for name, group in daily.groupby("Portfolio", sort=False):
            axes[0].plot(pd.to_datetime(group.Return_End_Time), group.Net_Equity, label=name, linewidth=1.1)
            axes[1].plot(pd.to_datetime(group.Return_End_Time), group.Realized_Weight, linewidth=.7)
        axes[0].set_ylabel("Net wealth per initial dollar")
        axes[0].legend(fontsize=8)
        axes[1].set_ylabel("Equity exposure")
        axes[1].set_xlabel("Return-end date")
        axes[1].set_ylim(-.03, 1.03)
        for axis in axes:
            axis.grid(alpha=.2)
        figure.suptitle(caption, fontsize=10)
        figure.tight_layout()
        path = output/"wealth_and_exposure.png"
        figure.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(figure)
    return {"wealth_and_exposure": path}


def load_study(directory):
    directory = Path(directory)
    return StudyResult(json.loads((directory/"summary.json").read_text(encoding="utf-8")),
        pd.read_csv(directory/"metrics.csv"), pd.read_csv(directory/"inference.csv"),
        pd.read_csv(directory/"predictive.csv"),
        {"wealth_and_exposure": directory/"figures/wealth_and_exposure.png"}, directory)
