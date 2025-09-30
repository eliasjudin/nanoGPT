#!/usr/bin/env python3
"""Utility to compare baseline and Stiefel-aware training configs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VARIANTS: Dict[str, str] = {
    "baseline": "config/train_shakespeare_char.py",
    "stiefel": "config/train_shakespeare_char_stiefel.py",
    "modular": "config/train_shakespeare_char_modular.py",
}


@dataclass
class RunSummary:
    label: str
    config: Path
    out_dir: Path
    command: List[str]
    return_code: Optional[int] = None
    duration_sec: Optional[float] = None
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    train_curve: List[List[float]] = field(default_factory=list)
    eval_curve: List[List[float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:  # keep JSON friendly
        return {
            "label": self.label,
            "config": str(self.config),
            "out_dir": str(self.out_dir),
            "command": self.command,
            "return_code": self.return_code,
            "duration_sec": self.duration_sec,
            "metrics": self.metrics,
            "train_curve": self.train_curve,
            "eval_curve": self.eval_curve,
        }


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        metavar="LABEL=CONFIG",
        help="Override or add a variant mapping label to config path.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="FLAG=VALUE",
        help="Shared flag overrides applied to every run (e.g. --device=cpu).",
    )
    parser.add_argument(
        "--variant-override",
        action="append",
        default=[],
        metavar="LABEL:FLAG=VALUE",
        help="Variant specific override, may be repeated.",
    )
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Raw CLI argument forwarded to train.py for all runs.",
    )
    parser.add_argument(
        "--out-root",
        default=REPO_ROOT / "experiments" / "out" / "stiefel",
        type=Path,
        help="Directory where run subfolders are created.",
    )
    parser.add_argument(
        "--tag",
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Run tag used in generated directory names.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to launch train.py with.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs when their output directory already exists.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Create a PNG plot summarising loss curves when matplotlib is available.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        help="Optional explicit summary path; defaults to <out-root>/<tag>/summary.json.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Terminate runs that exceed this timeout in seconds.",
    )
    return parser.parse_args()


def parse_flag_dict(pairs: Iterable[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"Override must look like flag=value, got: {item}")
        flag, value = item.split("=", 1)
        if not flag:
            raise ValueError(f"Flag missing in override: {item}")
        result[flag] = value
    return result


def parse_variant_overrides(entries: Iterable[str]) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    for entry in entries:
        if ":" not in entry:
            raise ValueError(f"Variant override must look like label:flag=value, got: {entry}")
        label, payload = entry.split(":", 1)
        if "=" not in payload:
            raise ValueError(f"Variant override missing '=', got: {entry}")
        flag, value = payload.split("=", 1)
        result.setdefault(label, {})[flag] = value
    return result


def resolve_variants(variant_args: Optional[Sequence[str]]) -> Dict[str, str]:
    variants = dict(DEFAULT_VARIANTS)
    if not variant_args:
        return variants
    for item in variant_args:
        if "=" not in item:
            raise ValueError(f"Variant must look like label=config.py, got: {item}")
        label, config = item.split("=", 1)
        variants[label] = config
    return variants


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _normalize_value(value: object) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def build_run_cmd(
    python_bin: str,
    config_path: Path,
    common: Dict[str, str],
    extra: Dict[str, str],
    extra_args: Sequence[str],
) -> List[str]:
    cmd = [python_bin, "train.py", str(config_path)]
    for flag, value in {**common, **extra}.items():
        if value is None:
            continue
        cmd.append(f"{flag}={_normalize_value(value)}")
    if extra_args:
        for item in extra_args:
            if "=" in item:
                flag, value = item.split("=", 1)
                cmd.append(f"{flag}={value}")
            else:
                cmd.append(item)
    return cmd


def launch_run(summary: RunSummary, dry_run: bool, timeout: Optional[int]) -> RunSummary:
    ensure_dir(summary.out_dir)
    log_path = summary.out_dir / "train.log"
    env = os.environ.copy()
    # direct stdout/stderr to log to keep console clean
    if dry_run:
        print("DRY-RUN:", " ".join(summary.command))
        summary.return_code = 0
        summary.duration_sec = 0.0
        return summary
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            summary.command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=timeout,
            check=False,
        )
    summary.return_code = proc.returncode
    summary.duration_sec = time.time() - start
    return summary


def parse_metrics(out_dir: Path) -> Dict[str, object]:
    metrics_path = out_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return {}
    train_curve: List[List[float]] = []
    eval_curve: List[List[float]] = []
    best_val = None
    best_eval_iter = None
    last_train_loss = None
    try:
        with open(metrics_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = record.get("event")
                if event == "train_iter":
                    iter_idx = record.get("iter")
                    loss = record.get("loss")
                    if isinstance(iter_idx, (int, float)) and isinstance(loss, (int, float)):
                        train_curve.append([float(iter_idx), float(loss)])
                        last_train_loss = float(loss)
                elif event == "eval":
                    iter_idx = record.get("iter")
                    val_loss = record.get("val_loss") or record.get("loss")
                    if isinstance(iter_idx, (int, float)) and isinstance(val_loss, (int, float)):
                        val_loss = float(val_loss)
                        eval_curve.append([float(iter_idx), val_loss])
                        if best_val is None or val_loss < best_val:
                            best_val = val_loss
                            best_eval_iter = float(iter_idx)
    except OSError:
        return {}
    summary: Dict[str, Optional[float]] = {
        "last_train_loss": last_train_loss,
        "best_val_loss": best_val,
        "best_val_iter": best_eval_iter,
    }
    return {
        "summary": summary,
        "train_curve": train_curve,
        "eval_curve": eval_curve,
    }


def maybe_plot(runs: Sequence[RunSummary], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        print("matplotlib not installed; skipping plot generation.")
        return

    if not runs:
        print("No runs to plot.")
        return

    ensure_dir(out_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    for run in runs:
        if not run.eval_curve and not run.train_curve:
            continue
        if run.eval_curve:
            xs, ys = zip(*run.eval_curve)
            ax.plot(xs, ys, label=f"{run.label} eval", linewidth=2)
        if run.train_curve:
            xs, ys = zip(*run.train_curve)
            ax.plot(xs, ys, label=f"{run.label} train", linestyle="--", alpha=0.6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Stiefel benchmark curves")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend()
    plot_path = out_dir / "stiefel_benchmark.png"
    fig.tight_layout()
    fig.savefig(plot_path)
    print(f"Wrote plot to {plot_path}")


def summarise_runs(runs: Sequence[RunSummary], summary_path: Path) -> None:
    ensure_dir(summary_path.parent)
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "runs": [run.to_dict() for run in runs],
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Wrote summary to {summary_path}")


def main() -> None:
    args = parse_cli()
    variants = resolve_variants(args.variant)
    common = parse_flag_dict(args.override)
    variant_specific = parse_variant_overrides(args.variant_override)

    run_root = args.out_root / args.tag
    ensure_dir(run_root)

    runs: List[RunSummary] = []
    for label, config_path in variants.items():
        config = (REPO_ROOT / config_path).resolve()
        if not config.exists():
            print(f"Skipping {label}: config not found at {config}")
            continue
        out_dir = run_root / label
        if args.skip_existing and out_dir.exists():
            print(f"Skipping {label}: output directory already present")
            continue
        overrides = variant_specific.get(label, {})
        command = build_run_cmd(
            args.python,
            config,
            common,
            overrides,
            args.extra_arg,
        )
        run = RunSummary(
            label=label,
            config=config,
            out_dir=out_dir,
            command=command,
        )
        run = launch_run(run, args.dry_run, args.timeout)
        metrics_payload = parse_metrics(run.out_dir)
        if metrics_payload:
            run.metrics = metrics_payload.get("summary", {})
            run.train_curve = metrics_payload.get("train_curve", [])
            run.eval_curve = metrics_payload.get("eval_curve", [])
        runs.append(run)

    summary_path = args.summary or (run_root / "summary.json")
    summarise_runs(runs, summary_path)
    if args.plot:
        maybe_plot(runs, run_root)


if __name__ == "__main__":
    main()
