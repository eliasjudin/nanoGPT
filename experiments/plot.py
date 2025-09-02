import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
EXP_DIR = THIS_DIR
PLOTS_DIR = THIS_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def read_timeseries(path: Path) -> Dict[str, List[float]]:
    data: Dict[str, List[float]] = {}
    if not path.exists():
        return data
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in data:
                    data[k] = []
                if v == '' or v is None:
                    data[k].append(None)
                else:
                    try:
                        data[k].append(float(v))
                    except ValueError:
                        data[k].append(None)
    return data


def read_evals(path: Path) -> Dict[str, List[float]]:
    return read_timeseries(path)


def list_runs() -> List[str]:
    runs = []
    for p in EXP_DIR.glob('timeseries_*.csv'):
        name = p.stem.replace('timeseries_', '')
        runs.append(name)
    return sorted(runs)


def plot_overlay(xys: List[Tuple[List[float], List[float], str]], title: str, xlabel: str, ylabel: str, out_name: str):
    plt.figure(figsize=(8, 5))
    for x, y, label in xys:
        if not x or not y:
            continue
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / out_name, dpi=150)
    plt.close()


def nan_strip(xs: List[float], ys: List[float]) -> Tuple[List[float], List[float]]:
    x2, y2 = [], []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        x2.append(x)
        y2.append(y)
    return x2, y2


def main():
    runs = list_runs()
    if not runs:
        print('No runs found in experiments/. Expected timeseries_*.csv files.')
        return

    # Plot eval val loss overlay
    eval_xys: List[Tuple[List[float], List[float], str]] = []
    for name in runs:
        evals = read_evals(EXP_DIR / f'evals_{name}.csv')
        x, y = nan_strip(evals.get('iter', []), evals.get('val_loss', []))
        eval_xys.append((x, y, name))
    plot_overlay(eval_xys, 'Validation Loss', 'iter', 'val_loss', 'eval_val_loss.png')

    # Timeseries overlays
    def plot_ts(metric: str, ylabel: str, out_name: str):
        xys: List[Tuple[List[float], List[float], str]] = []
        for name in runs:
            ts = read_timeseries(EXP_DIR / f'timeseries_{name}.csv')
            x, y = nan_strip(ts.get('iter', []), ts.get(metric, []))
            xys.append((x, y, name))
        plot_overlay(xys, ylabel, 'iter', ylabel, out_name)

    plot_ts('loss', 'train iter loss', 'timeseries_loss.png')
    plot_ts('attn_var_mean', 'attn_var_mean', 'timeseries_attn_var.png')
    plot_ts('spec_q_mean', 'spec_q_mean', 'timeseries_spec_q.png')
    plot_ts('spec_k_mean', 'spec_k_mean', 'timeseries_spec_k.png')
    plot_ts('ortho_res_mean', 'ortho_res_mean', 'timeseries_ortho_res.png')

    print(f'Plots written to {PLOTS_DIR}')


if __name__ == '__main__':
    main()

