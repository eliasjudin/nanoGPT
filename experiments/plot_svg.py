import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

THIS_DIR = Path(__file__).resolve().parent
EXP_DIR = THIS_DIR
PLOTS_DIR = THIS_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_as_dict(path: Path) -> Dict[str, List[Optional[float]]]:
    data: Dict[str, List[Optional[float]]] = {}
    if not path.exists():
        return data
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in data:
                    data[k] = []
                if v is None or v == '':
                    data[k].append(None)
                else:
                    try:
                        data[k].append(float(v))
                    except Exception:
                        data[k].append(None)
    return data


def list_runs() -> List[str]:
    return sorted([p.stem.replace('timeseries_', '') for p in EXP_DIR.glob('timeseries_*.csv')])


def nan_strip(xs: List[Optional[float]], ys: List[Optional[float]]) -> Tuple[List[float], List[float]]:
    x2, y2 = [], []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            continue
        x2.append(float(x))
        y2.append(float(y))
    return x2, y2


def scale_points(xs: List[float], ys: List[float], w: int, h: int, margin: int) -> List[Tuple[float, float]]:
    if not xs or not ys:
        return []
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min:
        x_max += 1.0
    if y_max == y_min:
        y_max += 1.0
    pts = []
    for x, y in zip(xs, ys):
        X = margin + (x - x_min) / (x_max - x_min) * (w - 2 * margin)
        Y = h - margin - (y - y_min) / (y_max - y_min) * (h - 2 * margin)
        pts.append((X, Y))
    return pts


def svg_polyline(points: List[Tuple[float, float]], color: str, width: float = 1.5) -> str:
    if not points:
        return ''
    pts = ' '.join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}" points="{pts}" />\n'


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = 'start') -> str:
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-family="Arial, Helvetica, sans-serif">{text}</text>\n'


def write_svg(filename: Path, title: str, xlabel: str, ylabel: str, series: List[Tuple[List[float], List[float], str]], w: int = 800, h: int = 500, margin: int = 60):
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999']
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">\n']
    # border
    svg.append(f'<rect x="1" y="1" width="{w-2}" height="{h-2}" fill="white" stroke="#ddd"/>\n')
    # title
    svg.append(svg_text(w/2, margin/2, title, size=14, anchor='middle'))
    # axes
    svg.append(f'<line x1="{margin}" y1="{h-margin}" x2="{w-margin}" y2="{h-margin}" stroke="#333"/>\n')
    svg.append(f'<line x1="{margin}" y1="{h-margin}" x2="{margin}" y2="{margin}" stroke="#333"/>\n')
    # labels
    svg.append(svg_text(w/2, h-10, xlabel, size=12, anchor='middle'))
    svg.append(svg_text(15, h/2, ylabel, size=12, anchor='middle'))

    # plot lines
    for i, (xs, ys, name) in enumerate(series):
        pts = scale_points(xs, ys, w, h, margin)
        svg.append(svg_polyline(pts, colors[i % len(colors)], width=1.8))
    # legend
    lx, ly = w - margin - 150, margin
    for i, (_, _, name) in enumerate(series):
        color = colors[i % len(colors)]
        svg.append(f'<line x1="{lx}" y1="{ly + i*18}" x2="{lx+20}" y2="{ly + i*18}" stroke="{color}" stroke-width="2"/>\n')
        svg.append(svg_text(lx+25, ly + i*18 + 4, name, size=12, anchor='start'))

    svg.append('</svg>\n')
    filename.write_text(''.join(svg))


def main():
    runs = list_runs()
    if not runs:
        print('No runs found.')
        return

    # Eval val loss overlay
    eval_series = []
    for name in runs:
        path = EXP_DIR / f'evals_{name}.csv'
        data = read_csv_as_dict(path)
        x, y = nan_strip(data.get('iter', []), data.get('val_loss', []))
        eval_series.append((x, y, name))
    write_svg(PLOTS_DIR / 'eval_val_loss.svg', 'Validation Loss', 'iter', 'val_loss', eval_series)

    # Timeseries overlays
    def ts(metric: str, ylabel: str, out_name: str):
        series = []
        for name in runs:
            path = EXP_DIR / f'timeseries_{name}.csv'
            data = read_csv_as_dict(path)
            x, y = nan_strip(data.get('iter', []), data.get(metric, []))
            series.append((x, y, name))
        write_svg(PLOTS_DIR / out_name, ylabel, 'iter', ylabel, series)

    ts('loss', 'train iter loss', 'timeseries_loss.svg')
    ts('attn_var_mean', 'attn_var_mean', 'timeseries_attn_var.svg')
    ts('spec_q_mean', 'spec_q_mean', 'timeseries_spec_q.svg')
    ts('spec_k_mean', 'spec_k_mean', 'timeseries_spec_k.svg')
    ts('ortho_res_mean', 'ortho_res_mean', 'timeseries_ortho_res.svg')

    print(f'Plots written to {PLOTS_DIR}')


if __name__ == '__main__':
    main()

