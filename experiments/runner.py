import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parents[1]
TRAIN = str(ROOT / 'train.py')

def run(cmd, cwd=None):
    print('>>', ' '.join(cmd))
    p = subprocess.run(cmd, cwd=cwd or str(ROOT))
    if p.returncode != 0:
        raise SystemExit(p.returncode)

def read_metrics(out_dir) -> List[Dict[str, Any]]:
    p = Path(out_dir) / 'metrics.jsonl'
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def write_csv(path: Path, header: List[str], rows: List[List[Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(','.join(header) + '\n')
        for row in rows:
            f.write(','.join('' if v is None else str(v) for v in row) + '\n')

def summarize(metrics):
    last_eval = None
    for m in metrics[::-1]:
        if m.get('event') == 'eval':
            last_eval = m
            break
    last_iter = None
    for m in metrics[::-1]:
        if m.get('event') == 'train_iter':
            last_iter = m
            break
    return {
        'last_val_loss': last_eval.get('val_loss') if last_eval else None,
        'last_train_loss': last_eval.get('train_loss') if last_eval else (last_iter.get('loss') if last_iter else None),
        'last_ortho_res_mean': last_iter.get('ortho_res_mean') if last_iter else None,
        'last_attn_var_mean': last_iter.get('attn_var_mean') if last_iter else None,
        'last_spec_q_mean': last_iter.get('spec_q_mean') if last_iter else None,
        'last_spec_k_mean': last_iter.get('spec_k_mean') if last_iter else None,
        'last_grad_norm': last_iter.get('grad_norm') if last_iter else None,
        'grad_clipped_any': any(m.get('grad_clipped') for m in metrics if m.get('event')=='train_iter'),
        'iters': sum(1 for m in metrics if m.get('event')=='train_iter'),
        'mean_dt_ms': sum(m.get('dt_ms',0) for m in metrics if m.get('event')=='train_iter')/max(1,sum(1 for m in metrics if m.get('event')=='train_iter')),
    }

def main():
    # quick toggle: set to False for longer CPU/GPU runs
    QUICK = False
    iters = 30 if QUICK else 5000
    lrdec = iters
    common = {
        'dataset':'shakespeare_char','device':'cpu','compile':False,
        'eval_iters':50 if not QUICK else 5,'log_interval':50 if not QUICK else 1,
        'block_size':32,  # keep <= val set length (62)
        'batch_size':32 if not QUICK else 8,
        'n_layer':6 if not QUICK else 2,
        'n_head':6 if not QUICK else 2,
        'n_embd':384 if not QUICK else 64,
        'max_iters':iters,'lr_decay_iters':lrdec,'dropout':0.0,
        'log_specnorm':True,
        'gradient_accumulation_steps': 1 if not QUICK else 1,
    }
    exps = [
        {
            'name': 'baseline',
            'out': 'out-exp-baseline',
            'args': {
                **common,
                'use_stiefel_attn':False,
            },
        },
        {
            'name': 'stiefel_qk',
            'out': 'out-exp-stiefel-qk',
            'args': {
                **common,
                'use_stiefel_attn':True,'stiefel_q':True,'stiefel_k':True,
            },
        },
        {
            'name': 'stiefel_qko',
            'out': 'out-exp-stiefel-qko',
            'args': {
                **common,
                'use_stiefel_attn':True,'stiefel_q':True,'stiefel_k':True,'stiefel_o':True,
            },
        },
        {
            'name': 'lora_r2_B0',
            'out': 'out-exp-lora-r2-b0',
            'args': {
                **common,
                'use_stiefel_attn':True, # engages split path (LoRA on V/O)
                'lora_rank':2,'lora_alpha':4.0,'lora_dropout':0.05,'stiefel_lora_B':False,
            },
        },
        {
            'name': 'lora_r2_B1',
            'out': 'out-exp-lora-r2-b1',
            'args': {
                **common,
                'use_stiefel_attn':True,
                'lora_rank':2,'lora_alpha':4.0,'lora_dropout':0.05,'stiefel_lora_B':True,
            },
        },
    ]
    results = []
    out_dir = ROOT / 'experiments'
    out_dir.mkdir(exist_ok=True)
    for e in exps:
        args = [f"--{k}={v}" for k,v in e['args'].items()]
        args.append(f"--out_dir={e['out']}")
        cmd = [sys.executable, TRAIN] + args
        run(cmd)
        mets = read_metrics(e['out'])
        summ = summarize(mets)
        summ['name'] = e['name']
        results.append(summ)
        # write per-run timeseries CSVs
        iter_rows = [[m.get('iter'), m.get('loss'), m.get('dt_ms'), m.get('mfu_pct'), m.get('ortho_res_mean'), m.get('attn_var_mean'), m.get('spec_q_mean'), m.get('spec_k_mean'), m.get('grad_norm'), m.get('grad_clipped')] for m in mets if m.get('event')=='train_iter']
        eval_rows = [[m.get('iter'), m.get('train_loss'), m.get('val_loss'), m.get('lr'), m.get('mfu_pct')] for m in mets if m.get('event')=='eval']
        write_csv(out_dir / f'timeseries_{e["name"]}.csv', ['iter','loss','dt_ms','mfu_pct','ortho_res_mean','attn_var_mean','spec_q_mean','spec_k_mean','grad_norm','grad_clipped'], iter_rows)
        write_csv(out_dir / f'evals_{e["name"]}.csv', ['iter','train_loss','val_loss','lr','mfu_pct'], eval_rows)

    print('\nExperiment Summary:')
    for r in results:
        print(r)
    # write summary CSV
    header = ['name','last_val_loss','last_train_loss','last_ortho_res_mean','last_attn_var_mean','last_spec_q_mean','last_spec_k_mean','last_grad_norm','grad_clipped_any','iters','mean_dt_ms']
    rows = [[r.get(k) for k in header] for r in results]
    write_csv(out_dir / 'summary.csv', header, rows)

if __name__ == '__main__':
    main()
