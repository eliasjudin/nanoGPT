"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import json

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from layers.stiefel_linear import StiefelLinear
from layers.lora_stiefel import LoRALinear
from manifold.metrics import stiefel_orthogonality_residual

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# Stiefel/LoRA knobs (optional)
use_stiefel_attn = False
stiefel_q = True
stiefel_k = True
stiefel_o = False
stiefel_mode = 'columns'
stiefel_spectral_budget = 0.0
stiefel_budget_power_iters = 2
stiefel_budget_margin = 0.05
log_stiefel_spectral_stats = False
lora_rank = 0
lora_alpha = 0.0
lora_dropout = 0.0
stiefel_lora_B = False
# Logging knobs
log_specnorm = False
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
    metrics_path = os.path.join(out_dir, 'metrics.jsonl')
else:
    metrics_path = None
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
    bias=bias, vocab_size=None, dropout=dropout,
    # new knobs
    use_stiefel_attn=use_stiefel_attn,
    stiefel_q=stiefel_q, stiefel_k=stiefel_k, stiefel_o=stiefel_o,
    stiefel_mode=stiefel_mode,
    lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
    stiefel_lora_B=stiefel_lora_B,
    stiefel_spectral_budget=stiefel_spectral_budget,
    stiefel_budget_power_iters=stiefel_budget_power_iters,
    stiefel_budget_margin=stiefel_budget_margin,
    log_stiefel_spectral_stats=log_stiefel_spectral_stats,
) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
use_grad_scaler = dtype == 'float16' and device_type == 'cuda'
scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# --- Stiefel/LoRA: Riemannian gradient projection and retraction helpers ---
@torch.no_grad()
def _sym_(M: torch.Tensor) -> torch.Tensor:
    # in-place symmetric part; returns the view for chaining
    M.add_(M.t()).mul_(0.5)
    return M

@torch.no_grad()
def stiefel_project_grads_(module: torch.nn.Module):
    """Project Euclidean grads onto Stiefel tangent spaces for relevant params.
    For StiefelLinear.weight (W ∈ R^{out×in}):
      columns-mode: g_R = g - W sym(W^T g)
      rows-mode:    g_R = g - sym(g W^T) W
    For LoRALinear with stiefel_B: apply same projection to B.
    """
    for m in module.modules():
        if isinstance(m, StiefelLinear):
            W = m.weight
            g = W.grad
            if g is None:
                continue
            mode = getattr(m, 'stiefel_mode', 'columns')
            if mode == 'columns':
                # g <- g - W sym(W^T g)
                # tmp = W^T g  (in×in), sym in-place, then W @ tmp
                tmp = W.t().mm(g)
                _sym_(tmp)
                g.sub_(W.mm(tmp))
            else:
                # rows mode: g <- g - sym(g W^T) W
                tmp = g.mm(W.t())
                _sym_(tmp)
                g.sub_(tmp.mm(W))
        elif isinstance(m, LoRALinear) and getattr(m, 'stiefel_B', False):
            B = m.B
            if B is None or B.grad is None:
                continue
            g = B.grad
            mode = getattr(m, 'stiefel_mode', 'columns')
            if mode == 'columns':
                tmp = B.t().mm(g)
                _sym_(tmp)
                g.sub_(B.mm(tmp))
            else:
                tmp = g.mm(B.t())
                _sym_(tmp)
                g.sub_(tmp.mm(B))

@torch.no_grad()
def stiefel_reproject_all_(module: torch.nn.Module):
    """Retract Stiefel/LoRA-B params back to manifold after optimizer step."""
    for m in module.modules():
        if isinstance(m, StiefelLinear):
            m.reproject_()
        elif isinstance(m, LoRALinear) and getattr(m, 'stiefel_B', False):
            m.reproject_()


@torch.no_grad()
def collect_stiefel_stats(module: torch.nn.Module, mode: str = 'columns'):
    """Aggregate scalar diagnostics for Stiefel-aware modules."""

    ortho_vals = []
    attn_vars = []
    spectral_vals = []
    spectral_clips = []

    for submodule in module.modules():
        if isinstance(submodule, StiefelLinear):
            ortho_vals.append(
                stiefel_orthogonality_residual(
                    submodule.weight,
                    mode=getattr(submodule, 'stiefel_mode', mode),
                )
            )
            metrics = getattr(submodule, 'manifold_metrics', None)
            if isinstance(metrics, dict) and metrics:
                spec = metrics.get('spectral_norm')
                if spec is not None and math.isfinite(spec):
                    spectral_vals.append(float(spec))
                clipped = metrics.get('spectral_clipped')
                if clipped is not None and math.isfinite(clipped):
                    spectral_clips.append(float(clipped))
        if submodule.__class__.__name__ == 'CausalSelfAttentionStiefel':
            v = getattr(submodule, 'last_attn_var', None)
            if v is not None and math.isfinite(v):
                attn_vars.append(float(v))

    stats = {
        'ortho_res_mean': float(np.mean(ortho_vals)) if ortho_vals else None,
        'ortho_res_max': float(np.max(ortho_vals)) if ortho_vals else None,
        'attn_var_mean': float(np.mean(attn_vars)) if attn_vars else None,
        'spectral_norm_mean': float(np.mean(spectral_vals)) if spectral_vals else None,
        'spectral_norm_max': float(np.max(spectral_vals)) if spectral_vals else None,
        'spectral_clip_rate': float(np.mean(spectral_clips)) if spectral_clips else None,
    }
    return stats

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# enable optional attention stats logging when available
raw_model = model.module if ddp else model
for m in raw_model.modules():
    if hasattr(m, 'log_stats'):
        m.log_stats = True

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
last_grad_norm = None
last_grad_clipped = False
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        # persist eval metrics
        if metrics_path is not None:
            with open(metrics_path, 'a') as f:
                f.write(json.dumps({
                    'event': 'eval',
                    'iter': int(iter_num),
                    'train_loss': float(losses['train']),
                    'val_loss': float(losses['val']),
                    'lr': float(lr),
                    'mfu_pct': float(running_mfu*100.0),
                }) + "\n")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # Prepare gradients and optionally clip
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
    # Project grads to Stiefel tangent spaces before clipping (Riemannian update)
    stiefel_project_grads_(raw_model)
    last_grad_norm = None
    last_grad_clipped = False
    if grad_clip != 0.0:
        try:
            last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip).item()
            last_grad_clipped = last_grad_norm > grad_clip
        except Exception:
            last_grad_norm = None
            last_grad_clipped = False
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # Retract Stiefel/LoRA parameters back onto the manifold post-step
    stiefel_reproject_all_(raw_model)
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        msg = f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
        stiefel_stats = None
        if use_stiefel_attn:
            stiefel_stats = collect_stiefel_stats(raw_model)
            if stiefel_stats['ortho_res_mean'] is not None:
                msg += f", ortho_res(mean) {stiefel_stats['ortho_res_mean']:.3e}"
            if stiefel_stats['attn_var_mean'] is not None:
                msg += f", attn_var(mean) {stiefel_stats['attn_var_mean']:.3e}"
            if log_stiefel_spectral_stats:
                if stiefel_stats['spectral_norm_mean'] is not None:
                    msg += f", spec_norm(mean) {stiefel_stats['spectral_norm_mean']:.3f}"
                if stiefel_stats['spectral_clip_rate'] is not None:
                    msg += f", spec_clip(rate) {stiefel_stats['spectral_clip_rate']:.2f}"
            if wandb_log:
                import wandb

                payload = {
                    "iter": iter_num,
                    "diag/ortho_res_mean": stiefel_stats['ortho_res_mean'],
                    "diag/ortho_res_max": stiefel_stats['ortho_res_max'],
                    "diag/attn_var_mean": stiefel_stats['attn_var_mean'],
                }
                if log_stiefel_spectral_stats:
                    payload.update(
                        {
                            "diag/spectral_norm_mean": stiefel_stats['spectral_norm_mean'],
                            "diag/spectral_norm_max": stiefel_stats['spectral_norm_max'],
                            "diag/spectral_clip_rate": stiefel_stats['spectral_clip_rate'],
                        }
                    )
                wandb.log(payload)

        print(msg)
        if metrics_path is not None:
            rec = {
                'event': 'train_iter',
                'iter': int(iter_num),
                'loss': float(lossf),
                'dt_ms': float(dt*1000.0),
                'mfu_pct': float(running_mfu*100.0),
                'grad_norm': float(last_grad_norm) if last_grad_norm is not None else None,
                'grad_clipped': bool(last_grad_clipped),
                'lr': float(lr),
            }
            if stiefel_stats is not None:
                rec.update(
                    {
                        'ortho_res_mean': stiefel_stats['ortho_res_mean'],
                        'ortho_res_max': stiefel_stats['ortho_res_max'],
                        'attn_var_mean': stiefel_stats['attn_var_mean'],
                    }
                )
                if log_stiefel_spectral_stats:
                    rec.update(
                        {
                            'spectral_norm_mean': stiefel_stats['spectral_norm_mean'],
                            'spectral_norm_max': stiefel_stats['spectral_norm_max'],
                            'spectral_clip_rate': stiefel_stats['spectral_clip_rate'],
                        }
                    )
            with open(metrics_path, 'a') as f:
                f.write(json.dumps(rec) + "\n")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
