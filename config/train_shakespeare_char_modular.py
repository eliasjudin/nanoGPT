# Tiny Shakespeare with modular Stiefel constraints inspired by modular manifolds.

out_dir = 'out-shakespeare-char-stiefel-modular'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt-stiefel-modular'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

use_stiefel_attn = True
stiefel_mode = 'columns'

# Modular manifold controls
stiefel_spectral_budget = 1.05
stiefel_budget_power_iters = 4
stiefel_budget_margin = 0.05
log_stiefel_spectral_stats = True

# Suggested local CPU run for a quick smoke test:
# device = 'cpu'
# compile = False
# eval_iters = 20
# log_interval = 1
# block_size = 64
# batch_size = 12
# n_layer = 4
# n_head = 4
# n_embd = 128
# max_iters = 2000
# lr_decay_iters = 2000
# dropout = 0.0
