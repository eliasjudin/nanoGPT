from model import GPTConfig

train_config = dict(
    out_dir = 'out-shakespeare-lora-stiefel',
    eval_interval = 250,
    eval_iters = 200,
    log_interval = 10,
    block_size = 256,
    batch_size = 64,
    n_layer = 6,
    n_head = 6,
    n_embd = 384,
    max_iters = 3000,
    lr_decay_iters = 3000,
    dropout = 0.0,
)

model_config = GPTConfig(
    block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.0, bias=True,
    use_stiefel_attn=True,  # engages split path
    stiefel_q=True, stiefel_k=True, stiefel_o=False,
    lora_rank=8, lora_alpha=16.0, lora_dropout=0.05,
    stiefel_lora_B=True,     # orthonormalize LoRA's B
)

