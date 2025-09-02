from model import GPTConfig

# start from the stock tiny config and only flip the new bits
train_config = dict(
    out_dir = 'out-shakespeare-stiefel',
    eval_interval = 250,
    eval_iters = 200,
    log_interval = 10,
    block_size = 256,
    batch_size = 64,
    n_layer = 6,
    n_head = 6,
    n_embd = 384,
    max_iters = 5000,
    lr_decay_iters = 5000,
    dropout = 0.0,
)

model_config = GPTConfig(
    block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.0, bias=True,
    # NEW
    use_stiefel_attn=True,
    stiefel_q=True, stiefel_k=True, stiefel_o=False,
    stiefel_mode="columns",
    lora_rank=0, lora_alpha=0.0, lora_dropout=0.0,
    stiefel_lora_B=False,
)

