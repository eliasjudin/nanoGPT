import torch


@torch.no_grad()
def stiefel_reorth_(W: torch.Tensor, mode: str = "columns"):
    """
    In-place re-orthogonalization onto a Stiefel manifold constraint.

    Modes
    - "columns": enforce W^T W = I (requires rows >= cols)
    - "rows":    enforce W W^T = I (requires cols >= rows)
    - "auto":    choose columns if rows >= cols, else rows

    Notes
    - If columns (resp. rows) are orthonormal, the nonzero singular values are 1,
      so ||W||_2 = 1 and the (nonzero) condition number is 1 along the constrained
      dimension. This bounds amplification through the linear map.
    """
    if mode not in ("columns", "rows", "auto"):
        raise ValueError(f"mode must be 'columns', 'rows', or 'auto', got {mode}")

    m, n = W.shape  # (out, in)
    chosen = mode
    if mode == "auto":
        chosen = "columns" if m >= n else "rows"

    if chosen == "columns":
        if m < n:
            # Not feasible: fall back to row-orthonormal
            chosen = "rows"
        else:
            Q, _ = torch.linalg.qr(W, mode='reduced')  # Q: (m, n)
            W.copy_(Q)
            return W

    # rows-orthonormal: operate on W^T, then transpose back
    QT, _ = torch.linalg.qr(W.t(), mode='reduced')  # QT: (in, out)
    W.copy_(QT.t())
    return W
