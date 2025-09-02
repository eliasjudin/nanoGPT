import torch


@torch.no_grad()
def stiefel_reorth_(W: torch.Tensor, mode: str = "columns", eps: float = 1e-8):
    """
    In-place re-orthogonalization onto the Stiefel manifold.
    mode="columns":  enforce W^T W = I (requires rows >= cols)
    mode="rows":     enforce W W^T = I (requires cols >= rows)
    """
    if mode not in ("columns", "rows"):
        raise ValueError(f"mode must be 'columns' or 'rows', got {mode}")

    if mode == "columns":
        m, n = W.shape  # (out, in)
        if m < n:
            # fall back to row-orth if tallness violated
            return stiefel_reorth_(W, mode="rows", eps=eps)
        Q, _ = torch.linalg.qr(W, mode='reduced')  # Q: (m, n)
        W.copy_(Q)
        return W
    else:
        # rows-orthonormal: operate on W^T, then transpose back
        QT, _ = torch.linalg.qr(W.t(), mode='reduced')  # QT: (in, out)
        W.copy_(QT.t())
        return W
