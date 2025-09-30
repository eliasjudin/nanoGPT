import torch


@torch.no_grad()
def stiefel_orthogonality_residual(weight: torch.Tensor, mode: str = "columns") -> float:
    """Compute Frobenius-norm residual to Stiefel constraints.

    For columns: ||W^T W - I||_F / sqrt(d)
    For rows:    ||W W^T - I||_F / sqrt(d)

    Returns a Python float for convenient logging.
    """
    if weight.ndim != 2:
        raise ValueError("stiefel_orthogonality_residual expects a 2-D matrix")
    rows, cols = weight.shape
    if mode not in {"columns", "rows", "auto"}:
        raise ValueError("mode must be 'columns', 'rows', or 'auto'")
    chosen = mode
    if mode == "auto":
        chosen = "columns" if rows >= cols else "rows"

    if chosen == "columns":
        gram = weight.t() @ weight
        d = gram.shape[0]
        resid = torch.linalg.norm(gram - torch.eye(d, device=weight.device, dtype=weight.dtype), ord="fro")
        return (resid / torch.sqrt(torch.tensor(float(d), device=weight.device, dtype=weight.dtype))).item()
    else:
        gram = weight @ weight.t()
        d = gram.shape[0]
        resid = torch.linalg.norm(gram - torch.eye(d, device=weight.device, dtype=weight.dtype), ord="fro")
        return (resid / torch.sqrt(torch.tensor(float(d), device=weight.device, dtype=weight.dtype))).item()

