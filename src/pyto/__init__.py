try:
    import torch
    # PyTO assembles its sparse matrices itself and does not want per-tensor invariant checks (they cost time).
    # Saying so explicitly silences torch's "Sparse invariant checks are implicitly disabled" warning.
    torch.sparse.check_sparse_tensor_invariants.disable()
except ImportError:
    pass
