"""
Compatibility shim: replaces torch_scatter with pure-PyTorch implementations.
Import this module BEFORE any other project imports to monkey-patch torch_scatter.
"""
import sys
import os
import types

# Ensure project root is in sys.path so `from utils import ...` works
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import torch


def scatter(src, index, dim=-1, out=None, dim_size=None, fill_value=0, reduce="sum"):
    """Pure PyTorch scatter implementation compatible with torch_scatter API."""
    if dim < 0:
        dim = src.dim() + dim
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

    # Expand index to match src shape
    idx = index
    if idx.dim() < src.dim():
        for _ in range(src.dim() - idx.dim()):
            idx = idx.unsqueeze(-1)
        idx = idx.expand_as(src)

    size = list(src.size())
    size[dim] = dim_size

    if reduce == "sum" or reduce == "add":
        if out is None:
            out = torch.zeros(size, dtype=src.dtype, device=src.device)
        out.scatter_add_(dim, idx, src)
        return out
    elif reduce == "mean":
        out_sum = torch.zeros(size, dtype=src.dtype, device=src.device)
        out_sum.scatter_add_(dim, idx, src)
        count = torch.zeros(size, dtype=src.dtype, device=src.device)
        ones = torch.ones_like(src)
        count.scatter_add_(dim, idx, ones)
        count = count.clamp(min=1)
        return out_sum / count
    elif reduce == "max":
        out = torch.full(size, fill_value=float('-inf'), dtype=src.dtype, device=src.device)
        out.scatter_reduce_(dim, idx, src, reduce="amax", include_self=False)
        # Replace -inf with fill_value where no scatter happened
        mask = out == float('-inf')
        out[mask] = fill_value
        if reduce == "max":
            argmax = torch.full(size, fill_value=-1, dtype=torch.long, device=src.device)
            return out, argmax
        return out
    elif reduce == "min":
        out = torch.full(size, fill_value=float('inf'), dtype=src.dtype, device=src.device)
        out.scatter_reduce_(dim, idx, src, reduce="amin", include_self=False)
        mask = out == float('inf')
        out[mask] = fill_value
        return out
    else:
        raise ValueError(f"Unknown reduce: {reduce}")


def scatter_add(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    return scatter(src, index, dim=dim, out=out, dim_size=dim_size, fill_value=fill_value, reduce="sum")


def scatter_sum(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    return scatter(src, index, dim=dim, out=out, dim_size=dim_size, fill_value=fill_value, reduce="sum")


def scatter_mean(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    return scatter(src, index, dim=dim, out=out, dim_size=dim_size, fill_value=fill_value, reduce="mean")


def scatter_max(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    if dim < 0:
        dim = src.dim() + dim
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

    idx = index
    if idx.dim() < src.dim():
        for _ in range(src.dim() - idx.dim()):
            idx = idx.unsqueeze(-1)
        idx = idx.expand_as(src)

    size = list(src.size())
    size[dim] = dim_size
    result = torch.full(size, fill_value=float('-inf'), dtype=src.dtype, device=src.device)
    result.scatter_reduce_(dim, idx, src, reduce="amax", include_self=False)
    mask = result == float('-inf')
    result[mask] = fill_value
    argmax = torch.full(size, fill_value=0, dtype=torch.long, device=src.device)
    return result, argmax


def scatter_min(src, index, dim=-1, out=None, dim_size=None, fill_value=0):
    if dim < 0:
        dim = src.dim() + dim
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0

    idx = index
    if idx.dim() < src.dim():
        for _ in range(src.dim() - idx.dim()):
            idx = idx.unsqueeze(-1)
        idx = idx.expand_as(src)

    size = list(src.size())
    size[dim] = dim_size
    result = torch.full(size, fill_value=float('inf'), dtype=src.dtype, device=src.device)
    result.scatter_reduce_(dim, idx, src, reduce="amin", include_self=False)
    mask = result == float('inf')
    result[mask] = fill_value
    argmin = torch.full(size, fill_value=0, dtype=torch.long, device=src.device)
    return result, argmin


# Create a fake torch_scatter module
_mod = types.ModuleType("torch_scatter")
_mod.scatter = scatter
_mod.scatter_add = scatter_add
_mod.scatter_sum = scatter_sum
_mod.scatter_mean = scatter_mean
_mod.scatter_max = scatter_max
_mod.scatter_min = scatter_min
_mod.__file__ = __file__
_mod.__path__ = []

# Create utils submodule
_utils_mod = types.ModuleType("torch_scatter.utils")
def broadcast(src, other, dim):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src

_utils_mod.broadcast = broadcast
_mod.utils = _utils_mod

# Create composite submodule
_composite_mod = types.ModuleType("torch_scatter.composite")
_composite_mod.__path__ = []
_mod.composite = _composite_mod

# Register fake modules
sys.modules["torch_scatter"] = _mod
sys.modules["torch_scatter.utils"] = _utils_mod
sys.modules["torch_scatter.composite"] = _composite_mod

print("[scatter_compat] torch_scatter shimmed with pure-PyTorch implementations")

# ============================================================
# torch_sparse shim: provide SparseTensor from torch_geometric
# ============================================================

_sparse_mod = types.ModuleType("torch_sparse")
_sparse_mod.__file__ = __file__
_sparse_mod.__path__ = []

# Try to import SparseTensor from torch_geometric first
try:
    from torch_geometric.typing import SparseTensor as _SparseTensor
    _sparse_mod.SparseTensor = _SparseTensor
except (ImportError, AttributeError):
    # Fallback: create a minimal SparseTensor class
    class SparseTensor:
        """Minimal SparseTensor fallback."""
        def __init__(self, row=None, col=None, value=None, sparse_sizes=None, **kwargs):
            if row is not None and col is not None:
                if value is not None:
                    self._sparse = torch.sparse_coo_tensor(
                        torch.stack([row, col]), value, sparse_sizes
                    ).coalesce()
                else:
                    self._sparse = torch.sparse_coo_tensor(
                        torch.stack([row, col]),
                        torch.ones(row.size(0), device=row.device),
                        sparse_sizes
                    ).coalesce()
            self._row = row
            self._col = col
            self._value = value
            self._sparse_sizes = sparse_sizes

        def to_dense(self):
            return self._sparse.to_dense()

        @staticmethod
        def from_edge_index(edge_index, edge_attr=None, sparse_sizes=None):
            row, col = edge_index[0], edge_index[1]
            return SparseTensor(row=row, col=col, value=edge_attr, sparse_sizes=sparse_sizes)

    _sparse_mod.SparseTensor = SparseTensor

sys.modules["torch_sparse"] = _sparse_mod

print("[scatter_compat] torch_sparse shimmed")
