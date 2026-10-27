"""Minimal stand-in for torch_scatter (no wheels for torch 2.14): only segment_csr, as used by Sonata/PTv3 grid pooling."""

import torch


def segment_csr(src: torch.Tensor, indptr: torch.Tensor, reduce: str = "sum") -> torch.Tensor:
    return torch.segment_reduce(src, reduce, offsets=indptr.long(), axis=0, unsafe=True)
