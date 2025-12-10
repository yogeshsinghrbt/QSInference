import time
import torch
import triton
import triton.language as tl

from .quant_per_block import per_block_int8
from .quant_sparse_flash_attention import quant_sparse_attention

from typing import Any, List, Literal, Optional, Tuple, Union

def quant_sparse_causal_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    top_k: int,
    smooth_k: bool = True,
) -> torch.Tensor:
    """

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape: ``[batch_size, num_qo_heads, qo_len, head_dim]``.
  

    k : torch.Tensor
        The key tensor. Shape: ``[batch_size, num_kv_heads, kv_len, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape: ``[batch_size, num_kv_heads, kv_len, head_dim]``.


    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:``[batch_size, num_qo_heads, qo_len, head_dim]``.

    """

    dtype = q.dtype
    assert dtype in [torch.float16, torch.bfloat16, torch.float32], "Input tensors must be in dtype of torch.float16, torch.bfloat16, or torch.float32."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    headdim = q.size(-1)
    assert headdim == 128, "headdim should be 128."

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    seq_dim = 2

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        k -= km
    else:
        km = None

    if dtype == torch.bfloat16 or dtype == torch.float32:
        v = v.to(torch.float16)


    q_int8, q_scale, k_int8, k_scale = per_block_int8(q, k)
    o = quant_sparse_attention(q_int8, q_scale, k_int8, k_scale, v, top_k) 
    
    return o
