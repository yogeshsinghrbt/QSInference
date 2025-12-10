import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _triton_quant_block_sparse_attention(
    q_int8, q_scale, k_int8, k_scale, V,
    seqlens, sm_scale,
    block_index,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    NUM_ROWS, MAX_BLOCKS_PER_ROW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_scale_offset = (off_hz % H) * NUM_ROWS
    k_scale_offset = (off_hz % H) * NUM_ROWS

    Q_scale_ptr = q_scale + q_scale_offset + start_m
    K_scale_ptr = k_scale + k_scale_offset

    q_ptrs = q_int8 + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = k_int8 + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    blocks_ptr = block_index + (off_hz * NUM_ROWS + start_m) * MAX_BLOCKS_PER_ROW

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
 
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q_scale = tl.load(Q_scale_ptr)

    # loop over k, v and update accumulator
    m_mask = offs_m[:, None] < seqlen
    block_count = tl.minimum((start_m + 1) * BLOCK_M // BLOCK_N, MAX_BLOCKS_PER_ROW)

    for sparse_block_idx in range(block_count):
        real_block_idx = tl.load(blocks_ptr + sparse_block_idx)
        start_n = real_block_idx * BLOCK_N
        cols = start_n + offs_n
        # -- load k, v --
        K_scale_temp = K_scale_ptr + real_block_idx 
        k_scale_val = tl.load(K_scale_temp)

        k = tl.load(k_ptrs + cols[None, :] * stride_kn)
        v = tl.load(v_ptrs + cols[:, None] * stride_vn)
        # -- compute qk --
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # if start_n + BLOCK_N < seqlen:
        #     qk = tl.where(m_mask, qk, float("-inf"))
        # else:
        causal_mask = cols[None, :] <= offs_m[:, None]
        qk = tl.where(m_mask & causal_mask, qk, float("-inf"))
        qk += tl.dot(q, k).to(tl.float32) * (q_scale * k_scale_val)

        # -- compute scaling constant --
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc_scale = l_i * 0 + alpha  # workaround some compiler bug
        acc *= acc_scale[:, None]

        acc += tl.dot(p.to(dtype), v)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new

    # write back O
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


def quant_block_sparse_attention(
    q_int8, q_scale, k_int8, k_scale, v, 
    seqlens,          
    block_index,       # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), MAX_BLOCKS_PER_ROW]
    sm_scale,
    block_size_M=64,
    block_size_N=64,
) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q_int8.shape[-1], k_int8.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk == 128
    o = torch.zeros_like(v)
    grid = (triton.cdiv(q_int8.shape[2], block_size_M), q_int8.shape[0] * q_int8.shape[1], 1)

    dtype = tl.bfloat16 if v.dtype == torch.bfloat16 else tl.float16
    _triton_quant_block_sparse_attention[grid]( 
        q_int8, q_scale, k_int8, k_scale, v,
        seqlens, sm_scale,
        block_index,
        o,
        q_int8.stride(0), q_int8.stride(1), q_int8.stride(2), q_int8.stride(3),
        k_int8.stride(0), k_int8.stride(1), k_int8.stride(2), k_int8.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q_int8.shape[0], q_int8.shape[1], q_int8.shape[2],
        block_index.shape[-2], block_index.shape[-1],
        BLOCK_M=block_size_M, BLOCK_N=block_size_N,
        BLOCK_DMODEL=Lk,
        dtype=dtype,
        num_warps=4, num_stages=2,
    )

    return o

def _build_block_index(
    q_int8, q_scale, k_int8, k_scale,
    top_k: int,
    block_size_M: int = 64,
    block_size_N: int = 64,
    ):

    batch_size, num_heads, context_size, head_dim = q_int8.shape

    pad = block_size_M - (context_size & (block_size_M - 1))
    if pad != 64:
        q_int8 = torch.nn.functional.pad(q_int8, [0, 0, 0, pad, 0, 0, 0, 0])
        k_int8 = torch.nn.functional.pad(k_int8, [0, 0, 0, pad, 0, 0, 0, 0])

    query_pool = q_int8.reshape((batch_size, num_heads, -1, block_size_M, head_dim)).mean(dim=-2, dtype=torch.float32)
    key_pool = k_int8.reshape((batch_size, num_heads, -1, block_size_N, head_dim)).mean(dim=-2, dtype=torch.float32)


    arange_M = torch.arange(query_pool.shape[-2], dtype=torch.int32, device=q_int8.device) * block_size_M
    arange_N = torch.arange(key_pool.shape[-2], dtype=torch.int32, device=k_int8.device) * block_size_N

    query_pool = query_pool * q_scale
    key_pool = key_pool * k_scale

    p_pool = torch.einsum(f'bhmk, bhnk -> bhmn', query_pool, key_pool)
    p_pool = p_pool.where(arange_M[None, None, :, None] >= arange_N[None, None, None, :], -torch.inf)
    
    top_k = min(top_k, context_size // block_size_N)
    return torch.topk(p_pool, top_k, dim=-1).indices.to(torch.int32).sort(dim=-1).values

def quant_sparse_attention(
    q_int8, q_scale, k_int8, k_scale, value,
    top_k: int,
    block_size_M: int = 64,
    block_size_N: int = 64,
    ):

    batch_size, num_heads, context_size, head_dim = q_int8.shape
    seqlens = torch.tensor([context_size], dtype=torch.int32, device=q_int8.device)
    sm_scale = head_dim ** -0.5
    block_index = _build_block_index(q_int8, q_scale, k_int8, k_scale, top_k, block_size_N, block_size_N)
    out = quant_block_sparse_attention(q_int8, q_scale, k_int8, k_scale, value, seqlens, block_index, sm_scale, block_size_M, block_size_N)
    return out[..., :context_size, :]